"""
Gestor de cámara y transmisión de video para la interfaz web.
Mantiene un hilo único de captura con DirectShow, evalúa la detección facial
una vez por fotograma, ejecuta inferencia del modelo facial para control de acceso
y simulación de puerta abierta/cerrada, y almacena un búfer circular de fotogramas originales.
"""
import base64
import collections
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from reconocimiento.adquisicion.camara import (
    CapturadorCamara,
    guardar_fotograma_atomico,
    sanitizar_componente_ruta
)
from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    POSES_VALIDAS,
    PARTICIONES_VALIDAS,
    TIPOS_SUJETO_VALIDOS,
    DESTINOS_DESCONOCIDOS_VALIDOS,
    DISTANCIAS_VALIDAS,
    PERSONAS_AUTORIZADAS_PROPIO,
    MAPEO_GRUPO_PARTICION_DESCONOCIDOS,
    normalizar_identificador,
    validar_coherencia_sesion_particion_propio
)
from reconocimiento.preprocesamiento.fabrica import crear_detector
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion, ResultadoDeteccion
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_resolucion,
    normalizar_intensidades
)
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.pipeline_candidato import (
    PipelineCandidatoCongelado,
    ErrorConfiguracionCandidato,
)
from reconocimiento.clasificacion.regla_acceso import (
    es_valor_probabilidad_valido,
    evaluar_decision_acceso,
    evaluar_regla_acceso_completa,
    validar_vector_probabilidades,
)


@dataclass
class FotogramaBuffer:
    """Fotograma almacenado en el búfer circular con metadatos de detección y reconocimiento."""
    id_fotograma: int
    timestamp: float
    frame_raw_bgr: np.ndarray  # Fotograma original BGR sin anotaciones
    estado_deteccion: EstadoDeteccion
    recuadro: Optional[Tuple[int, int, int, int]]
    total_rostros: int
    frame_display_jpeg: bytes  # JPEG codificado con recuadro y HUD para la web
    ancho: int
    alto: int
    persona_reconocida: Optional[str] = None
    confianza: float = 0.0
    es_autorizado: bool = False
    estado_puerta: str = "CERRADA"
    mensaje_acceso: str = "Puerta Cerrada"


class GestorCamaraWeb:
    """
    Singleton / Gestor centralizado del flujo de video para la interfaz web.
    Garantiza que una sola instancia controle la cámara, ejecutando detección e inferencia
    una sola vez por fotograma publicado.
    """

    _instancia: Optional["GestorCamaraWeb"] = None
    _lock_singleton = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock_singleton:
            if cls._instancia is None:
                cls._instancia = super().__new__(cls)
                cls._instancia._inicializado = False
            return cls._instancia

    def __init__(self, tamano_buffer: int = 40):
        if self._inicializado:
            return
        self._inicializado = True
        self.tamano_buffer = tamano_buffer
        self.buffer: collections.deque[FotogramaBuffer] = collections.deque(maxlen=tamano_buffer)
        self.lock_buffer = threading.RLock()
        self.condicion_fotograma = threading.Condition(self.lock_buffer)
        self.lock_captura = threading.Lock()
        self.lock_modelo = threading.RLock()
        self.lock_inferencia = threading.RLock()

        self.capturador: Optional[CapturadorCamara] = None
        self.detector = crear_detector()
        self.indice_activo: Optional[int] = None

        self.hilo_trabajador: Optional[threading.Thread] = None
        self.detener_evento = threading.Event()

        self.contador_frames: int = 0
        self.fps_efectivo: float = 0.0
        self.tiempos_frames: collections.deque[float] = collections.deque(maxlen=30)
        self.senal_perdida: bool = False
        self.ultimo_error: str = ""

        # Caché de idempotencia (idempotency_key -> respuesta)
        self.cache_idempotencia: Dict[str, Dict[str, Any]] = {}

        # Directorios base de datos (configurables para aislamiento de tests)
        self.dir_propio: Path = Path("data/propio")
        self.dir_ensayos: Path = Path("data/ensayos")

        # Módulo de reconocimiento facial y simulación de puerta
        self.modelo_reconocimiento: Optional[Any] = None
        self.pipeline_candidato: Optional[PipelineCandidatoCongelado] = None
        self.extractor_caracteristicas = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16, rango_intensidad=(0.0, 1.0))
        self.mapeo_clases: Dict[str, Any] = {
            "id_to_persona": {"0": "persona_1", "1": "persona_2", "2": "persona_3"},
            "persona_to_id": {"persona_1": 0, "persona_2": 1, "persona_3": 2},
            "personas_autorizadas": ["persona_1", "persona_2", "persona_3"]
        }
        self.personas_autorizadas: List[str] = ["persona_1", "persona_2", "persona_3"]
        self.nombre_modelo_activo: str = "ninguno"
        self.es_cnn: bool = False
        self.umbral_confianza: float = 0.65
        # Indicador opcional de LED (Arduino). None = sin hardware conectado.
        self.indicador_leds: Optional[Any] = None
        self.ultimo_resultado_reconocimiento: Dict[str, Any] = {
            "estado_puerta": "CERRADA",
            "persona_reconocida": "DESCONOCIDO",
            "confianza": 0.0,
            "es_autorizado": False,
            "mensaje_acceso": "Puerta Cerrada (Esperando rostro)",
            "modelo_activo": "ninguno",
            "umbral": 65.0
        }

    def _obtener_instantanea_modelo(self) -> Dict[str, Any]:
        """Retorna una instantánea coherente del detector, preprocesamiento, modelo y umbral activos."""
        with self.lock_modelo:
            pipe = self.pipeline_candidato
            detector_activo = pipe.detector if pipe is not None else self.detector
            return {
                "pipeline_candidato": pipe,
                "detector": detector_activo,
                "modelo_reconocimiento": self.modelo_reconocimiento,
                "es_cnn": self.es_cnn,
                "umbral_confianza": self.umbral_confianza,
                "personas_autorizadas": list(self.personas_autorizadas),
                "mapeo_clases": dict(self.mapeo_clases),
                "nombre_modelo_activo": self.nombre_modelo_activo,
            }

    def cargar_modelo(self, directorio_modelo: Path = Path("resultados/cnn_propio"), umbral: float = 0.65) -> Dict[str, Any]:
        """
        Carga un clasificador facial entrenado una sola vez en memoria y prepara
        el pipeline de preprocesamiento idéntico al entrenamiento.
        """
        directorio_modelo = Path(directorio_modelo)
        if not directorio_modelo.exists():
            return {"status": "error", "mensaje": f"No existe el directorio de modelo: {directorio_modelo}"}

        with self.lock_modelo:
            self.pipeline_candidato = None
            # Cargar mapeo de clases si existe
        ruta_mapeo = directorio_modelo / "clases_mapeo.json"
        if ruta_mapeo.exists():
            try:
                with open(ruta_mapeo, "r", encoding="utf-8") as f:
                    map_data = json.load(f)
                    if "id_to_persona" in map_data:
                        self.mapeo_clases = map_data
                    else:
                        self.mapeo_clases = {"id_to_persona": map_data}
            except Exception as e:
                print(f"Aviso al cargar clases_mapeo.json: {e}")

        # Cargar personas autorizadas si existe archivo separado
        ruta_aut = directorio_modelo / "personas_autorizadas.json"
        if ruta_aut.exists():
            try:
                with open(ruta_aut, "r", encoding="utf-8") as f:
                    self.personas_autorizadas = json.load(f)
            except Exception as e:
                print(f"Aviso al cargar personas_autorizadas.json: {e}")
        elif "personas_autorizadas" in self.mapeo_clases:
            self.personas_autorizadas = list(self.mapeo_clases["personas_autorizadas"])

        self.umbral_confianza = float(umbral)
        nombre_dir = directorio_modelo.name.lower()

        try:
            if (directorio_modelo / "modelo.pt").exists() and ("cnn" in nombre_dir):
                self.modelo_reconocimiento = ClasificadorCNN.cargar(directorio_modelo / "modelo.pt")
                self.es_cnn = True
                self.nombre_modelo_activo = "CNN (Red Convolucional)"
            elif (directorio_modelo / "modelo.pt").exists() and ("ann" in nombre_dir):
                self.modelo_reconocimiento = ClasificadorANN.cargar(directorio_modelo / "modelo.pt")
                self.es_cnn = False
                self.nombre_modelo_activo = "ANN (Red Densa)"
            elif (directorio_modelo / "modelo.joblib").exists() and ("svm" in nombre_dir):
                self.modelo_reconocimiento = ClasificadorSVM.cargar(directorio_modelo / "modelo.joblib")
                self.es_cnn = False
                self.nombre_modelo_activo = "SVM (Máquina Vectores Soporte)"
            elif (directorio_modelo / "modelo.joblib").exists() and ("arbol" in nombre_dir):
                self.modelo_reconocimiento = ClasificadorArbol.cargar(directorio_modelo / "modelo.joblib")
                self.es_cnn = False
                self.nombre_modelo_activo = "Árbol de Decisión"
            elif ("knn" in nombre_dir):
                if (directorio_modelo / "modelo.joblib").exists():
                    self.modelo_reconocimiento = ClasificadorKNN.cargar(directorio_modelo / "modelo.joblib")
                    self.es_cnn = False
                    self.nombre_modelo_activo = "KNN (K-Nearest Neighbors)"
                else:
                    raise FileNotFoundError(
                        f"El clasificador KNN en '{directorio_modelo}' no contiene 'modelo.joblib' "
                        f"con los ejemplos de entrenamiento restaurables."
                    )
            else:
                # Detección heurística por archivos
                if (directorio_modelo / "modelo.pt").exists():
                    self.modelo_reconocimiento = ClasificadorCNN.cargar(directorio_modelo / "modelo.pt")
                    self.es_cnn = True
                    self.nombre_modelo_activo = "CNN"
                elif (directorio_modelo / "modelo.joblib").exists():
                    self.modelo_reconocimiento = ClasificadorSVM.cargar(directorio_modelo / "modelo.joblib")
                    self.es_cnn = False
                    self.nombre_modelo_activo = "SVM"

            self.ultimo_resultado_reconocimiento["umbral"] = round(self.umbral_confianza * 100, 1)
            self.ultimo_resultado_reconocimiento["modelo_activo"] = self.nombre_modelo_activo

            return {
                "status": "ok",
                "mensaje": f"Modelo '{self.nombre_modelo_activo}' cargado exitosamente.",
                "modelo_activo": self.nombre_modelo_activo,
                "es_cnn": self.es_cnn,
                "umbral": round(self.umbral_confianza * 100, 1),
                "umbral_decimal": round(self.umbral_confianza, 4),
                "personas_autorizadas": self.personas_autorizadas
            }
        except Exception as e:
            return {"status": "error", "mensaje": f"Fallo al cargar modelo: {e}"}

    def cargar_desde_configuracion_congelada(self, ruta_config: Path) -> Dict[str, Any]:
        """
        Carga explícitamente el modelo y parámetros del candidato desde un archivo
        de configuración congelada utilizando el pipeline canónico unificado,
        validando hashes SHA-256 de pesos y detector, y verificando coherencia de clases.
        """
        ruta_config = Path(ruta_config)
        if not ruta_config.exists():
            self._reiniciar_estado_reconocimiento_bloqueado(f"No existe el archivo de configuración: {ruta_config}")
            return {"status": "error", "mensaje": f"No existe el archivo de configuración: {ruta_config}"}

        try:
            pipeline = PipelineCandidatoCongelado(ruta_config)
            with self.lock_modelo:
                self.pipeline_candidato = pipeline
                self.modelo_reconocimiento = pipeline.clasificador
                self.detector = pipeline.detector
                self.es_cnn = True
                self.umbral_confianza = pipeline.umbral_score
                self.personas_autorizadas = list(pipeline.personas_autorizadas)
                self.mapeo_clases = {
                    "id_to_persona": {str(k): v for k, v in pipeline.mapeo_id_clase.items()},
                    "personas_autorizadas": list(pipeline.personas_autorizadas),
                }
                self.nombre_modelo_activo = f"{pipeline.nombre} (Candidato Congelado, th={self.umbral_confianza:.2f})"

            with self.lock_buffer:
                self.ultimo_resultado_reconocimiento = {
                    "estado_puerta": "CERRADA",
                    "persona_reconocida": "",
                    "confianza": 0.0,
                    "es_autorizado": False,
                    "mensaje_acceso": "Puerta Cerrada (Candidato Congelado Cargado)",
                    "modelo_activo": self.nombre_modelo_activo,
                    "umbral": round(self.umbral_confianza * 100, 1),
                }

            return {
                "status": "ok",
                "mensaje": f"Candidato congelado '{self.nombre_modelo_activo}' cargado exitosamente.",
                "modelo_activo": self.nombre_modelo_activo,
                "es_cnn": True,
                "umbral": round(self.umbral_confianza * 100, 1),
                "umbral_decimal": round(self.umbral_confianza, 4),
                "personas_autorizadas": self.personas_autorizadas,
                "sha256": pipeline.sha256_modelo,
            }
        except Exception as e:
            with self.lock_modelo:
                self.modelo_reconocimiento = None
                self.pipeline_candidato = None
                self.es_cnn = False
                self.nombre_modelo_activo = "ninguno"
            self._reiniciar_estado_reconocimiento_bloqueado(f"Fallo al cargar candidato congelado: {e}")
            return {"status": "error", "mensaje": f"Error al cargar candidato congelado: {e}"}

    def desactivar_modelo(self, motivo: str = "Autorización desactivada: sin modelo calibrado.") -> Dict[str, Any]:
        """
        Desactiva el clasificador facial en memoria y mantiene la simulación
        de puerta estrictamente CERRADA y la autorización inactiva.
        Permite que la captura y adquisición web sigan operando normalmente.
        """
        with self.lock_modelo:
            self.modelo_reconocimiento = None
            self.pipeline_candidato = None
            self.nombre_modelo_activo = "ninguno"
            self.es_cnn = False
        with self.lock_buffer:
            self.ultimo_resultado_reconocimiento = {
                "estado_puerta": "CERRADA",
                "persona_reconocida": "",
                "confianza": 0.0,
                "es_autorizado": False,
                "mensaje_acceso": motivo,
                "modelo_activo": "ninguno",
                "umbral": round(self.umbral_confianza * 100, 1)
            }
        return {"status": "ok", "mensaje": motivo, "modelo_activo": "ninguno"}

    def actualizar_umbral(self, nuevo_umbral: float) -> None:
        """Actualiza el umbral de confianza para la regla de rechazo."""
        with self.lock_modelo:
            self.umbral_confianza = max(0.1, min(0.99, float(nuevo_umbral)))
            if self.pipeline_candidato is not None:
                self.pipeline_candidato.umbral_score = self.umbral_confianza
            with self.lock_buffer:
                self.ultimo_resultado_reconocimiento["umbral"] = round(self.umbral_confianza * 100, 1)

    def iniciar(self, indice: int = 1, fuente_custom: Any = None) -> bool:
        """Inicia el hilo de captura en el índice especificado si no está ya activo."""
        with self.lock_buffer:
            if self.capturador is not None and self.capturador.esta_activo and self.indice_activo == indice:
                return True

            if self.capturador is not None:
                self._detener_sin_lock()

            self.indice_activo = indice
            self.detener_evento.clear()
            self.senal_perdida = False
            self.ultimo_error = ""

            self.capturador = CapturadorCamara(
                indice=indice,
                backend=cv2.CAP_DSHOW,
                intentos_arranque=4,
                max_fallos_consecutivos=8,
                fuente_custom=fuente_custom
            )

            try:
                self.capturador.abrir()
            except Exception as e:
                self.ultimo_error = str(e)
                self.capturador = None
                self.indice_activo = None
                raise e

            self.hilo_trabajador = threading.Thread(
                target=self._bucle_captura,
                name="HiloCamaraWeb",
                daemon=True
            )
            self.hilo_trabajador.start()
            return True

    def _bucle_captura(self) -> None:
        """Hilo de fondo que lee a 12-15 FPS, detecta rostros, clasifica y actualiza el búfer."""
        target_period = 1.0 / 14.0  # ~14 FPS

        while not self.detener_evento.is_set():
            t_inicio = time.time()
            if self.capturador is None:
                break

            try:
                ret, frame_raw = self.capturador.leer()
            except Exception as e:
                self.senal_perdida = True
                self.ultimo_error = f"Pérdida de señal de video: {e}"
                self._reiniciar_estado_reconocimiento_bloqueado(f"Pérdida de señal: {e}")
                break

            if not ret or frame_raw is None or frame_raw.size == 0:
                time.sleep(0.04)
                continue

            self.senal_perdida = False
            h_f, w_f = frame_raw.shape[:2]

            # 1 y 2. Instantánea coherente de detector, preprocesamiento, modelo y umbral
            snap = self._obtener_instantanea_modelo()
            with self.lock_inferencia:
                resultado_det = snap["detector"].detectar(frame_raw, es_pre_recortada=False)
                self._procesar_y_almacenar_fotograma(frame_raw, resultado_det, snap=snap)

            elapsed = time.time() - t_inicio
            restante = target_period - elapsed
            if restante > 0:
                time.sleep(restante)

        self._detener_sin_lock()

    def _procesar_y_almacenar_fotograma(
        self,
        frame_raw: np.ndarray,
        resultado_det: ResultadoDeteccion,
        snap: Optional[Dict[str, Any]] = None,
    ) -> Optional[FotogramaBuffer]:
        """
        Aplica reconocimiento facial y regla de rechazo al rostro detectado,
        renderiza el HUD y estado de la puerta, y publica el fotograma en el búfer circular.
        """
        if snap is None:
            snap = self._obtener_instantanea_modelo()
        pipe_cand = snap["pipeline_candidato"]
        modelo_rec = snap["modelo_reconocimiento"]
        es_cnn_snap = snap["es_cnn"]
        umbral_snap = snap["umbral_confianza"]
        aut_snap = snap["personas_autorizadas"]
        mapeo_snap = snap["mapeo_clases"]
        nombre_mod_snap = snap["nombre_modelo_activo"]

        h_f, w_f = frame_raw.shape[:2]
        persona_rec = "DESCONOCIDO"
        confianza_rec = 0.0
        es_aut = False
        estado_puerta = "CERRADA"
        msg_acceso = "Puerta Cerrada"
        color_recuadro = (0, 180, 255)

        if resultado_det.estado == EstadoDeteccion.UN_ROSTRO:
            if modelo_rec is not None and resultado_det.rostro_recortado is not None:
                try:
                    usar_alin = getattr(pipe_cand, "usar_alineacion", False) if pipe_cand else False
                    recorte = (
                        resultado_det.recorte_alineado
                        if (usar_alin and resultado_det.recorte_alineado is not None)
                        else resultado_det.rostro_recortado
                    )
                    if pipe_cand is not None:
                        decision_acceso, probs = pipe_cand.evaluar_recorte(recorte)
                        pred_idx = int(np.argmax(probs))
                        confianza_rec = float(probs[pred_idx])
                    else:
                        gris = a_escala_de_grises(recorte) if len(recorte.shape) == 3 else recorte
                        norm_res = normalizar_resolucion(gris, (64, 64))
                        norm_int = normalizar_intensidades(norm_res, "float32_0_1")
                        suavizada = suavizado_gaussiano(norm_int, sigma=0.5)

                        if es_cnn_snap:
                            in_tensor = np.expand_dims(suavizada, axis=0)
                            probs = modelo_rec.predecir_probabilidades(in_tensor)[0]
                        else:
                            rep = self.extractor_caracteristicas.extraer(suavizada)
                            in_feat = np.expand_dims(rep.vector_caracteristicas, axis=0)
                            probs = modelo_rec.predecir_probabilidades(in_feat)[0]

                        valido, mot = validar_vector_probabilidades(probs, n_clases_esperadas=len(probs))
                        if not valido:
                            raise ValueError(f"Probabilidades de salida inválidas: {mot}")

                        pred_idx = int(np.argmax(probs))
                        confianza_rec = float(probs[pred_idx])
                        id_str = str(pred_idx)
                        id_to_p = mapeo_snap.get("id_to_persona", mapeo_snap)
                        nombre_pred = id_to_p.get(id_str, id_to_p.get(pred_idx, f"persona_{pred_idx}"))
                        aut_list = aut_snap or mapeo_snap.get("personas_autorizadas", ["persona_1", "persona_2", "persona_3"])

                        # REGLA CANÓNICA DE DECISIÓN DE ACCESO
                        decision_acceso = evaluar_regla_acceso_completa(
                            estado_deteccion="UN_ROSTRO",
                            procesamiento_exitoso=True,
                            clase_predicha=nombre_pred,
                            score=confianza_rec,
                            umbral=umbral_snap,
                            personas_autorizadas=aut_list,
                        )

                    es_aut = decision_acceso.es_autorizado
                    estado_puerta = decision_acceso.estado_puerta
                    persona_rec = decision_acceso.persona_identificada
                    msg_acceso = decision_acceso.mensaje
                    color_recuadro = (0, 255, 0) if es_aut else (0, 140, 255)
                except Exception as e_inf:
                    aut_list = aut_snap or mapeo_snap.get("personas_autorizadas", ["persona_1", "persona_2", "persona_3"])
                    decision_acceso = evaluar_regla_acceso_completa(
                        estado_deteccion="UN_ROSTRO",
                        procesamiento_exitoso=False,
                        clase_predicha=None,
                        score=0.0,
                        umbral=umbral_snap,
                        personas_autorizadas=aut_list,
                        error_procesamiento=str(e_inf),
                    )
                    msg_acceso = decision_acceso.mensaje
                    estado_puerta = decision_acceso.estado_puerta
                    es_aut = False
                    persona_rec = "DESCONOCIDO"
                    confianza_rec = 0.0
                    color_recuadro = (0, 0, 255)
            else:
                color_recuadro = (0, 180, 255)
                msg_acceso = "Rostro detectado (Autorización desactivada: sin modelo calibrado)"
                estado_puerta = "CERRADA"
                es_aut = False
                persona_rec = ""
                confianza_rec = 0.0
        elif resultado_det.estado == EstadoDeteccion.SIN_ROSTRO:
            color_recuadro = (0, 180, 255)
            msg_acceso = "Esperando persona..."
            estado_puerta = "CERRADA"
            es_aut = False
            persona_rec = ""
            confianza_rec = 0.0
        else:  # VARIOS_ROSTROS
            color_recuadro = (0, 0, 255)
            msg_acceso = "Múltiples rostros detectados (Acceso Bloqueado)"
            estado_puerta = "CERRADA"
            es_aut = False
            persona_rec = ""
            confianza_rec = 0.0

        with self.lock_buffer:
            self.ultimo_resultado_reconocimiento = {
                "estado_puerta": estado_puerta,
                "persona_reconocida": persona_rec,
                "confianza": round(confianza_rec * 100, 1),
                "es_autorizado": es_aut,
                "mensaje_acceso": msg_acceso,
                "modelo_activo": nombre_mod_snap,
                "umbral": round(umbral_snap * 100, 1)
            }

        # LED del Arduino: solo se enciende el de la persona autorizada cuya puerta está ABIERTA
        if self.indicador_leds is not None:
            self.indicador_leds.actualizar(persona_rec if (estado_puerta == "ABIERTA" and es_aut) else None)

        # 3. Dibujar visualización con HUD y estado de Puerta
        display = frame_raw.copy()
        if resultado_det.estado == EstadoDeteccion.UN_ROSTRO and resultado_det.recuadro is not None:
            rx, ry, rw, rh = resultado_det.recuadro
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), color_recuadro, 2)
            txt_label = f"{persona_rec} [{confianza_rec*100:.1f}%]" if persona_rec else f"UN_ROSTRO [{rw}x{rh}]"
            cv2.putText(
                display,
                txt_label,
                (rx, max(22, ry - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_recuadro,
                2,
                cv2.LINE_AA
            )

        # Barra superior con estado de cámara y FPS
        cv2.rectangle(display, (0, 0), (w_f, 26), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"DETECTOR YuNet: {resultado_det.estado.value} | MODELO: {self.nombre_modelo_activo} | FPS: {self.fps_efectivo:.1f}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color_recuadro,
            1,
            cv2.LINE_AA
        )

        # Banner inferior de Simulación de Puerta
        color_banner = (0, 160, 0) if estado_puerta == "ABIERTA" else (0, 0, 160)
        cv2.rectangle(display, (0, h_f - 32), (w_f, h_f), color_banner, -1)
        cv2.putText(
            display,
            f"PUERTA: {estado_puerta} | {msg_acceso}",
            (10, h_f - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        ret_enc, jpeg_bytes = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_data = jpeg_bytes.tobytes() if ret_enc else b""

        with self.lock_buffer:
            self.contador_frames += 1
            fid = self.contador_frames
            ahora = time.time()
            self.tiempos_frames.append(ahora)

            if len(self.tiempos_frames) > 1:
                delta = self.tiempos_frames[-1] - self.tiempos_frames[0]
                if delta > 0:
                    self.fps_efectivo = round((len(self.tiempos_frames) - 1) / delta, 1)

            item = FotogramaBuffer(
                id_fotograma=fid,
                timestamp=ahora,
                frame_raw_bgr=frame_raw,
                estado_deteccion=resultado_det.estado,
                recuadro=resultado_det.recuadro,
                total_rostros=resultado_det.total_rostros_detectados,
                frame_display_jpeg=jpeg_data,
                ancho=w_f,
                alto=h_f,
                persona_reconocida=persona_rec,
                confianza=round(confianza_rec * 100, 1),
                es_autorizado=es_aut,
                estado_puerta=estado_puerta,
                mensaje_acceso=msg_acceso
            )
            self.buffer.append(item)
            self.condicion_fotograma.notify_all()
            return item

    def procesar_fotograma_para_reconocimiento(
        self,
        frame_raw: np.ndarray,
        resultado_det: Optional[ResultadoDeteccion] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta síncronamente la detección (si no se proporciona) e inferencia
        sobre un fotograma, actualizando de forma atómica el estado de reconocimiento y puerta.
        Retorna el diccionario con el resultado de reconocimiento actualizado.
        """
        snap = self._obtener_instantanea_modelo()
        with self.lock_inferencia:
            if resultado_det is None:
                resultado_det = snap["detector"].detectar(frame_raw, es_pre_recortada=False)
            self._procesar_y_almacenar_fotograma(frame_raw, resultado_det, snap=snap)
        with self.lock_buffer:
            return dict(self.ultimo_resultado_reconocimiento)

    def _reiniciar_estado_reconocimiento_bloqueado(self, razon: str = "Cámara detenida (Acceso Bloqueado)") -> None:
        """Fuerza inmediatamente el bloqueo de la puerta y resetea las variables de reconocimiento."""
        with self.lock_buffer:
            self.ultimo_resultado_reconocimiento = {
                "estado_puerta": "CERRADA",
                "persona_reconocida": "",
                "confianza": 0.0,
                "es_autorizado": False,
                "mensaje_acceso": razon,
                "modelo_activo": self.nombre_modelo_activo,
                "umbral": round(self.umbral_confianza * 100, 1)
            }
            if self.buffer:
                ultimo = self.buffer[-1]
                ultimo.estado_puerta = "CERRADA"
                ultimo.es_autorizado = False
                ultimo.persona_reconocida = "DESCONOCIDO"
                ultimo.confianza = 0.0
                ultimo.mensaje_acceso = razon
            self.condicion_fotograma.notify_all()
        if self.indicador_leds is not None:
            self.indicador_leds.apagar()

    def _detener_sin_lock(self) -> None:
        """Detiene y libera recursos (debe llamarse con control de hilo)."""
        self.detener_evento.set()
        if self.capturador is not None:
            self.capturador.liberar()
            self.capturador = None
        self.indice_activo = None
        self._reiniciar_estado_reconocimiento_bloqueado("Cámara detenida (Acceso Bloqueado)")

    def detener(self) -> None:
        """Detiene limpiamente el capturador y libera el hardware de la cámara."""
        with self.lock_buffer:
            self.detener_evento.set()
            if self.capturador is not None:
                self.capturador.liberar()
                self.capturador = None
            self.indice_activo = None
            self._reiniciar_estado_reconocimiento_bloqueado("Cámara detenida (Acceso Bloqueado)")

    def obtener_ultimo_fotograma(self) -> Optional[FotogramaBuffer]:
        """Retorna el fotograma más reciente publicado en el búfer."""
        with self.lock_buffer:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def esperar_nuevo_fotograma(self, id_anterior: Optional[int] = None, timeout: float = 1.0) -> Optional[FotogramaBuffer]:
        """Espera de forma sincronizada hasta que se procese y publique un fotograma nuevo en el búfer."""
        with self.condicion_fotograma:
            limite = time.time() + timeout
            while time.time() < limite:
                if self.buffer:
                    ultimo = self.buffer[-1]
                    if id_anterior is None or ultimo.id_fotograma > id_anterior:
                        return ultimo
                restante = limite - time.time()
                if restante <= 0:
                    break
                self.condicion_fotograma.wait(timeout=min(0.05, restante))
            return self.buffer[-1] if self.buffer else None

    def obtener_fotograma_por_id(self, id_fotograma: int) -> Optional[FotogramaBuffer]:
        """Busca y recupera un fotograma específico por su identificador en el búfer."""
        with self.lock_buffer:
            for item in reversed(self.buffer):
                if item.id_fotograma == id_fotograma:
                    return item
        return None

    def obtener_estado(self) -> Dict[str, Any]:
        """Retorna el estado de conexión, salud del flujo y reconocimiento facial."""
        with self.lock_buffer:
            activo = (self.capturador is not None and self.capturador.esta_activo and not self.senal_perdida and not self.detener_evento.is_set())
            if not activo:
                self.ultimo_resultado_reconocimiento["estado_puerta"] = "CERRADA"
                self.ultimo_resultado_reconocimiento["es_autorizado"] = False
                self.ultimo_resultado_reconocimiento["persona_reconocida"] = "DESCONOCIDO"
                self.ultimo_resultado_reconocimiento["confianza"] = 0.0
                self.ultimo_resultado_reconocimiento["mensaje_acceso"] = "Cámara detenida (Acceso Bloqueado)"
            return {
                "activa": activo,
                "indice": self.indice_activo,
                "fps_efectivo": self.fps_efectivo,
                "total_fotogramas": self.contador_frames,
                "senal_perdida": self.senal_perdida,
                "ultimo_error": self.ultimo_error,
                "reconocimiento": self.ultimo_resultado_reconocimiento
            }

    def procesar_captura_web(
        self,
        id_fotograma: int,
        modo: str,
        persona: str,
        tipo_sujeto: str,
        sesion: str,
        pose: str,
        particion: str,
        consentimiento: bool,
        es_ensayo: bool,
        idempotency_key: str = "",
        caja: Optional[List[int]] = None,
        decision: Optional[str] = None,
        origen_caja: Optional[str] = None,
        grupo: Optional[str] = None,
        distancia: Optional[str] = None
    ) -> Dict[str, Any]:
        """Procesa una solicitud de captura asociando el id_fotograma mostrado."""
        if idempotency_key and idempotency_key in self.cache_idempotencia:
            return self.cache_idempotencia[idempotency_key]

        with self.lock_captura:
            if idempotency_key and idempotency_key in self.cache_idempotencia:
                return self.cache_idempotencia[idempotency_key]

            if not consentimiento:
                raise ValueError("Se requiere consentimiento informado explícito para registrar la captura.")

            persona = normalizar_identificador(persona)
            if not persona:
                raise ValueError("El identificador de la persona no puede estar vacío.")

            if pose not in POSES_VALIDAS:
                raise ValueError(f"Pose inválida: '{pose}'. Válidas: {sorted(POSES_VALIDAS)}")

            if particion not in PARTICIONES_VALIDAS:
                raise ValueError(f"Partición inválida: '{particion}'. Válidas: {sorted(PARTICIONES_VALIDAS)}")

            if tipo_sujeto not in TIPOS_SUJETO_VALIDOS:
                raise ValueError(f"Tipo de sujeto '{tipo_sujeto}' inválido. Válidos: {sorted(TIPOS_SUJETO_VALIDOS)}")

            if tipo_sujeto == "desconocido":
                if persona in PERSONAS_AUTORIZADAS_PROPIO:
                    raise ValueError(
                        f"Conflicto de rol: La identidad '{persona}' es una persona autorizada y no puede registrarse como desconocido."
                    )
                if not grupo:
                    raise ValueError("El grupo experimental es obligatorio para sujetos desconocidos.")
                grupo = normalizar_identificador(grupo)
                if grupo not in DESTINOS_DESCONOCIDOS_VALIDOS:
                    raise ValueError(
                        f"Grupo experimental '{grupo}' inválido. Opciones válidas: {sorted(DESTINOS_DESCONOCIDOS_VALIDOS)}"
                    )
                particion_esperada = MAPEO_GRUPO_PARTICION_DESCONOCIDOS[grupo]
                if particion != particion_esperada:
                    raise ValueError(
                        f"Inconsistencia de partición: El grupo '{grupo}' debe corresponder a la partición '{particion_esperada}', pero se especificó '{particion}'."
                    )
                if not distancia:
                    raise ValueError("La distancia de captura es obligatoria para sujetos desconocidos.")
                distancia = normalizar_identificador(distancia)
                if distancia not in DISTANCIAS_VALIDAS:
                    raise ValueError(
                        f"Distancia '{distancia}' inválida. Opciones válidas: {sorted(DISTANCIAS_VALIDAS)}"
                    )
            elif tipo_sujeto == "autorizado":
                if not es_ensayo:
                    es_val_proto, err_proto = validar_coherencia_sesion_particion_propio(sesion, particion)
                    if not es_val_proto:
                        raise ValueError(err_proto)
                if grupo and grupo in DESTINOS_DESCONOCIDOS_VALIDOS:
                    raise ValueError(f"Conflicto de rol: Un sujeto autorizado no puede asignarse a un grupo de desconocidos '{grupo}'.")
                if distancia and distancia not in DISTANCIAS_VALIDAS:
                    raise ValueError(f"Distancia '{distancia}' inválida. Opciones válidas: {sorted(DISTANCIAS_VALIDAS)}")

            item = self.obtener_fotograma_por_id(id_fotograma)
            if item is None:
                raise KeyError(
                    f"El fotograma #{id_fotograma} ya ha caducado en el búfer. "
                    "Por favor, mantenga la pose y vuelva a presionar Capturar."
                )

            if modo == "valida":
                # Si el usuario corrigió manualmente el recuadro, permitimos guardar aunque el detector automático hubiera dudado
                if caja is None and item.estado_deteccion != EstadoDeteccion.UN_ROSTRO:
                    raise ValueError(
                        f"No se puede guardar captura válida: Se requiere exactamente UN_ROSTRO, pero se detectó {item.estado_deteccion.value}."
                    )
            elif modo == "fallo":
                if caja is None and item.estado_deteccion == EstadoDeteccion.UN_ROSTRO:
                    raise ValueError(
                        "No se puede registrar como fallo de detección porque el fotograma contiene un rostro válido (UN_ROSTRO)."
                    )
            else:
                raise ValueError(f"Modo de captura desconocido: '{modo}'.")

            if es_ensayo:
                dir_base = self.dir_ensayos
                ruta_manifiesto = dir_base / "manifiesto_ensayo.csv"
                prefijo_id = "ensayo"
            else:
                dir_base = self.dir_propio
                ruta_manifiesto = dir_base / "manifiesto.csv"
                prefijo_id = sanitizar_componente_ruta(persona)

            manifiesto = ManifiestoDataset(ruta_manifiesto)

            # Validación preventiva contra el manifiesto existente antes de escribir cualquier archivo en disco
            if tipo_sujeto == "desconocido":
                for r in manifiesto.registros:
                    if r.tipo_sujeto == "desconocido" and r.persona == persona:
                        dest_existente = r.destino_experimental or r.particion
                        if dest_existente != grupo:
                            raise ValueError(
                                f"Fuga de identidad externa: El voluntario desconocido '{persona}' ya está asignado "
                                f"al grupo '{dest_existente}' y no puede registrarse en '{grupo}'. "
                                f"Cada voluntario desconocido debe pertenecer a un único grupo experimental."
                            )
                    if r.tipo_sujeto == "autorizado" and r.persona == persona:
                        raise ValueError(
                            f"Conflicto de identidad: '{persona}' ya existe como persona autorizada y no puede añadirse como desconocido."
                        )
            elif tipo_sujeto == "autorizado":
                for r in manifiesto.registros:
                    if r.tipo_sujeto == "desconocido" and r.persona == persona:
                        raise ValueError(
                            f"Conflicto de identidad: '{persona}' ya existe como sujeto desconocido y no puede añadirse como autorizado."
                        )

            dir_imagenes = dir_base / "imagenes" / sanitizar_componente_ruta(persona) / sanitizar_componente_ruta(sesion) / sanitizar_componente_ruta(pose)
            dir_imagenes.mkdir(parents=True, exist_ok=True)

            ids_existentes = manifiesto.identificadores_existentes()

            fotos_existentes = list(dir_imagenes.glob("*.jpg"))
            num_siguiente = len(fotos_existentes) + 1
            identificador = f"{prefijo_id}_{sanitizar_componente_ruta(sesion)}_{sanitizar_componente_ruta(pose)}_{num_siguiente:02d}"
            nombre_archivo = f"{prefijo_id}_{sanitizar_componente_ruta(sesion)}_{sanitizar_componente_ruta(pose)}_f{num_siguiente:02d}.jpg"
            ruta_destino = dir_imagenes / nombre_archivo

            while identificador in ids_existentes or ruta_destino.exists():
                num_siguiente += 1
                identificador = f"{prefijo_id}_{sanitizar_componente_ruta(sesion)}_{sanitizar_componente_ruta(pose)}_{num_siguiente:02d}"
                nombre_archivo = f"{prefijo_id}_{sanitizar_componente_ruta(sesion)}_{sanitizar_componente_ruta(pose)}_f{num_siguiente:02d}.jpg"
                ruta_destino = dir_imagenes / nombre_archivo

            caja_a_guardar = tuple(caja) if caja is not None else item.recuadro
            if caja_a_guardar is not None:
                orig_caja = origen_caja or ("manual" if caja is not None and caja != item.recuadro else "vivo")
                dec_humana = decision or ("corregida" if orig_caja == "manual" else ("aprobada" if modo == "valida" else "pendiente"))
                caja_str = f";caja={int(caja_a_guardar[0])},{int(caja_a_guardar[1])},{int(caja_a_guardar[2])},{int(caja_a_guardar[3])};origen_caja={orig_caja};decision={dec_humana}"
            else:
                dec_humana = decision or ("aprobada" if modo == "valida" else "pendiente")
                caja_str = f";decision={dec_humana}"

            anot_meta = ""
            if grupo:
                anot_meta += f";destino={grupo}"
            if distancia:
                anot_meta += f";distancia={distancia}"

            anotacion = f"captura_web;modo={modo};frame_id={id_fotograma};estado={item.estado_deteccion.value}{caja_str}{anot_meta}"

            registro = RegistroCaptura(
                identificador=identificador,
                ruta=str(ruta_destino).replace("\\", "/"),
                procedencia="camara_web",
                persona=persona,
                tipo_sujeto=tipo_sujeto,
                sesion=sesion,
                pose=pose,
                particion=particion,
                consentimiento=consentimiento,
                anotaciones=anotacion
            )

            # Guardado atómico con verificación de aislamiento y duplicados
            guardar_fotograma_atomico(item.frame_raw_bgr, ruta_destino, registro, manifiesto)

            progreso = calcular_progreso_sesion(manifiesto, persona, sesion)

            respuesta = {
                "status": "ok",
                "modo": modo,
                "identificador": identificador,
                "ruta": str(ruta_destino).replace("\\", "/"),
                "ruta_guardada": str(ruta_destino).replace("\\", "/"),
                "id_fotograma": id_fotograma,
                "estado_deteccion": item.estado_deteccion.value,
                "recuadro": caja_a_guardar,
                "decision": dec_humana,
                "progreso_sesion": progreso,
                "mensaje": f"Captura ({modo}) guardada exitosamente como {identificador}."
            }

            if idempotency_key:
                self.cache_idempotencia[idempotency_key] = respuesta

            return respuesta


def calcular_progreso_sesion(
    manifiesto: ManifiestoDataset,
    persona: str,
    sesion: str,
    meta_por_pose: int = 4,
    verificar_disco: bool = False
) -> Dict[str, Any]:
    """Calcula las capturas válidas y fallos por cada pose para una persona y sesión específica."""
    validas_por_pose = {p: 0 for p in sorted(list(POSES_VALIDAS))}
    fallos_por_pose = {p: 0 for p in sorted(list(POSES_VALIDAS))}
    total_validas = 0
    total_fallos = 0

    for reg in manifiesto.registros:
        if reg.persona == persona and reg.sesion == sesion:
            if verificar_disco and not Path(reg.ruta).is_file():
                continue

            pose = reg.pose if reg.pose in validas_por_pose else "frontal"
            es_fallo = ("modo=fallo" in reg.anotaciones) or ("forzado=si" in reg.anotaciones)
            if es_fallo:
                fallos_por_pose[pose] += 1
                total_fallos += 1
            else:
                validas_por_pose[pose] += 1
                total_validas += 1

    detalles_poses = {}
    for pose in sorted(list(POSES_VALIDAS)):
        v = validas_por_pose[pose]
        f = fallos_por_pose[pose]
        detalles_poses[pose] = {
            "validas": v,
            "fallos": f,
            "guardadas": v,
            "meta": meta_por_pose,
            "faltantes": max(0, meta_por_pose - v),
            "completa": (v >= meta_por_pose),
            "completada": (v >= meta_por_pose)
        }

    total_meta = len(validas_por_pose) * meta_por_pose
    sesion_completa = all(v["completada"] for v in detalles_poses.values())
    pct = round((total_validas / total_meta) * 100.0, 1) if total_meta > 0 else 0.0

    return {
        "persona": persona,
        "sesion": sesion,
        "poses": detalles_poses,
        "total_validas": total_validas,
        "total_fallos": total_fallos,
        "fallos_deliberados": total_fallos,
        "meta_total": total_meta,
        "porcentaje_completado": min(100.0, pct),
        "sesion_completa": sesion_completa
    }
