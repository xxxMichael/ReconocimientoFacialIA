"""
Gestor de cámara y transmisión de video para la interfaz web.
Mantiene un hilo único de captura con DirectShow, evalúa la detección facial
una vez por fotograma, ejecuta inferencia del modelo facial para control de acceso
y simulación de puerta abierta/cerrada, y almacena un búfer circular de fotogramas originales.
"""
import base64
import collections
from dataclasses import dataclass
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
    PARTICIONES_VALIDAS
)
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
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
from reconocimiento.clasificacion.regla_acceso import evaluar_decision_acceso


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
        self.lock_buffer = threading.Lock()
        self.lock_captura = threading.Lock()

        self.capturador: Optional[CapturadorCamara] = None
        self.detector = DetectorFacial()
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

        # Módulo de reconocimiento facial y simulación de puerta
        self.modelo_reconocimiento: Optional[Any] = None
        self.extractor_caracteristicas = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16, rango_intensidad=(0.0, 1.0))
        self.mapeo_clases: Dict[str, Any] = {
            "id_to_persona": {"0": "persona_1", "1": "persona_2", "2": "persona_3"},
            "persona_to_id": {"persona_1": 0, "persona_2": 1, "persona_3": 2},
            "personas_autorizadas": ["persona_1", "persona_2", "persona_3"]
        }
        self.nombre_modelo_activo: str = "ninguno"
        self.es_cnn: bool = False
        self.umbral_confianza: float = 0.65
        self.ultimo_resultado_reconocimiento: Dict[str, Any] = {
            "estado_puerta": "CERRADA",
            "persona_reconocida": "",
            "confianza": 0.0,
            "es_autorizado": False,
            "mensaje_acceso": "Puerta Cerrada (Esperando rostro)",
            "modelo_activo": "ninguno",
            "umbral": 65.0
        }

    def cargar_modelo(self, directorio_modelo: Path = Path("resultados/cnn_propio"), umbral: float = 0.65) -> Dict[str, Any]:
        """
        Carga un clasificador facial entrenado una sola vez en memoria y prepara
        el pipeline de preprocesamiento idéntico al entrenamiento.
        """
        directorio_modelo = Path(directorio_modelo)
        if not directorio_modelo.exists():
            return {"status": "error", "mensaje": f"No existe el directorio de modelo: {directorio_modelo}"}

        # Cargar mapeo de clases si existe
        ruta_mapeo = directorio_modelo / "clases_mapeo.json"
        if ruta_mapeo.exists():
            try:
                with open(ruta_mapeo, "r", encoding="utf-8") as f:
                    self.mapeo_clases = json.load(f)
            except Exception as e:
                print(f"Aviso al cargar clases_mapeo.json: {e}")

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
                "personas_autorizadas": self.mapeo_clases.get("personas_autorizadas", [])
            }
        except Exception as e:
            return {"status": "error", "mensaje": f"Fallo al cargar modelo: {e}"}

    def desactivar_modelo(self, motivo: str = "Autorización desactivada: sin modelo calibrado.") -> Dict[str, Any]:
        """
        Desactiva el clasificador facial en memoria y mantiene la simulación
        de puerta estrictamente CERRADA y la autorización inactiva.
        Permite que la captura y adquisición web sigan operando normalmente.
        """
        self.modelo_reconocimiento = None
        self.nombre_modelo_activo = "ninguno"
        self.es_cnn = False
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
        self.umbral_confianza = max(0.1, min(0.99, float(nuevo_umbral)))
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

            # 1. Detección facial sobre el fotograma ORIGINAL
            resultado_det = self.detector.detectar(frame_raw, es_pre_recortada=False)

            # 2. Inferencia, HUD y almacenamiento en búfer
            self._procesar_y_almacenar_fotograma(frame_raw, resultado_det)

            elapsed = time.time() - t_inicio
            restante = target_period - elapsed
            if restante > 0:
                time.sleep(restante)

        self._detener_sin_lock()

    def _procesar_y_almacenar_fotograma(
        self,
        frame_raw: np.ndarray,
        resultado_det: ResultadoDeteccion
    ) -> Optional[FotogramaBuffer]:
        """
        Aplica reconocimiento facial y regla de rechazo al rostro detectado,
        renderiza el HUD y estado de la puerta, y publica el fotograma en el búfer circular.
        """
        h_f, w_f = frame_raw.shape[:2]
        persona_rec = ""
        confianza_rec = 0.0
        es_aut = False
        estado_puerta = "CERRADA"
        msg_acceso = "Puerta Cerrada"
        color_recuadro = (0, 180, 255)

        if resultado_det.estado == EstadoDeteccion.UN_ROSTRO:
            if self.modelo_reconocimiento is not None and resultado_det.rostro_recortado is not None:
                try:
                    recorte = resultado_det.rostro_recortado
                    gris = a_escala_de_grises(recorte) if len(recorte.shape) == 3 else recorte
                    norm_res = normalizar_resolucion(gris, (64, 64))
                    norm_int = normalizar_intensidades(norm_res, "float32_0_1")
                    suavizada = suavizado_gaussiano(norm_int, sigma=0.5)

                    if self.es_cnn:
                        in_tensor = np.expand_dims(suavizada, axis=0)
                        probs = self.modelo_reconocimiento.predecir_probabilidades(in_tensor)[0]
                    else:
                        rep = self.extractor_caracteristicas.extraer(suavizada)
                        in_feat = np.expand_dims(rep.vector_caracteristicas, axis=0)
                        probs = self.modelo_reconocimiento.predecir_probabilidades(in_feat)[0]

                    pred_idx = int(np.argmax(probs))
                    confianza_rec = float(probs[pred_idx])
                    id_str = str(pred_idx)
                    id_to_p = self.mapeo_clases.get("id_to_persona", {})
                    nombre_pred = id_to_p.get(id_str, id_to_p.get(pred_idx, f"persona_{pred_idx}"))

                    # REGLA CANÓNICA DE DECISIÓN DE ACCESO
                    decision_acceso = evaluar_decision_acceso(
                        nombre_predicho=nombre_pred,
                        confianza=confianza_rec,
                        umbral_confianza=self.umbral_confianza,
                        personas_autorizadas=self.mapeo_clases.get("personas_autorizadas", []),
                    )
                    es_aut = decision_acceso.es_autorizado
                    estado_puerta = decision_acceso.estado_puerta
                    persona_rec = decision_acceso.persona_identificada
                    msg_acceso = decision_acceso.mensaje
                    color_recuadro = (0, 255, 0) if es_aut else (0, 140, 255)
                except Exception as e_inf:
                    msg_acceso = f"Error inferencia: {e_inf}"
                    color_recuadro = (0, 255, 255)
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
        else:  # VARIOS_ROSTROS
            color_recuadro = (0, 0, 255)
            msg_acceso = "Múltiples rostros detectados (Acceso Bloqueado)"
            estado_puerta = "CERRADA"

        self.ultimo_resultado_reconocimiento = {
            "estado_puerta": estado_puerta,
            "persona_reconocida": persona_rec,
            "confianza": round(confianza_rec * 100, 1),
            "es_autorizado": es_aut,
            "mensaje_acceso": msg_acceso,
            "modelo_activo": self.nombre_modelo_activo,
            "umbral": round(self.umbral_confianza * 100, 1)
        }

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
            f"DETECTOR: {resultado_det.estado.value} | MODELO: {self.nombre_modelo_activo} | FPS: {self.fps_efectivo:.1f}",
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
            return item

    def _reiniciar_estado_reconocimiento_bloqueado(self, razon: str = "Cámara detenida (Acceso Bloqueado)") -> None:
        """Fuerza inmediatamente el bloqueo de la puerta y resetea las variables de reconocimiento."""
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
            ultimo.persona_reconocida = ""
            ultimo.confianza = 0.0
            ultimo.mensaje_acceso = razon

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
            item = self.buffer[-1]
            # Si la cámara está detenida o sin señal, garantizar que la puerta permanezca CERRADA
            if self.capturador is None or not self.capturador.esta_activo or self.detener_evento.is_set():
                item.estado_puerta = "CERRADA"
                item.es_autorizado = False
                item.mensaje_acceso = "Cámara detenida (Acceso Bloqueado)"
            return item

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
        origen_caja: Optional[str] = None
    ) -> Dict[str, Any]:
        """Procesa una solicitud de captura asociando el id_fotograma mostrado."""
        if idempotency_key and idempotency_key in self.cache_idempotencia:
            return self.cache_idempotencia[idempotency_key]

        with self.lock_captura:
            if idempotency_key and idempotency_key in self.cache_idempotencia:
                return self.cache_idempotencia[idempotency_key]

            if not consentimiento:
                raise ValueError("Se requiere consentimiento informado explícito para registrar la captura.")

            if pose not in POSES_VALIDAS:
                raise ValueError(f"Pose inválida: '{pose}'. Válidas: {list(POSES_VALIDAS)}")

            if particion not in PARTICIONES_VALIDAS:
                raise ValueError(f"Partición inválida: '{particion}'. Válidas: {list(PARTICIONES_VALIDAS)}")

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
                dir_base = Path("data/ensayos")
                ruta_manifiesto = dir_base / "manifiesto_ensayo.csv"
                prefijo_id = "ensayo"
            else:
                dir_base = Path("data/propio")
                ruta_manifiesto = dir_base / "manifiesto.csv"
                prefijo_id = sanitizar_componente_ruta(persona)

            dir_imagenes = dir_base / "imagenes" / sanitizar_componente_ruta(persona) / sanitizar_componente_ruta(sesion) / sanitizar_componente_ruta(pose)
            dir_imagenes.mkdir(parents=True, exist_ok=True)

            manifiesto = ManifiestoDataset(ruta_manifiesto)
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

            anotacion = f"captura_web;modo={modo};frame_id={id_fotograma};estado={item.estado_deteccion.value}{caja_str}"

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
