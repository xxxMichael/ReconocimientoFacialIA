from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.regla_acceso import (
    DecisionAcceso,
    ETIQUETAS_DESCONOCIDO_RESERVADAS,
    es_valor_probabilidad_valido,
    evaluar_decision_acceso,
    evaluar_regla_acceso_completa,
    validar_vector_probabilidades,
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from reconocimiento.preprocesamiento.calidad import ConfiguracionCalidad, evaluar_calidad_rostro
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion,
)


def calcular_sha256_archivo(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en disco."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


class ErrorConfiguracionCandidato(Exception):
    """Excepción alzada cuando la configuración congelada es inválida o viola restricciones de seguridad."""
    pass


class PipelineCandidatoCongelado:
    """
    Pipeline unificado de extremo a extremo para el candidato congelado de 4 clases.
    """

    def __init__(self, ruta_config: Path, raiz_proyecto: Optional[Path] = None):
        self.ruta_config = Path(ruta_config).resolve()
        self.raiz_proyecto = Path(raiz_proyecto).resolve() if raiz_proyecto else Path(__file__).resolve().parents[3]
        
        # Validar y cargar configuración
        self.config = self._cargar_y_validar_configuracion(self.ruta_config)
        
        # Parámetros del candidato
        c = self.config["candidato"]
        self.identificador: str = c["identificador"]
        self.nombre: str = c["nombre"]
        self.umbral_score: float = float(c["umbral_score"])
        self.orden_clases: List[str] = list(c["orden_clases"])
        self.mapeo_id_clase: Dict[int, str] = {int(k): v for k, v in c["mapeo_clases"].items()}
        self.personas_autorizadas: List[str] = list(c["personas_autorizadas"])
        self.ruta_modelo = self._resolver_ruta(c["ruta_modelo"])
        self.sha256_modelo = c["sha256_modelo"].lower()
        
        # Parámetros del detector YuNet
        det_cfg = self.config["detector_facial"]
        self.ruta_detector = self._resolver_ruta(det_cfg["ruta_modelo"])
        self.sha256_detector = det_cfg["sha256_modelo"].lower()
        self.params_detector = det_cfg["parametros"]
        
        # Parámetros de preprocesamiento
        prep_cfg = self.config["preprocesamiento"]
        self.resolucion: Tuple[int, int] = tuple(prep_cfg["resolucion"])  # (64, 64)
        self.sigma_gaussiano: float = float(prep_cfg["suavizado_gaussiano"]["sigma"])  # 0.5
        self.metodo_preprocesamiento: str = prep_cfg.get("metodo", "recorte_seguro")
        self.metodo_remuestreo: str = prep_cfg.get("metodo_remuestreo", "estandar")
        self.usar_alineacion: bool = (self.metodo_preprocesamiento == "alineacion_5puntos")
        self.config_calidad: ConfiguracionCalidad = ConfiguracionCalidad.desde_diccionario(
            self.config.get("filtro_calidad")
        )

        # Verificar integridad criptográfica
        self._verificar_integridad_archivos()

        # Instanciar detector facial YuNet con parámetros declarados
        modo_multi = self.params_detector.get("modo_multiorientacion", "cascada" if self.usar_alineacion else "estandar")
        self.detector = DetectorYuNet(
            ruta_modelo=str(self.ruta_detector),
            score_threshold=float(self.params_detector.get("score_threshold", 0.6)),
            nms_threshold=float(self.params_detector.get("nms_threshold", 0.3)),
            margen_expansion=float(self.params_detector.get("margen_expansion", 0.1)),
            modo_multiorientacion=modo_multi,
            alinear_rostro=self.usar_alineacion,
            metodo_remuestreo=self.metodo_remuestreo,
        )
        
        # Instanciar clasificador CNN y verificar salida
        self.clasificador = ClasificadorCNN.cargar(self.ruta_modelo)
        self._verificar_modelo_cargado()

    def _resolver_ruta(self, ruta_rel_o_abs: str) -> Path:
        p = Path(ruta_rel_o_abs)
        if p.is_absolute() and p.exists():
            return p
        # Probar relativo a la raíz del proyecto
        p_raiz = self.raiz_proyecto / p
        if p_raiz.exists():
            return p_raiz
        # Probar relativo al directorio del archivo de configuración
        p_cfg = self.ruta_config.parent / p
        if p_cfg.exists():
            return p_cfg
        return p_raiz  # Retornar la ruta en la raíz para mensajes de error claros

    def _cargar_y_validar_configuracion(self, ruta: Path) -> Dict[str, Any]:
        if not ruta.exists():
            raise ErrorConfiguracionCandidato(f"No existe el archivo de configuración congelada: {ruta}")

        try:
            with open(ruta, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            raise ErrorConfiguracionCandidato(f"Error al parsear JSON de configuración: {e}")

        # Secciones obligatorias
        for sec in ("candidato", "detector_facial", "preprocesamiento", "regla_acceso"):
            if sec not in cfg or not isinstance(cfg[sec], dict):
                raise ErrorConfiguracionCandidato(f"Falta sección obligatoria '{sec}' en configuración congelada")

        # Validaciones de candidato
        c = cfg["candidato"]
        for campo in ("identificador", "nombre", "ruta_modelo", "sha256_modelo", "orden_clases", "mapeo_clases", "personas_autorizadas", "umbral_score"):
            if campo not in c:
                raise ErrorConfiguracionCandidato(f"Falta campo obligatorio '{campo}' en candidato")

        # Validar umbral
        if not es_valor_probabilidad_valido(c["umbral_score"]):
            raise ErrorConfiguracionCandidato(f"Umbral de score inválido en candidato: {c['umbral_score']}")

        # Validar orden de clases
        if len(c["orden_clases"]) != 4:
            raise ErrorConfiguracionCandidato(f"Se esperaban exactamente 4 clases, encontradas: {len(c['orden_clases'])}")

        # Validar personas autorizadas (desconocido nunca debe estar en autorizados)
        for aut in c["personas_autorizadas"]:
            if aut.lower() in ETIQUETAS_DESCONOCIDO_RESERVADAS:
                raise ErrorConfiguracionCandidato(
                    f"Violación de seguridad: etiqueta reservada '{aut}' no puede pertenecer a personas_autorizadas"
                )

        # Validaciones de detector
        det = cfg["detector_facial"]
        for campo in ("ruta_modelo", "sha256_modelo", "parametros"):
            if campo not in det:
                raise ErrorConfiguracionCandidato(f"Falta campo obligatorio '{campo}' en detector_facial")

        # Validaciones de preprocesamiento
        prep = cfg["preprocesamiento"]
        if "resolucion" not in prep or len(prep["resolucion"]) != 2:
            raise ErrorConfiguracionCandidato("Resolución de preprocesamiento debe ser una lista de 2 enteros [alto, ancho]")
        if "suavizado_gaussiano" not in prep or "sigma" not in prep["suavizado_gaussiano"]:
            raise ErrorConfiguracionCandidato("Falta configuración de suavizado_gaussiano.sigma")

        return cfg

    def _verificar_integridad_archivos(self) -> None:
        """Verifica los hashes SHA-256 de los pesos de la CNN y del detector YuNet."""
        if not self.ruta_modelo.exists():
            raise FileNotFoundError(f"No existe el archivo de pesos del modelo en: {self.ruta_modelo}")
        hash_mod = calcular_sha256_archivo(self.ruta_modelo)
        if hash_mod != self.sha256_modelo:
            raise ValueError(
                f"Integridad comprometida en modelo: esperado {self.sha256_modelo}, obtenido {hash_mod}"
            )

        if not self.ruta_detector.exists():
            raise FileNotFoundError(f"No existe el archivo del detector YuNet en: {self.ruta_detector}")
        hash_det = calcular_sha256_archivo(self.ruta_detector)
        if hash_det != self.sha256_detector:
            raise ValueError(
                f"Integridad comprometida en detector YuNet: esperado {self.sha256_detector}, obtenido {hash_det}"
            )

    def _verificar_modelo_cargado(self) -> None:
        """Verifica que el clasificador produzca exactamente un tensor de 4 probabilidades válidas."""
        dummy = np.zeros((1, 1, self.resolucion[0], self.resolucion[1]), dtype=np.float32)
        probs = self.clasificador.predecir_probabilidades(dummy)
        if probs.shape != (1, 4):
            raise ValueError(f"Salida del clasificador inválida: se esperaba forma (1, 4), obtenida {probs.shape}")
        valido, motivo = validar_vector_probabilidades(probs[0], n_clases_esperadas=4)
        if not valido:
            raise ValueError(f"Salida del clasificador no es una distribución de probabilidad válida: {motivo}")

    def preprocesar_recorte(self, recorte_bgr: np.ndarray) -> np.ndarray:
        """
        Aplica el pipeline canónico estricto de preprocesamiento:
          1. Conversión a escala de grises monocromática.
          2. Normalización de resolución a 64x64 con antialiasing
             (cv2.INTER_AREA al reducir, cv2.INTER_LINEAR al ampliar).
          3. Normalización de intensidades a float32 continuo en [0.0, 1.0].
          4. Suavizado gaussiano pasa-bajos bidimensional con sigma=0.5 (kernel 5x5).
        """
        if recorte_bgr is None or recorte_bgr.size == 0:
            raise ValueError("Recorte facial vacío o nulo")

        gris = a_escala_de_grises(recorte_bgr) if len(recorte_bgr.shape) == 3 else recorte_bgr
        norm_res = normalizar_resolucion(gris, self.resolucion, usar_antialiasing=True)
        norm_int = normalizar_intensidades(norm_res, "float32_0_1")
        suavizada = suavizado_gaussiano(norm_int, sigma=self.sigma_gaussiano)
        return suavizada

    def inferir_probabilidades(self, imagen_preprocesada: np.ndarray) -> np.ndarray:
        """
        Ejecuta la inferencia de la red convolucional sobre una imagen preprocesada de 64x64.
        Valida que el vector de salida cumpla todas las restricciones de probabilidad.
        """
        in_tensor = np.expand_dims(imagen_preprocesada, axis=0)  # (1, 64, 64) -> _asegurar_tensor_4d lo hace (1, 1, 64, 64)
        probs_batch = self.clasificador.predecir_probabilidades(in_tensor)
        probs = probs_batch[0]

        valido, motivo = validar_vector_probabilidades(probs, n_clases_esperadas=4)
        if not valido:
            raise ValueError(f"Vector de probabilidades inválido producido por el modelo: {motivo}")

        return probs

    def evaluar_recorte(self, recorte_bgr: np.ndarray) -> Tuple[DecisionAcceso, np.ndarray]:
        """
        Preprocesa e infiere sobre un recorte facial conocido.
        Retorna la decisión canónica de acceso y el vector de 4 probabilidades.
        """
        suavizada = self.preprocesar_recorte(recorte_bgr)
        probs = self.inferir_probabilidades(suavizada)
        pred_idx = int(np.argmax(probs))
        score = float(probs[pred_idx])
        nombre_pred = self.mapeo_id_clase.get(pred_idx, f"desconocido_{pred_idx}")

        decision = evaluar_regla_acceso_completa(
            estado_deteccion="UN_ROSTRO",
            procesamiento_exitoso=True,
            clase_predicha=nombre_pred,
            score=score,
            umbral=self.umbral_score,
            personas_autorizadas=self.personas_autorizadas,
        )
        if (
            self.config_calidad.habilitado
            and self.config_calidad.margen_minimo > 0.0
            and decision.estado_puerta == "ABIERTA"
            and len(probs) >= 2
        ):
            probs_ord = np.sort(probs)
            margen = float(probs_ord[-1] - probs_ord[-2])
            if margen < self.config_calidad.margen_minimo:
                decision = DecisionAcceso(
                    estado_puerta="CERRADA",
                    es_autorizado=False,
                    persona_identificada="DESCONOCIDO",
                    confianza=score,
                    umbral=self.umbral_score,
                    motivo_cierre="MARGEN_INSUFICIENTE",
                    mensaje=f"Acceso denegado: margen ambiguo ({margen:.2f} < {self.config_calidad.margen_minimo:.2f}).",
                )
        return decision, probs

    def evaluar_fotograma_completo(
        self,
        frame_bgr: np.ndarray,
        es_pre_recortada: bool = False,
    ) -> Tuple[ResultadoDeteccion, DecisionAcceso, Optional[np.ndarray]]:
        """
        Pipeline canónico completo de fotograma a decisión de acceso:
          1. Detección facial YuNet.
          2. Si no hay exactamente un rostro, deniega acceso físicamente.
          3. Si hay exactamente un rostro, extrae recorte, preprocesa, infiere y aplica regla canónica.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            res_det = ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                rostro_recortado=None,
            )
            decision = evaluar_regla_acceso_completa(
                estado_deteccion=res_det.estado.value,
                procesamiento_exitoso=False,
                clase_predicha=None,
                score=0.0,
                umbral=self.umbral_score,
                personas_autorizadas=self.personas_autorizadas,
                error_procesamiento="Fotograma vacío o nulo",
            )
            return res_det, decision, None

        res_det = self.detector.detectar(frame_bgr, es_pre_recortada=es_pre_recortada)

        if res_det.estado != EstadoDeteccion.UN_ROSTRO or res_det.rostro_recortado is None:
            decision = evaluar_regla_acceso_completa(
                estado_deteccion=res_det.estado.value,
                procesamiento_exitoso=False,
                clase_predicha=None,
                score=0.0,
                umbral=self.umbral_score,
                personas_autorizadas=self.personas_autorizadas,
            )
            return res_det, decision, None

        try:
            recorte_a_evaluar = res_det.recorte_alineado if (self.usar_alineacion and res_det.recorte_alineado is not None) else res_det.rostro_recortado
            if self.config_calidad.habilitado:
                res_cal = evaluar_calidad_rostro(recorte_a_evaluar, res_det.recuadro, self.config_calidad)
                if not res_cal.aprobada:
                    decision_cal = DecisionAcceso(
                        estado_puerta="CERRADA",
                        es_autorizado=False,
                        persona_identificada="DESCONOCIDO",
                        confianza=0.0,
                        umbral=self.umbral_score,
                        motivo_cierre=res_cal.codigo,
                        mensaje=res_cal.motivo,
                    )
                    return res_det, decision_cal, None
            decision, probs = self.evaluar_recorte(recorte_a_evaluar)
            return res_det, decision, probs
        except Exception as e_proc:
            decision = evaluar_regla_acceso_completa(
                estado_deteccion="UN_ROSTRO",
                procesamiento_exitoso=False,
                clase_predicha=None,
                score=0.0,
                umbral=self.umbral_score,
                personas_autorizadas=self.personas_autorizadas,
                error_procesamiento=str(e_proc),
            )
            return res_det, decision, None
