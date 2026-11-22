"""
Detector facial basado en YuNet (red neuronal profunda ligera ONNX de OpenCV Zoo).
Diseñado para manejar inclinaciones extremas (yaw, pitch, roll), variaciones de pose,
fotogramas rotados (0°, 90°, 180°, 270°) y alineación facial por similitud.
"""
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.preprocesamiento.orientacion import (
    ORIENTACIONES_VALIDAS,
    DeteccionOrientada,
    calcular_iou_y_iomin,
    fusionar_detecciones_multiorientacion,
    rotar_fotograma,
    transformar_puntos_a_original,
    transformar_recuadro_a_original,
)
from reconocimiento.preprocesamiento.alineador import alinear_rostro


class DetectorYuNet:
    """
    Detector facial basado en YuNet (OpenCV DNN / FaceDetectorYN).
    Soporta:
      - Detección estándar a 0° (configuración de referencia).
      - Detección multi-orientación configurable (0°, 90°, 180°, 270°) con fusión de duplicados.
      - Alineación facial por similitud 2D hacia plantilla canónica de 5 puntos.
    """

    def __init__(
        self,
        ruta_modelo: Optional[str] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        margen_expansion: float = 0.1,
        orientaciones: Optional[Tuple[int, ...]] = None,
        modo_multiorientacion: str = "estandar",
        alinear_rostro: bool = False,
        metodo_remuestreo: str = "estandar",
    ):
        self._lock = Lock()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.margen_expansion = margen_expansion
        self.modo_multiorientacion = modo_multiorientacion.lower()
        self.alinear_rostro = alinear_rostro
        self.metodo_remuestreo = metodo_remuestreo

        # Configurar lista de orientaciones según el modo
        if orientaciones is not None:
            self.orientaciones = tuple(orientaciones)
        elif self.modo_multiorientacion == "estandar":
            self.orientaciones = (0,)
        elif self.modo_multiorientacion == "fijo_90":
            self.orientaciones = (90,)
        elif self.modo_multiorientacion == "fijo_180":
            self.orientaciones = (180,)
        elif self.modo_multiorientacion == "fijo_270":
            self.orientaciones = (270,)
        elif self.modo_multiorientacion in ("cascada", "auto", "exhaustivo"):
            self.orientaciones = (0, 90, 180, 270)
        else:
            self.orientaciones = (0,)

        if ruta_modelo is None:
            ruta_defecto = Path(__file__).resolve().parents[3] / "modelos/detector/face_detection_yunet_2023mar.onnx"
            ruta_modelo = str(ruta_defecto)

        self.ruta_modelo = Path(ruta_modelo)
        if not self.ruta_modelo.exists():
            raise FileNotFoundError(f"No se encontró el modelo YuNet en: {self.ruta_modelo}")

        self.detector = cv2.FaceDetectorYN.create(
            str(self.ruta_modelo),
            "",
            (320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )

    @staticmethod
    def calcular_iou_y_iomin(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """Calcula Intersection-over-Union (IoU) y Intersection-over-Minimum (IoMin)."""
        return calcular_iou_y_iomin(box_a, box_b)

    def _detectar_en_angulo(
        self,
        img_bgr: np.ndarray,
        angulo: int,
        h_orig: int,
        w_orig: int
    ) -> List[DeteccionOrientada]:
        """Ejecuta inferencia YuNet sobre una rotación específica y transforma las coordenadas."""
        rotada = rotar_fotograma(img_bgr, angulo)
        h_r, w_r = rotada.shape[:2]

        with self._lock:
            self.detector.setInputSize((w_r, h_r))
            _, caras = self.detector.detect(rotada)

        if caras is None or len(caras) == 0:
            return []

        detecciones: List[DeteccionOrientada] = []
        for c in caras:
            score = float(c[14]) if len(c) > 14 else 1.0
            bx_r, by_r, bw_r, bh_r = int(round(c[0])), int(round(c[1])), int(round(c[2])), int(round(c[3]))

            # Clamping en imagen rotada
            x2_r, y2_r = min(w_r, bx_r + bw_r), min(h_r, by_r + bh_r)
            bx_r, by_r = max(0, bx_r), max(0, by_r)
            bw_r, bh_r = x2_r - bx_r, y2_r - by_r
            if bw_r < 20 or bh_r < 20:
                continue

            # Transformar recuadro de vuelta a las coordenadas originales
            box_orig = transformar_recuadro_a_original(
                (bx_r, by_r, bw_r, bh_r), angulo, h_orig, w_orig
            )

            # Transformar los 5 puntos faciales de vuelta a las coordenadas originales (si están presentes)
            if len(c) >= 14:
                pts_r = np.array([c[4:6], c[6:8], c[8:10], c[10:12], c[12:14]], dtype=np.float32)
                pts_orig = transformar_puntos_a_original(pts_r, angulo, h_orig, w_orig)
            else:
                cx = box_orig[0] + box_orig[2] * 0.5
                cy = box_orig[1] + box_orig[3] * 0.5
                wb, hb = float(box_orig[2]), float(box_orig[3])
                pts_orig = np.array([
                    [cx - wb * 0.18, cy - hb * 0.12],
                    [cx + wb * 0.18, cy - hb * 0.12],
                    [cx, cy + hb * 0.05],
                    [cx - wb * 0.14, cy + hb * 0.22],
                    [cx + wb * 0.14, cy + hb * 0.22],
                ], dtype=np.float32)

            detecciones.append(
                DeteccionOrientada(
                    recuadro_original=box_orig,
                    puntos_original=pts_orig,
                    score=score,
                    angulo_deteccion=angulo,
                    recuadro_rotado=(bx_r, by_r, bw_r, bh_r)
                )
            )

        return detecciones

    def detectar(
        self,
        imagen: np.ndarray,
        es_pre_recortada: bool = False,
        recuadros_simulados: Optional[list] = None
    ) -> ResultadoDeteccion:
        """
        Localiza rostros en la imagen usando YuNet.
        Soporta modos estándar (0°), cascada progresiva (0° -> 90°/270°/180°) y exhaustivo.
        """
        if recuadros_simulados is not None:
            n_rostros = len(recuadros_simulados)
            if n_rostros == 0:
                return ResultadoDeteccion(
                    estado=EstadoDeteccion.SIN_ROSTRO,
                    recuadro=None,
                    total_rostros_detectados=0,
                    mensaje="No se localizó ningún rostro en la escena simulada."
                )
            elif n_rostros > 1:
                return ResultadoDeteccion(
                    estado=EstadoDeteccion.VARIOS_ROSTROS,
                    recuadro=None,
                    total_rostros_detectados=n_rostros,
                    mensaje=f"Se detectaron {n_rostros} rostros simulados."
                )
            x, y, w, h = recuadros_simulados[0]
            recorte = self.recortar_con_margen_seguro(imagen, (x, y, w, h)) if imagen is not None else None
            return ResultadoDeteccion(
                estado=EstadoDeteccion.UN_ROSTRO,
                recuadro=(int(x), int(y), int(w), int(h)),
                total_rostros_detectados=1,
                rostro_recortado=recorte,
                mensaje="Rostro único detectado (simulado)."
            )

        if imagen is None or imagen.size == 0:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                mensaje="Imagen vacía o nula."
            )

        if es_pre_recortada:
            h, w = imagen.shape[:2]
            return ResultadoDeteccion(
                estado=EstadoDeteccion.UN_ROSTRO,
                recuadro=(0, 0, w, h),
                total_rostros_detectados=1,
                rostro_recortado=imagen.copy(),
                mensaje="Imagen pre-recortada procesada directamente como rostro."
            )

        # Convertir a formato 3 canales BGR uint8 si es escala de grises
        if len(imagen.shape) == 2:
            img_bgr = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = imagen

        h_orig, w_orig = img_bgr.shape[:2]

        # Estrategia de detección según modo configurado
        todas_detecciones: List[DeteccionOrientada] = []

        if self.modo_multiorientacion in ("cascada", "auto"):
            # Paso 1: Intentar primero en 0° (caso común, coste de 1 inferencia)
            dets_0 = self._detectar_en_angulo(img_bgr, 0, h_orig, w_orig)
            if dets_0:
                todas_detecciones = dets_0
            else:
                # Paso 2: Si no detecta en 0°, evaluar las orientaciones restantes
                for ang in (90, 270, 180):
                    dets_ang = self._detectar_en_angulo(img_bgr, ang, h_orig, w_orig)
                    todas_detecciones.extend(dets_ang)
        else:
            # Modo explícito: evaluar la lista de orientaciones configurada
            for ang in self.orientaciones:
                dets_ang = self._detectar_en_angulo(img_bgr, ang, h_orig, w_orig)
                todas_detecciones.extend(dets_ang)

        # Fusionar detecciones duplicadas del mismo rostro
        rostros_unicos = fusionar_detecciones_multiorientacion(todas_detecciones)

        n_rostros = len(rostros_unicos)
        if n_rostros == 0:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                mensaje="No se localizó ningún rostro con YuNet."
            )
        elif n_rostros > 1:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.VARIOS_ROSTROS,
                recuadro=None,
                total_rostros_detectados=n_rostros,
                mensaje=f"Se detectaron {n_rostros} rostros en la escena con YuNet."
            )

        # Exactamente UN_ROSTRO
        det = rostros_unicos[0]
        x, y, w, h = det.recuadro_original
        recorte_seguro = self.recortar_con_margen_seguro(imagen, (x, y, w, h))

        recorte_al = None
        meta_al = None
        if self.alinear_rostro:
            recorte_al, meta_obj = alinear_rostro(
                imagen=imagen,
                puntos_faciales=det.puntos_original,
                recuadro=(x, y, w, h),
                resolucion=(64, 64),
                metodo_remuestreo=self.metodo_remuestreo,
            )
            meta_al = meta_obj.a_diccionario()
            meta_al["angulo_orientacion_detector"] = det.angulo_deteccion

        return ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(x, y, w, h),
            total_rostros_detectados=1,
            rostro_recortado=recorte_seguro,
            mensaje=f"Rostro único detectado (orientación: {det.angulo_deteccion}°, score: {det.score:.2f}).",
            puntos_faciales=det.puntos_original,
            angulo_orientacion=det.angulo_deteccion,
            confianza_deteccion=det.score,
            recorte_alineado=recorte_al,
            metadatos_alineacion=meta_al
        )

    def recortar_con_margen_seguro(
        self,
        imagen: np.ndarray,
        recuadro: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Recorta la región facial manteniendo relación de aspecto 1:1 y padding seguro."""
        x, y, w, h = recuadro
        img_h, img_w = imagen.shape[:2]

        dx = int(w * self.margen_expansion)
        dy = int(h * self.margen_expansion)

        x1 = x - dx
        y1 = y - dy
        x2 = x + w + dx
        y2 = y + h + dy

        ancho = x2 - x1
        alto = y2 - y1
        if ancho > alto:
            dif = ancho - alto
            y1 -= dif // 2
            y2 += dif - (dif // 2)
        elif alto > ancho:
            dif = alto - ancho
            x1 -= dif // 2
            x2 += dif - (dif // 2)

        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - img_h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - img_w)

        src_y1 = max(0, y1)
        src_y2 = min(img_h, y2)
        src_x1 = max(0, x1)
        src_x2 = min(img_w, x2)

        recorte = imagen[src_y1:src_y2, src_x1:src_x2]

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            recorte = cv2.copyMakeBorder(
                recorte,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REFLECT_101
            )

        return recorte
