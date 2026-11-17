"""
Detector facial basado en YuNet (red neuronal profunda ligera ONNX de OpenCV Zoo).
Diseñado para manejar inclinaciones extremas (yaw, pitch, roll), variaciones de pose
y condiciones de iluminación donde los clasificadores Haar tienden a perder el rostro
o engancharse falsamente en texturas de ropa o pelo.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion


class DetectorYuNet:
    """
    Detector facial basado en YuNet (OpenCV DNN / FaceDetectorYN).
    Soporta giros laterales, inclinaciones verticales y ladeos de cabeza.
    """

    def __init__(
        self,
        ruta_modelo: Optional[str] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        margen_expansion: float = 0.1
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.margen_expansion = margen_expansion

        if ruta_modelo is None:
            ruta_defecto = Path("modelos/detector/face_detection_yunet_2023mar.onnx")
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
        x1_a, y1_a, w_a, h_a = box_a
        x1_b, y1_b, w_b, h_b = box_b

        x2_a, y2_a = x1_a + w_a, y1_a + h_a
        x2_b, y2_b = x1_b + w_b, y1_b + h_b

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0, 0.0

        area_a = w_a * h_a
        area_b = w_b * h_b
        union_area = area_a + area_b - inter_area
        min_area = min(area_a, area_b)

        iou = inter_area / union_area if union_area > 0 else 0.0
        iomin = inter_area / min_area if min_area > 0 else 0.0
        return float(iou), float(iomin)

    def detectar(
        self,
        imagen: np.ndarray,
        es_pre_recortada: bool = False,
        recuadros_simulados: Optional[list] = None
    ) -> ResultadoDeteccion:
        """Localiza rostros en la imagen usando YuNet."""
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

        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))

        _, caras = self.detector.detect(img_bgr)

        if caras is None or len(caras) == 0:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                mensaje="No se localizó ningún rostro con YuNet."
            )

        # Filtrar recuadros válidos con dimensiones positivas
        recuadros_validos: List[Tuple[int, int, int, int]] = []
        for c in caras:
            bx, by, bw, bh = int(round(c[0])), int(round(c[1])), int(round(c[2])), int(round(c[3]))
            # Clamping a los límites de la imagen
            bx = max(0, min(bx, w - 1))
            by = max(0, min(by, h - 1))
            bw = max(1, min(bw, w - bx))
            bh = max(1, min(bh, h - by))
            if bw >= 20 and bh >= 20:
                recuadros_validos.append((bx, by, bw, bh))

        n_rostros = len(recuadros_validos)
        if n_rostros == 0:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                mensaje="No se localizó ningún rostro con dimensiones válidas."
            )
        elif n_rostros > 1:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.VARIOS_ROSTROS,
                recuadro=None,
                total_rostros_detectados=n_rostros,
                mensaje=f"Se detectaron {n_rostros} rostros en la escena con YuNet."
            )

        x, y, w, h = recuadros_validos[0]
        recorte_seguro = self.recortar_con_margen_seguro(imagen, (x, y, w, h))

        return ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(x, y, w, h),
            total_rostros_detectados=1,
            rostro_recortado=recorte_seguro,
            mensaje="Rostro único detectado y recortado correctamente con YuNet."
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
