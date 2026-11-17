"""
Módulo de detección facial, validación de recuadros y manejo de bordes.
Implementa estados explícitos (UN_ROSTRO, SIN_ROSTRO, VARIOS_ROSTROS) y
asegura el recorte con preservación de relación de aspecto y padding seguro en límites.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import cv2
import numpy as np


class EstadoDeteccion(str, Enum):
    UN_ROSTRO = "UN_ROSTRO"
    SIN_ROSTRO = "SIN_ROSTRO"
    VARIOS_ROSTROS = "VARIOS_ROSTROS"


@dataclass
class ResultadoDeteccion:
    """Resultado del proceso de detección facial."""
    estado: EstadoDeteccion
    recuadro: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) en coords originales
    total_rostros_detectados: int
    rostro_recortado: Optional[np.ndarray] = None
    mensaje: str = ""


class DetectorFacial:
    """
    Detector facial multi-vista basado en clasificadores en cascada de Haar (OpenCV).
    Detecta rostros frontales y giros laterales (perfiles) en ambas direcciones
    mediante evaluación normal y con reflejo horizontal.
    Combina detecciones solapadas correspondientes a una misma persona y discrimina
    escenas con múltiples sujetos.
    """

    def __init__(
        self,
        ruta_cascada: Optional[str] = None,
        ruta_cascada_perfil: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        margen_expansion: float = 0.1,
        umbral_iou_fusion: float = 0.3,
        umbral_iomin_fusion: float = 0.45,
        habilitar_perfil: bool = True
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.margen_expansion = margen_expansion
        self.umbral_iou_fusion = umbral_iou_fusion
        self.umbral_iomin_fusion = umbral_iomin_fusion
        self.habilitar_perfil = habilitar_perfil

        if ruta_cascada is None:
            ruta_cascada = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.clasificador = cv2.CascadeClassifier(ruta_cascada)
        if self.clasificador.empty():
            raise RuntimeError(f"No se pudo cargar el clasificador Haar frontal desde: {ruta_cascada}")
        self.clasificador_frontal = self.clasificador

        if ruta_cascada_perfil is None:
            ruta_cascada_perfil = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self.clasificador_perfil = cv2.CascadeClassifier(ruta_cascada_perfil)
        if self.clasificador_perfil.empty() and self.habilitar_perfil:
            raise RuntimeError(f"No se pudo cargar el clasificador Haar de perfil desde: {ruta_cascada_perfil}")

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

    def fusionar_recuadros(
        self,
        recuadros: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Agrupa recuadros que solapan significativamente (misma cara detectada
        por frontal y/o perfil) y promedia sus coordenadas.
        Mantiene separados recuadros de rostros distintos.
        """
        if not recuadros:
            return []

        n = len(recuadros)
        visitados = [False] * n
        grupos = []

        for i in range(n):
            if visitados[i]:
                continue
            grupo = [recuadros[i]]
            visitados[i] = True
            cola = [i]
            while cola:
                curr = cola.pop(0)
                for j in range(n):
                    if not visitados[j]:
                        iou, iomin = self.calcular_iou_y_iomin(recuadros[curr], recuadros[j])
                        if iou >= self.umbral_iou_fusion or iomin >= self.umbral_iomin_fusion:
                            visitados[j] = True
                            grupo.append(recuadros[j])
                            cola.append(j)
            grupos.append(grupo)

        fusiones: List[Tuple[int, int, int, int]] = []
        for g in grupos:
            avg_x = int(round(sum(b[0] for b in g) / len(g)))
            avg_y = int(round(sum(b[1] for b in g) / len(g)))
            avg_w = int(round(sum(b[2] for b in g) / len(g)))
            avg_h = int(round(sum(b[3] for b in g) / len(g)))
            fusiones.append((avg_x, avg_y, avg_w, avg_h))

        return fusiones

    def detectar(
        self,
        imagen: np.ndarray,
        es_pre_recortada: bool = False,
        recuadros_simulados: Optional[list] = None
    ) -> ResultadoDeteccion:
        """
        Localiza el rostro en la imagen y extrae el recorte seguro.

        Parámetros:
        - imagen: np.ndarray en escala de grises (2D) o BGR/RGB (3D).
        - es_pre_recortada: si True (ej. Olivetti), asume que la imagen completa ya es el rostro.
        - recuadros_simulados: lista opcional de recuadros (x, y, w, h) para simulación o pruebas controladas.
        """
        if recuadros_simulados is not None:
            rostros = self.fusionar_recuadros([tuple(b) for b in recuadros_simulados])
            n_rostros = len(rostros)
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
                    mensaje=f"Se detectaron {n_rostros} rostros en la escena simulada. Se requiere exactamente uno."
                )
            x, y, w, h = rostros[0]
            recorte_seguro = self.recortar_con_margen_seguro(imagen, (x, y, w, h)) if imagen is not None else None
            return ResultadoDeteccion(
                estado=EstadoDeteccion.UN_ROSTRO,
                recuadro=(int(x), int(y), int(w), int(h)),
                total_rostros_detectados=1,
                rostro_recortado=recorte_seguro,
                mensaje="Rostro único detectado (simulado) y recortado correctamente."
            )

        if imagen is None or imagen.size == 0:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                mensaje="Imagen vacía o nula."
            )

        # Si el dataset es pre-recortado (como Olivetti), la cara es la imagen entera
        if es_pre_recortada:
            h, w = imagen.shape[:2]
            return ResultadoDeteccion(
                estado=EstadoDeteccion.UN_ROSTRO,
                recuadro=(0, 0, w, h),
                total_rostros_detectados=1,
                rostro_recortado=imagen.copy(),
                mensaje="Imagen pre-recortada procesada directamente como rostro."
            )

        # Convertir a escala de grises uint8 para el detector
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen

        if gris.dtype != np.uint8:
            if gris.max() <= 1.0:
                gris = (gris * 255).astype(np.uint8)
            else:
                gris = gris.astype(np.uint8)

        img_h, img_w = gris.shape[:2]
        candidatos: List[Tuple[int, int, int, int]] = []

        # 1. Detección frontal
        rostros_frontales = self.clasificador.detectMultiScale(
            gris,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        for (x, y, w, h) in rostros_frontales:
            candidatos.append((int(x), int(y), int(w), int(h)))

        # 2. Detección de perfil (giros laterales) si está habilitada
        if self.habilitar_perfil and hasattr(self, "clasificador_perfil") and not self.clasificador_perfil.empty():
            # Perfil en orientación natural
            rostros_perfil = self.clasificador_perfil.detectMultiScale(
                gris,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            for (x, y, w, h) in rostros_perfil:
                candidatos.append((int(x), int(y), int(w), int(h)))

            # Perfil reflejado horizontalmente (permite detectar giro opuesto)
            gris_reflejada = cv2.flip(gris, 1)
            rostros_perfil_reflejados = self.clasificador_perfil.detectMultiScale(
                gris_reflejada,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            for (xf, yf, wf, hf) in rostros_perfil_reflejados:
                # Convertir coordenadas reflejadas al marco original
                x_orig = img_w - (int(xf) + int(wf))
                candidatos.append((int(x_orig), int(yf), int(wf), int(hf)))

        # Fusión de detecciones solapadas correspondientes al mismo sujeto
        rostros = self.fusionar_recuadros(candidatos)
        n_rostros = len(rostros)

        if n_rostros == 0:
            return ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                mensaje="No se localizó ningún rostro en la escena."
            )
        elif n_rostros > 1:
            # Reportar estado explícito de múltiples rostros
            return ResultadoDeteccion(
                estado=EstadoDeteccion.VARIOS_ROSTROS,
                recuadro=None,
                total_rostros_detectados=n_rostros,
                mensaje=f"Se detectaron {n_rostros} rostros en la escena. Se requiere exactamente uno."
            )

        # Exactamente un rostro: recortar con manejo seguro de bordes
        x, y, w, h = rostros[0]
        recorte_seguro = self.recortar_con_margen_seguro(imagen, (x, y, w, h))

        return ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(int(x), int(y), int(w), int(h)),
            total_rostros_detectados=1,
            rostro_recortado=recorte_seguro,
            mensaje="Rostro único detectado y recortado correctamente."
        )

    def recortar_con_margen_seguro(
        self,
        imagen: np.ndarray,
        recuadro: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Recorta la región facial manteniendo relación de aspecto cuadrada (1:1)
        y aplicando padding si el recuadro toca o sobrepasa los bordes físicos de la imagen.
        """
        x, y, w, h = recuadro
        img_h, img_w = imagen.shape[:2]

        # Expandir con margen
        dx = int(w * self.margen_expansion)
        dy = int(h * self.margen_expansion)

        x1 = x - dx
        y1 = y - dy
        x2 = x + w + dx
        y2 = y + h + dy

        # Hacer el recuadro perfectamente cuadrado para preservar relación de aspecto
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

        # Calcular cuánto excede los límites (padding necesario)
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - img_h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - img_w)

        # Región válida dentro de la imagen
        src_y1 = max(0, y1)
        src_y2 = min(img_h, y2)
        src_x1 = max(0, x1)
        src_x2 = min(img_w, x2)

        recorte = imagen[src_y1:src_y2, src_x1:src_x2]

        # Si desborda algún borde, aplicar padding reflectante o constante
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            if len(imagen.shape) == 3:
                recorte = cv2.copyMakeBorder(
                    recorte,
                    pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )
            else:
                recorte = cv2.copyMakeBorder(
                    recorte,
                    pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )

        return recorte
