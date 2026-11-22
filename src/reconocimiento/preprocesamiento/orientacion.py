"""
Módulo de orientación espacial de fotogramas y transformaciones geométricas.

Permite:
1. Rotar fotogramas a 0°, 90°, 180° y 270° considerando el intercambio de dimensiones (H <-> W).
2. Transformar recuadros delimitadores y puntos faciales (landmarks) de vuelta a las
   coordenadas del fotograma original sin pérdida de precisión.
3. Fusionar detecciones duplicadas del mismo rostro obtenidas en distintas orientaciones
   sin fusionar rostros pertenecientes a personas distintas.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


ORIENTACIONES_VALIDAS = (0, 90, 180, 270)


@dataclass
class DeteccionOrientada:
    """Representa una detección facial en una orientación específica con sus datos transformados."""
    recuadro_original: Tuple[int, int, int, int]  # (x, y, w, h) en coords originales
    puntos_original: np.ndarray                   # (5, 2) en coords originales
    score: float                                  # Confianza de YuNet
    angulo_deteccion: int                         # Ángulo en que fue detectado (0, 90, 180, 270)
    recuadro_rotado: Tuple[int, int, int, int]    # (x, y, w, h) en la imagen rotada


def rotar_fotograma(imagen: np.ndarray, angulo: int) -> np.ndarray:
    """
    Aplica rotación rígida en ángulos ortogonales (0°, 90°, 180°, 270° en sentido horario).
    En 90° y 270°, las dimensiones de la imagen rotada intercambian ancho y alto:
        (alto_orig, ancho_orig) -> (ancho_orig, alto_orig)
    """
    if angulo == 0:
        return imagen
    elif angulo == 90:
        return cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
    elif angulo == 180:
        return cv2.rotate(imagen, cv2.ROTATE_180)
    elif angulo == 270:
        return cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Ángulo de rotación no soportado: {angulo}. Opciones válidas: {ORIENTACIONES_VALIDAS}")


def transformar_punto_a_original(
    punto_rotado: Tuple[float, float],
    angulo: int,
    alto_orig: int,
    ancho_orig: int
) -> Tuple[float, float]:
    """
    Transforma un punto (x_rot, y_rot) de la imagen rotada hacia las coordenadas
    de la imagen original de dimensiones (alto_orig, ancho_orig).
    """
    xr, yr = punto_rotado
    if angulo == 0:
        return float(xr), float(yr)
    elif angulo == 90:
        # En rotación 90° CW: x_rot = alto_orig - 1 - y_orig, y_rot = x_orig
        # Inversa: x_orig = y_rot, y_orig = alto_orig - 1 - x_rot
        return float(yr), float(alto_orig - 1.0 - xr)
    elif angulo == 180:
        # En rotación 180°: x_rot = ancho_orig - 1 - x_orig, y_rot = alto_orig - 1 - y_orig
        # Inversa: x_orig = ancho_orig - 1 - x_rot, y_orig = alto_orig - 1 - y_rot
        return float(ancho_orig - 1.0 - xr), float(alto_orig - 1.0 - yr)
    elif angulo == 270:
        # En rotación 270° CW: x_rot = y_orig, y_rot = ancho_orig - 1 - x_orig
        # Inversa: x_orig = ancho_orig - 1 - y_rot, y_orig = x_rot
        return float(ancho_orig - 1.0 - yr), float(xr)
    else:
        raise ValueError(f"Ángulo no válido: {angulo}")


def transformar_recuadro_a_original(
    recuadro_rotado: Tuple[float, float, float, float],
    angulo: int,
    alto_orig: int,
    ancho_orig: int
) -> Tuple[int, int, int, int]:
    """
    Transforma un recuadro (x_rot, y_rot, w_rot, h_rot) detectado en la imagen rotada
    hacia las coordenadas originales (x, y, w, h).
    Considera el intercambio de ancho y alto en 90° y 270°.
    """
    xr, yr, wr, hr = recuadro_rotado
    if angulo == 0:
        x, y, w, h = xr, yr, wr, hr
    elif angulo == 90:
        x = yr
        y = alto_orig - (xr + wr)
        w = hr
        h = wr
    elif angulo == 180:
        x = ancho_orig - (xr + wr)
        y = alto_orig - (yr + hr)
        w = wr
        h = hr
    elif angulo == 270:
        x = ancho_orig - (yr + hr)
        y = xr
        w = hr
        h = wr
    else:
        raise ValueError(f"Ángulo no válido: {angulo}")

    # Clamping a los límites válidos de la imagen original
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(ancho_orig, int(round(x + w)))
    y2 = min(alto_orig, int(round(y + h)))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def transformar_puntos_a_original(
    puntos_rotados: np.ndarray,
    angulo: int,
    alto_orig: int,
    ancho_orig: int
) -> np.ndarray:
    """
    Transforma un conjunto de puntos faciales (N, 2) de coordenadas rotadas
    hacia las coordenadas del fotograma original.
    Preserva el orden semántico exacto de los puntos.
    """
    if puntos_rotados is None or len(puntos_rotados) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.asarray(puntos_rotados, dtype=np.float32)
    if angulo == 0:
        return pts.copy()

    salida = np.zeros_like(pts)
    for i in range(len(pts)):
        x_orig, y_orig = transformar_punto_a_original(
            (pts[i, 0], pts[i, 1]), angulo, alto_orig, ancho_orig
        )
        salida[i] = [x_orig, y_orig]
    return salida


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


def fusionar_detecciones_multiorientacion(
    detecciones: List[DeteccionOrientada],
    umbral_iou: float = 0.35,
    umbral_iomin: float = 0.50,
) -> List[DeteccionOrientada]:
    """
    Fusiona detecciones duplicadas del MISMO rostro que fueron detectadas en
    distintas orientaciones, sin fusionar rostros pertenecientes a personas distintas.

    Criterio de agrupación:
      Dos detecciones pertenecen a la misma persona si tienen IoU >= umbral_iou
      o IoMin >= umbral_iomin.
    Para cada grupo correspondiente a una misma persona, se selecciona como
    representante la detección que obtuvo la mayor confianza de detección en YuNet,
    lo que garantiza los puntos faciales y recuadro más precisos.
    """
    if not detecciones:
        return []

    if len(detecciones) == 1:
        return [detecciones[0]]

    n = len(detecciones)
    visitados = [False] * n
    grupos: List[List[DeteccionOrientada]] = []

    for i in range(n):
        if visitados[i]:
            continue
        grupo = [detecciones[i]]
        visitados[i] = True

        for j in range(i + 1, n):
            if visitados[j]:
                continue
            iou, iomin = calcular_iou_y_iomin(
                detecciones[i].recuadro_original,
                detecciones[j].recuadro_original
            )
            if iou >= umbral_iou or iomin >= umbral_iomin:
                grupo.append(detecciones[j])
                visitados[j] = True

        grupos.append(grupo)

    # Para cada rostro único detectado, seleccionar el de mayor score
    rostros_fusionados: List[DeteccionOrientada] = []
    for g in grupos:
        mejor = max(g, key=lambda d: d.score)
        rostros_fusionados.append(mejor)

    # Ordenar por score descendente
    rostros_fusionados.sort(key=lambda d: d.score, reverse=True)
    return rostros_fusionados
