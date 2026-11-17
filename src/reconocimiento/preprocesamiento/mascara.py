"""
Módulo de binarización, máscara facial y segmentación por componentes conexos con 8-vecinos.
Incluye verificación matemática y empírica de conectividad 8 vs 4 en diagonales.
"""
from typing import Dict, Tuple
import cv2
import numpy as np


def generar_mascara_binaria(
    imagen: np.ndarray,
    metodo: str = "otsu",
    umbral_fijo: int = 127
) -> Tuple[np.ndarray, float]:
    """
    Genera una máscara binaria a partir de la imagen en escala de grises.

    Métodos soportados:
    - 'otsu': Cálculo automático del umbral que minimiza la varianza intra-clase (Otsu).
    - 'fijo': Umbralización por valor escalar fijo.

    Retorna: (mascara_uint8, umbral_calculado)
    """
    if len(imagen.shape) != 2:
        raise ValueError("La imagen debe ser bidimensional (escala de grises).")

    img_uint8 = imagen.astype(np.uint8) if imagen.dtype != np.uint8 else imagen

    if metodo.lower() == "otsu":
        umbral, mascara = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif metodo.lower() == "fijo":
        umbral, mascara = cv2.threshold(img_uint8, umbral_fijo, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Método de binarización '{metodo}' no soportado. Opciones: 'otsu', 'fijo'.")

    return mascara, float(umbral)


def aislar_componente_mayor_8vecinos(
    mascara_binaria: np.ndarray
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Aplica componentes conexos con conectividad de 8 vecinos para aislar la
    región de primer plano mayoritaria (eliminando ruido o artefactos aislados).

    Parámetro de conectividad: connectivity=8 (agrupa píxeles contiguos ortogonales y diagonales).
    """
    mascara_limpia = (mascara_binaria > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mascara_limpia,
        connectivity=8
    )

    if num_labels <= 1:
        # Solo fondo (o vacía)
        return mascara_binaria.copy(), {
            "num_componentes": num_labels,
            "area_componente_mayor": 0,
            "centroide": (0.0, 0.0)
        }

    # stats[:, cv2.CC_STAT_AREA]: índice 0 es el fondo; buscamos el mayor índice >= 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx_mayor = int(np.argmax(areas) + 1)
    area_mayor = int(stats[idx_mayor, cv2.CC_STAT_AREA])
    centroide_mayor = (float(centroids[idx_mayor][0]), float(centroids[idx_mayor][1]))

    # Máscara binaria del componente seleccionado
    mascara_resultado = (labels == idx_mayor).astype(np.uint8) * 255

    metadatos = {
        "num_componentes": int(num_labels),
        "area_componente_mayor": area_mayor,
        "centroide": centroide_mayor,
        "conectividad": 8
    }

    return mascara_resultado, metadatos


def verificar_conectividad_8_vs_4() -> Dict[str, any]:
    """
    Demostración experimental de conectividad en una máscara 3x3 sintética:
    M = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]

    Los dos píxeles en (0,0) y (1,1) están unidos únicamente por su diagonal.
    - Con conectividad 4 (solo vecinos ortogonales): forman 2 componentes conexos independientes.
    - Con conectividad 8 (vecinos ortogonales y diagonales): forman 1 único componente conexo.
    """
    mascara_test = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.uint8)

    num_labels_4, labels_4 = cv2.connectedComponents(mascara_test, connectivity=4)
    num_labels_8, labels_8 = cv2.connectedComponents(mascara_test, connectivity=8)

    # Restamos 1 para excluir la etiqueta de fondo (0)
    comps_primer_plano_4 = num_labels_4 - 1
    comps_primer_plano_8 = num_labels_8 - 1

    mismo_componente_en_8 = bool(labels_8[0, 0] == labels_8[1, 1])
    mismo_componente_en_4 = bool(labels_4[0, 0] == labels_4[1, 1])

    return {
        "mascara_sintetica": mascara_test.tolist(),
        "conectividad_4": {
            "num_componentes_primer_plano": comps_primer_plano_4,
            "etiqueta_p00": int(labels_4[0, 0]),
            "etiqueta_p11": int(labels_4[1, 1]),
            "comparten_componente": mismo_componente_en_4
        },
        "conectividad_8": {
            "num_componentes_primer_plano": comps_primer_plano_8,
            "etiqueta_p00": int(labels_8[0, 0]),
            "etiqueta_p11": int(labels_8[1, 1]),
            "comparten_componente": mismo_componente_en_8
        },
        "conclusion": (
            "En conectividad 8, los píxeles unidos diagonalmente pertenecen al mismo componente. "
            "En conectividad 4, se clasifican como componentes separados."
        )
    }
