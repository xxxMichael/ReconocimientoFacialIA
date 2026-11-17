"""
Módulo de normalización de resolución e intensidades.
Garantiza dimensiones fijas, preservación de relación de aspecto y
escalado controlado de valores de pixel.
"""
from typing import Tuple, Union
import cv2
import numpy as np


def a_escala_de_grises(imagen: np.ndarray) -> np.ndarray:
    """
    Convierte imagen a escala de grises conservando la condición monocromática
    R = G = B (referencia Digital_Image_Processing_EE368_CS232.md pág 6).
    """
    if len(imagen.shape) == 2:
        return imagen.copy()
    if len(imagen.shape) == 3:
        if imagen.shape[2] == 1:
            return imagen[:, :, 0].copy()
        elif imagen.shape[2] == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        elif imagen.shape[2] == 4:
            return cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Dimensiones de imagen no soportadas: {imagen.shape}")


def normalizar_resolucion(
    imagen: np.ndarray,
    dimension_destino: Tuple[int, int] = (64, 64),
    usar_antialiasing: bool = True
) -> np.ndarray:
    """
    Redimensiona el rostro a una resolución fija (por defecto 64x64, compatible con Olivetti).
    Aplica interpolación cv2.INTER_AREA (antialiasing) cuando se reduce tamaño,
    o cv2.INTER_CUBIC cuando se amplía.
    """
    alto_orig, ancho_orig = imagen.shape[:2]
    alto_dest, ancho_dest = dimension_destino

    if (alto_orig, ancho_orig) == (alto_dest, ancho_dest):
        return imagen.copy()

    # Si se reduce la escala en ambas dimensiones y se pide antialiasing,
    # cv2.INTER_AREA actúa como filtro integrador pasa-bajos.
    if usar_antialiasing and (alto_dest < alto_orig or ancho_dest < ancho_orig):
        metodo_interp = cv2.INTER_AREA
    else:
        metodo_interp = cv2.INTER_LINEAR

    redimensionada = cv2.resize(imagen, (ancho_dest, alto_dest), interpolation=metodo_interp)
    return redimensionada


def normalizar_intensidades(
    imagen: np.ndarray,
    rango_salida: str = "float32_0_1"
) -> np.ndarray:
    """
    Normaliza el rango dinámico de intensidades:
    - 'float32_0_1': valores float32 en [0.0, 1.0] (formato estándar de Olivetti).
    - 'uint8_0_255': valores uint8 en [0, 255] (formato para visualización y OpenCV).
    """
    img = imagen.astype(np.float32)

    val_max = img.max()
    val_min = img.min()

    # Si la imagen ya está en [0, 1] y se pide float32_0_1
    if rango_salida == "float32_0_1":
        if val_max > 1.0:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    elif rango_salida == "uint8_0_255":
        if val_max <= 1.0:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Rango de salida desconocido: {rango_salida}")
