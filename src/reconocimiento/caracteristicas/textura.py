"""
Módulo de extracción de características de textura por regiones espaciales.
Calcula histogramas de intensidad normalizados dividiendo el rostro en una cuadrícula Gy x Gx,
generando un vector estructurado con nombres, dimensiones y orden documentado.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class RepresentacionFacial:
    """Contenedor de la representación extraída para una muestra facial."""
    vector_caracteristicas: np.ndarray  # 1D array de longitud (Gy * Gx * n_bins)
    nombres_caracteristicas: List[str]
    dimension: int
    imagen_normalizada_2d: np.ndarray  # Conservada explícitamente para la CNN
    configuracion_cuadricula: Tuple[int, int]
    n_bins: int


class ExtractorHistogramasRegionales:
    """
    Extractor de descriptores de textura mediante histogramas de intensidad
    normalizados por subregiones ortogonales.
    """

    def __init__(
        self,
        cuadricula: Tuple[int, int] = (4, 4),
        n_bins: int = 16,
        rango_intensidad: Tuple[float, float] = (0.0, 1.0)
    ):
        self.gy, self.gx = cuadricula
        self.n_bins = n_bins
        self.rango_intensidad = rango_intensidad
        self.dimension_total = self.gy * self.gx * self.n_bins

        # Generar nombres estables y descriptivos para cada componente del vector
        self.nombres_caracteristicas: List[str] = []
        for ry in range(self.gy):
            for rx in range(self.gx):
                for b in range(self.n_bins):
                    self.nombres_caracteristicas.append(f"reg_y{ry}_x{rx}_bin{b:02d}")

    def extraer(self, imagen_2d: np.ndarray) -> RepresentacionFacial:
        """
        Extrae el vector de histogramas concatenados para una imagen 2D.
        """
        if len(imagen_2d.shape) != 2:
            raise ValueError(f"La imagen de entrada debe ser 2D (escala de grises). Recibido: {imagen_2d.shape}")

        img = imagen_2d.astype(np.float32)
        # Asegurar rango [0, 1] si el rango configurado es (0.0, 1.0)
        if self.rango_intensidad == (0.0, 1.0) and img.max() > 1.0:
            img = img / 255.0

        alto, ancho = img.shape
        paso_y = alto / self.gy
        paso_x = ancho / self.gx

        histogramas_regionales = []

        for ry in range(self.gy):
            y_inicio = int(round(ry * paso_y))
            y_fin = int(round((ry + 1) * paso_y)) if ry < self.gy - 1 else alto

            for rx in range(self.gx):
                x_inicio = int(round(rx * paso_x))
                x_fin = int(round((rx + 1) * paso_x)) if rx < self.gx - 1 else ancho

                region = img[y_inicio:y_fin, x_inicio:x_fin]

                # Cálculo de histograma con n_bins
                conteo, _ = np.histogram(
                    region,
                    bins=self.n_bins,
                    range=self.rango_intensidad
                )

                conteo_float = conteo.astype(np.float32)
                suma = conteo_float.sum()
                # Normalización para suma unitaria (invarianza a escala local) con guarda anti-cero
                if suma > 1e-7:
                    hist_norm = conteo_float / suma
                else:
                    hist_norm = np.zeros(self.n_bins, dtype=np.float32)

                histogramas_regionales.append(hist_norm)

        vector = np.concatenate(histogramas_regionales)

        return RepresentacionFacial(
            vector_caracteristicas=vector,
            nombres_caracteristicas=self.nombres_caracteristicas,
            dimension=self.dimension_total,
            imagen_normalizada_2d=img.copy(),
            configuracion_cuadricula=(self.gy, self.gx),
            n_bins=self.n_bins
        )

    def extraer_lote(self, imagenes: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Procesa una matriz de N imágenes (N, H, W) y retorna la matriz de características (N, D).
        """
        n_muestras = len(imagenes)
        X_caract = np.zeros((n_muestras, self.dimension_total), dtype=np.float32)

        for i in range(n_muestras):
            rep = self.extraer(imagenes[i])
            X_caract[i] = rep.vector_caracteristicas

        return X_caract, self.nombres_caracteristicas
