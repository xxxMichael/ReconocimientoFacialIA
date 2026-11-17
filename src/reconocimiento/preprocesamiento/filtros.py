"""
Módulo de filtros espaciales, suavizado gaussiano y remuestreo con análisis de Nyquist.
Vinculado a:
- Tratamiento de imágenes 5 SC.md: págs 8-9 (filtro gaussiano) y págs 20-26 (dominio frecuencia/aliasing).
- Digital_Image_Processing_EE368_CS232.md: págs 5-6 (muestreo y resolución espacial).
"""
from typing import Optional, Tuple
import cv2
import numpy as np


def suavizado_gaussiano(
    imagen: np.ndarray,
    sigma: float = 1.0,
    ksize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Aplica filtro pasa-bajos gaussiano bidimensional:
    T(x,y) = exp( - (x^2 + y^2) / (2 * pi * sigma^2) )  [Tratamiento de imágenes 5 SC, pág 8]

    Parámetros:
    - sigma: desviación estándar del filtro. Si sigma == 0, devuelve copia intacta.
    - ksize: tamaño de la máscara (debe ser impar). Si es None, se calcula como 2 * ceil(3*sigma) + 1.
    """
    if sigma <= 0.0:
        return imagen.copy()

    if ksize is None:
        # Regla estándar: kernel que cubra al menos 3*sigma a cada lado
        k = int(2 * np.ceil(3 * sigma) + 1)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        ksize = (k, k)

    return cv2.GaussianBlur(imagen, ksize, sigmaX=sigma, sigmaY=sigma)


def submuestrear(
    imagen: np.ndarray,
    factor: int = 2,
    aplicar_antialiasing: bool = True
) -> np.ndarray:
    """
    Reduce la resolución espacial por un factor entero 'factor'.

    Fundamento teórico (Nyquist y aliasing):
    - Al submuestrear por un factor S=2, la frecuencia de corte de Nyquist disminuye
      de f_s/2 = 0.5 ciclos/píxel a (f_s/S)/2 = 0.25 ciclos/píxel original.
    - Toda componente de frecuencia espacial superior a 0.25 ciclos/píxel se refleja
      hacia bajas frecuencias, generando frecuencias espurias (aliasing y patrones de Moiré).
    - Para prevenirlo, 'aplicar_antialiasing=True' aplica un filtro pasa-bajos gaussiano previo
      o integración por área (cv2.INTER_AREA) antes de decimar.
    - 'aplicar_antialiasing=False' realiza decimación directa (salto de píxeles), evidenciando el aliasing.
    """
    if factor <= 1:
        return imagen.copy()

    if aplicar_antialiasing:
        # Pre-filtrado pasa-bajos gaussiano con sigma proporcional al factor de reducción
        sigma_antialiasing = float(factor) / 2.0
        filtrada = suavizado_gaussiano(imagen, sigma=sigma_antialiasing)
        # Decimación sobre la señal filtrada
        return filtrada[::factor, ::factor]
    else:
        # Submuestreo directo por decimanación sin filtro previo
        return imagen[::factor, ::factor]


def generar_patron_sintetico_franjas(
    dimensiones: Tuple[int, int] = (256, 256),
    periodo_inicial: float = 32.0,
    periodo_final: float = 2.0
) -> np.ndarray:
    """
    Genera un patrón de chirrido / franjas sinusoidales cuya frecuencia espacial
    crece horizontalmente de izquierda a derecha hasta alcanzar la frecuencia límite de Nyquist.
    Permite observar visual y cuantitativamente la aparición de aliasing al submuestrear.
    """
    alto, ancho = dimensiones
    x = np.linspace(0, 1, ancho, dtype=np.float32)

    # Frecuencia espacial creciente (chirp espacial)
    frecuencia = np.linspace(1.0 / periodo_inicial, 1.0 / periodo_final, ancho, dtype=np.float32)
    fase = 2.0 * np.pi * np.cumsum(frecuencia)

    franja_1d = 0.5 * (1.0 + np.sin(fase))
    patron = np.tile(franja_1d, (alto, 1))
    return (patron * 255.0).astype(np.uint8)
