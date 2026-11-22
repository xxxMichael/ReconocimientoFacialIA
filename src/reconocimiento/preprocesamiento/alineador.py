"""
Módulo de alineación facial compartida mediante transformación de similitud 2D.

Alinea el rostro detectado hacia una plantilla canónica fija de 5 puntos faciales
(ojos, nariz, comisuras bucales) garantizando:
1. Determinante positivo de la matriz lineal (previene estrictamente reflejos especulares).
2. Tratamiento seguro de bordes mediante cv2.BORDER_REFLECT_101.
3. Interpolación de alta calidad (cv2.INTER_AREA para reducción, cv2.INTER_LINEAR para ampliación).
4. Comportamiento explícito y documentado de fallback cuando los puntos no sean utilizables.
5. Exportación de metadatos completos para trazabilidad y reproducibilidad.
"""
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
)
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano


# Plantilla canónica fija de 5 puntos faciales para resolución 64x64.
# Coordenadas (x, y) en píxeles:
# - p0: ojo derecho del sujeto (lado izquierdo de la imagen)
# - p1: ojo izquierdo del sujeto (lado derecho de la imagen)
# - p2: punta de la nariz
# - p3: comisura labial derecha
# - p4: comisura labial izquierda
# Propiedades geométricas:
# - Simetría vertical exacta respecto al eje x = 32.0.
# - Línea interpupilar a y = 24.0 (37.5% de la altura).
# - Distancia interpupilar de 22.0 px (34.4% del ancho), dejando 21 px a cada lado.
# - Nariz centrada en (32.0, 36.0).
# - Boca a y = 48.0 (75% de la altura) con ancho de 17.0 px.
PLANTILLA_FACIAL_64x64 = np.array([
    [21.0, 24.0],  # ojo derecho
    [43.0, 24.0],  # ojo izquierdo
    [32.0, 36.0],  # nariz
    [23.5, 48.0],  # comisura derecha
    [40.5, 48.0],  # comisura izquierda
], dtype=np.float32)


@dataclass
class MetadatosAlineacion:
    """Metadatos de la transformación de alineación facial."""
    alineado: bool
    metodo: str  # "similitud_5puntos" | "fallback_recorte_seguro"
    angulo_grados: float
    escala: float
    determinante: float
    matriz_afinidad: List[List[float]]
    puntos_origen: Optional[List[List[float]]]
    plantilla_destino: List[List[float]]
    resolucion_salida: Tuple[int, int]
    motivo_fallback: str = ""
    metodo_remuestreo: str = "estandar"
    factor_sobremuestreo: int = 1

    def a_diccionario(self) -> Dict[str, Any]:
        return {
            "alineado": self.alineado,
            "metodo": self.metodo,
            "angulo_grados": round(self.angulo_grados, 2),
            "escala": round(self.escala, 4),
            "determinante": round(self.determinante, 6),
            "matriz_afinidad": [[round(v, 5) for v in fila] for fila in self.matriz_afinidad],
            "puntos_origen": [[round(v, 2) for v in p] for p in self.puntos_origen] if self.puntos_origen else None,
            "plantilla_destino": [[round(v, 2) for v in p] for p in self.plantilla_destino],
            "resolucion_salida": list(self.resolucion_salida),
            "motivo_fallback": self.motivo_fallback,
        }


def estimar_similitud_sin_reflejo(
    puntos_origen: np.ndarray,
    puntos_destino: np.ndarray
) -> Optional[np.ndarray]:
    """
    Estima una transformación afín parcial de 4 grados de libertad (similitud 2D:
    escala uniforme, rotación y traslación):
        [x']   [ s*cos(th)  -s*sin(th)  tx ] [x]
        [y'] = [ s*sin(th)   s*cos(th)  ty ] [y]
                                             [1]
    Garantiza formalmente que det(M[:2, :2]) = s^2 > 0, impidiendo cualquier
    reflejo especular accidental (espejado) de la cara.
    """
    if puntos_origen.shape != (5, 2) or puntos_destino.shape != (5, 2):
        return None

    # cv2.estimateAffinePartial2D calcula por mínimos cuadrados la matriz afín parcial (similitud)
    M, _ = cv2.estimateAffinePartial2D(puntos_origen, puntos_destino, method=cv2.LMEDS)
    if M is None or M.shape != (2, 3):
        # Alternativa de mínimos cuadrados estándar si LMEDS no converge
        M, _ = cv2.estimateAffinePartial2D(puntos_origen, puntos_destino)

    if M is None or M.shape != (2, 3):
        return None

    # Verificar determinante de la submatriz lineal 2x2
    a, b = float(M[0, 0]), float(M[1, 0])
    c, d = float(M[0, 1]), float(M[1, 1])
    det = a * d - b * c

    # En una similitud estricta, c == -b y d == a, por lo que det == a^2 + b^2 = s^2 > 0
    if det <= 1e-6 or math.isnan(det) or math.isinf(det):
        return None

    # Verificar que la escala sea no nula y razonable
    escala = math.sqrt(a * a + b * b)
    if escala <= 1e-4 or escala > 100.0:
        return None

    return M


def alinear_rostro(
    imagen: np.ndarray,
    puntos_faciales: Optional[np.ndarray],
    recuadro: Optional[Tuple[int, int, int, int]] = None,
    resolucion: Tuple[int, int] = (64, 64),
    plantilla: Optional[np.ndarray] = None,
    margen_expansion_fallback: float = 0.1,
    metodo_remuestreo: str = "estandar",
) -> Tuple[np.ndarray, MetadatosAlineacion]:
    """
    Alinea una región facial hacia la plantilla canónica mediante similitud 2D.

    Parámetros:
      - imagen: fotograma o imagen original (BGR o escala de grises uint8).
      - puntos_faciales: array (5, 2) con las coordenadas de [ojos, nariz, comisuras].
      - recuadro: tupla opcional (x, y, w, h) para recorte seguro en caso de fallback.
      - resolucion: tupla (alto, ancho) de salida, por defecto (64, 64).
      - plantilla: plantilla de referencia (5, 2), por defecto PLANTILLA_FACIAL_64x64.
      - margen_expansion_fallback: margen de expansión para el fallback de recorte.

    Retorna:
      - recorte_alineado: imagen recortada y alineada de tamaño `resolucion`.
      - metadatos: objeto `MetadatosAlineacion` con información completa de la transformación.
    """
    if imagen is None or imagen.size == 0:
        raise ValueError("La imagen de entrada es nula o vacía.")

    if plantilla is None:
        plantilla = PLANTILLA_FACIAL_64x64

    ancho_dst, alto_dst = resolucion[1], resolucion[0]
    img_h, img_w = imagen.shape[:2]

    # Verificar si los puntos faciales son válidos y computables
    puntos_validos = False
    dist_ojos = 0.0
    if puntos_faciales is not None and isinstance(puntos_faciales, np.ndarray):
        puntos_arr = np.asarray(puntos_faciales, dtype=np.float32)
        if puntos_arr.shape == (5, 2) and not np.isnan(puntos_arr).any() and not np.isinf(puntos_arr).any():
            dist_ojos = float(np.linalg.norm(puntos_arr[1] - puntos_arr[0]))
            # La distancia interpupilar debe ser al menos de 4 píxeles para evitar singularidades
            if dist_ojos >= 4.0:
                puntos_validos = True

    M = None
    if puntos_validos:
        M = estimar_similitud_sin_reflejo(puntos_arr, plantilla)

    # Caso 1: Alineación exitosa mediante similitud 2D
    if M is not None:
        a, b = float(M[0, 0]), float(M[1, 0])
        escala = math.sqrt(a * a + b * b)
        angulo_rad = math.atan2(b, a)
        angulo_deg = math.degrees(angulo_rad)
        det = a * M[1, 1] - b * M[0, 1]

        factor_sobremuestreo = 1
        if metodo_remuestreo == "supersampling_area" and escala < 0.9:
            # Reducción con antialiasing real:
            # 1) Warpear a lienzo intermedio sobremuestreado k x (ancho_dst, alto_dst)
            #    donde k = clamp(round(1 / escala), 2, 16) de modo que el warp opere cerca de 1:1.
            # 2) Ajustar traslación M_sup = k*M + (k-1)/2 para centrar cada bloque k x k
            #    exactamente sobre la coordenada sub-píxel de la plantilla canónica.
            # 3) Reducir con cv2.resize(..., INTER_AREA) para integración real por área.
            k = min(max(2, int(round(1.0 / max(escala, 1e-6)))), 16)
            factor_sobremuestreo = k
            M_sup = M.copy()
            M_sup[:2, :2] = M[:2, :2] * float(k)
            M_sup[0, 2] = M[0, 2] * float(k) + (float(k) - 1.0) * 0.5
            M_sup[1, 2] = M[1, 2] * float(k) + (float(k) - 1.0) * 0.5
            inter = cv2.warpAffine(
                imagen,
                M_sup,
                (ancho_dst * k, alto_dst * k),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            alineado = cv2.resize(inter, (ancho_dst, alto_dst), interpolation=cv2.INTER_AREA)
        else:
            # Método estándar histórico (o ampliación cuando escala >= 0.9)
            interp = cv2.INTER_AREA if escala < 0.9 else cv2.INTER_LINEAR
            alineado = cv2.warpAffine(
                imagen,
                M,
                (ancho_dst, alto_dst),
                flags=interp,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        meta = MetadatosAlineacion(
            alineado=True,
            metodo="similitud_5puntos",
            angulo_grados=angulo_deg,
            escala=escala,
            determinante=det,
            matriz_afinidad=M.tolist(),
            puntos_origen=puntos_arr.tolist(),
            plantilla_destino=plantilla.tolist(),
            resolucion_salida=resolucion,
            motivo_fallback="",
            metodo_remuestreo=metodo_remuestreo,
            factor_sobremuestreo=factor_sobremuestreo,
        )
        return alineado, meta

    # Caso 2: Fallback controlado a recorte cuadrado seguro
    motivo = "puntos_faciales_no_disponibles_o_degenerados" if not puntos_validos else "error_estimacion_afinidad"

    # Si no se provee recuadro explícito, usar centro o imagen completa
    if recuadro is None:
        if puntos_faciales is not None and len(puntos_faciales) > 0:
            pts = np.asarray(puntos_faciales)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            w_box = max(20, int(max_x - min_x))
            h_box = max(20, int(max_y - min_y))
            x_box = max(0, int(min_x - w_box * 0.2))
            y_box = max(0, int(min_y - h_box * 0.3))
            recuadro = (x_box, y_box, int(w_box * 1.4), int(h_box * 1.5))
        else:
            recuadro = (0, 0, img_w, img_h)

    # Recorte seguro con preservación de aspecto 1:1 y padding reflect
    x, y, w, h = recuadro
    dx = int(w * margen_expansion_fallback)
    dy = int(h * margen_expansion_fallback)
    x1, y1 = x - dx, y - dy
    x2, y2 = x + w + dx, y + h + dy

    ancho_b = x2 - x1
    alto_b = y2 - y1
    if ancho_b > alto_b:
        dif = ancho_b - alto_b
        y1 -= dif // 2
        y2 += dif - (dif // 2)
    elif alto_b > ancho_b:
        dif = alto_b - ancho_b
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

    recorte_seguro = imagen[src_y1:src_y2, src_x1:src_x2]
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        recorte_seguro = cv2.copyMakeBorder(
            recorte_seguro,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )

    # Escalar a la resolución de salida esperada
    if recorte_seguro.shape[:2] != resolucion:
        recorte_final = cv2.resize(recorte_seguro, (ancho_dst, alto_dst), interpolation=cv2.INTER_AREA)
    else:
        recorte_final = recorte_seguro

    meta = MetadatosAlineacion(
        alineado=False,
        metodo="fallback_recorte_seguro",
        angulo_grados=0.0,
        escala=1.0,
        determinante=1.0,
        matriz_afinidad=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        puntos_origen=puntos_faciales.tolist() if (puntos_faciales is not None and isinstance(puntos_faciales, np.ndarray)) else None,
        plantilla_destino=plantilla.tolist(),
        resolucion_salida=resolucion,
        motivo_fallback=motivo
    )
    return recorte_final, meta


def preprocesar_rostro_alineado(
    imagen: np.ndarray,
    puntos_faciales: Optional[np.ndarray],
    recuadro: Optional[Tuple[int, int, int, int]] = None,
    resolucion: Tuple[int, int] = (64, 64),
    sigma: float = 0.5,
    metodo_remuestreo: str = "estandar",
) -> Tuple[np.ndarray, MetadatosAlineacion]:
    """
    Pipeline canónico de preprocesamiento facial con alineación por similitud:
      1. Alineación 2D hacia plantilla de 5 puntos (o fallback seguro).
      2. Conversión a escala de grises.
      3. Normalización min-max de intensidades a float32 [0.0, 1.0].
      4. Suavizado gaussiano con sigma configurable (por defecto 0.5).

    Retorna:
      - imagen_procesada: array 2D float32 (64, 64) en rango [0.0, 1.0].
      - metadatos: MetadatosAlineacion.
    """
    recorte_al, meta = alinear_rostro(
        imagen=imagen,
        puntos_faciales=puntos_faciales,
        recuadro=recuadro,
        resolucion=resolucion,
        metodo_remuestreo=metodo_remuestreo,
    )
    gris = a_escala_de_grises(recorte_al)
    norm = normalizar_intensidades(gris, "float32_0_1")
    suav = suavizado_gaussiano(norm, sigma=sigma)
    return suav, meta
