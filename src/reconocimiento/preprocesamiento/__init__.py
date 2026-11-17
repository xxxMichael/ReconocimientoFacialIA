"""
Módulo de preprocesamiento facial.
"""
from reconocimiento.preprocesamiento.detector import (
    DetectorFacial,
    EstadoDeteccion,
    ResultadoDeteccion
)
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion
)
from reconocimiento.preprocesamiento.filtros import (
    generar_patron_sintetico_franjas,
    submuestrear,
    suavizado_gaussiano
)
from reconocimiento.preprocesamiento.mascara import (
    aislar_componente_mayor_8vecinos,
    generar_mascara_binaria,
    verificar_conectividad_8_vs_4
)

__all__ = [
    "DetectorFacial",
    "EstadoDeteccion",
    "ResultadoDeteccion",
    "a_escala_de_grises",
    "normalizar_resolucion",
    "normalizar_intensidades",
    "suavizado_gaussiano",
    "submuestrear",
    "generar_patron_sintetico_franjas",
    "generar_mascara_binaria",
    "aislar_componente_mayor_8vecinos",
    "verificar_conectividad_8_vs_4"
]
