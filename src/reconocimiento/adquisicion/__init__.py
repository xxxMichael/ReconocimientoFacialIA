"""
Módulo de adquisición de datos para reconocimiento facial.
"""
from reconocimiento.adquisicion.olivetti import DatasetParticionado, cargar_olivetti
from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    generar_manifiesto_plantilla_300,
    POSES_VALIDAS,
    PARTICIONES_VALIDAS,
    MAPEO_SESION_PARTICION_PROPIO,
    obtener_particion_esperada_propio,
    validar_coherencia_sesion_particion_propio
)
from reconocimiento.adquisicion.camara import (
    CapturadorCamara,
    ejecutar_modo_captura,
    ejecutar_vista_previa,
    guardar_fotograma_atomico,
    sanitizar_componente_ruta,
    sondear_camaras
)

__all__ = [
    "DatasetParticionado",
    "cargar_olivetti",
    "ManifiestoDataset",
    "RegistroCaptura",
    "generar_manifiesto_plantilla_300",
    "POSES_VALIDAS",
    "PARTICIONES_VALIDAS",
    "MAPEO_SESION_PARTICION_PROPIO",
    "obtener_particion_esperada_propio",
    "validar_coherencia_sesion_particion_propio",
    "CapturadorCamara",
    "sondear_camaras",
    "ejecutar_vista_previa",
    "ejecutar_modo_captura",
    "guardar_fotograma_atomico",
    "sanitizar_componente_ruta"
]
