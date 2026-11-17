"""
Módulo de interfaz web local para adquisición de imágenes.
"""
from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import GestorCamaraWeb

__all__ = ["crear_app", "GestorCamaraWeb"]
