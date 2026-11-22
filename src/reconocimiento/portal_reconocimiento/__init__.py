"""
Portal independiente de reconocimiento facial remoto.
Aislado de la web administrativa y del portal de recopilación de datos.
"""

from .app import crear_portal_reconocimiento_app
from .servicio import CANDIDATOS_PORTAL, ServicioPortalReconocimiento
from .servidor import iniciar_servidor_portal_reconocimiento

__all__ = [
    "CANDIDATOS_PORTAL",
    "ServicioPortalReconocimiento",
    "crear_portal_reconocimiento_app",
    "iniciar_servidor_portal_reconocimiento",
]
