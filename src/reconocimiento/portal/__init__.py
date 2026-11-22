"""
Módulo del portal público de captura independiente para voluntarios externos.
Permite adquisición controlada bajo protocolo (5 poses x 3 distancias = 15 fotos),
aislado del panel administrativo y sin dependencias de inferencia de modelos.
"""

from reconocimiento.portal.gestor_invitaciones import GestorInvitaciones
from reconocimiento.portal.almacenamiento import GestorAlmacenamientoExterno
from reconocimiento.portal.app import crear_portal_app

__all__ = [
    "GestorInvitaciones",
    "GestorAlmacenamientoExterno",
    "crear_portal_app",
]
