"""
Módulo canónico de regla de decisión para control de acceso físico simulado.
Garantiza coherencia estricta entre la interfaz web en vivo y el auditor/evaluador externo.
"""
from dataclasses import dataclass
from typing import List, Optional, Set

# Etiquetas que denotan explícitamente a un sujeto desconocido o no enrolado
ETIQUETAS_DESCONOCIDO_RESERVADAS: Set[str] = {
    "desconocido",
    "desconocida",
    "unknown",
    "rechazo",
    "no_autorizado",
    "background",
    "fondo",
    "outlier",
}


@dataclass
class DecisionAcceso:
    """Resultado estructurado de la regla de control de acceso físico."""
    estado_puerta: str           # "ABIERTA" | "CERRADA"
    es_autorizado: bool          # True si se concedió acceso, False si se denegó
    persona_identificada: str    # Nombre predicho autorizado o "DESCONOCIDO"
    confianza: float             # Confianza de la predicción (0.0 a 1.0)
    umbral: float                # Umbral de confianza exigido
    motivo_cierre: str           # Causa del cierre si CERRADA; "" si ABIERTA
    mensaje: str                 # Mensaje legible para el usuario / HUD / logs


def evaluar_decision_acceso(
    nombre_predicho: str,
    confianza: float,
    umbral_confianza: float,
    personas_autorizadas: List[str],
) -> DecisionAcceso:
    """
    Función canónica de decisión de acceso físico para todo el sistema.
    Utilizada tanto por el streamer web en vivo como por el evaluador externo/auditor.

    Regla:
      - Si la clase predicha corresponde a un desconocido (explícito o ausente de autorizados),
        la puerta se mantiene estrictamente CERRADA.
      - Si la confianza es inferior al umbral, la puerta se mantiene CERRADA.
      - Solo si el sujeto está autorizado Y la confianza es >= umbral, la puerta se ABRE.
    """
    nom_limpio = (nombre_predicho or "").strip()
    nom_lower = nom_limpio.lower()

    # 1. Comprobación de clase desconocida explícita
    if nom_lower in ETIQUETAS_DESCONOCIDO_RESERVADAS:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=confianza,
            umbral=umbral_confianza,
            motivo_cierre=f"CLASE_DESCONOCIDO ({nom_limpio})",
            mensaje=f"ACCESO DENEGADO: Sujeto desconocido ({confianza * 100:.1f}%)",
        )

    # 2. Comprobación de lista de autorizados
    if nom_limpio not in personas_autorizadas:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=confianza,
            umbral=umbral_confianza,
            motivo_cierre=f"CLASE_NO_AUTORIZADA ({nom_limpio})",
            mensaje=f"ACCESO DENEGADO: Sujeto no autorizado ({nom_limpio})",
        )

    # 3. Comprobación de umbral de confianza
    if confianza < umbral_confianza:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=confianza,
            umbral=umbral_confianza,
            motivo_cierre=f"RECHAZO_UMBRAL ({confianza:.4f} < {umbral_confianza:.4f})",
            mensaje=f"ACCESO DENEGADO: Desconocido ({confianza * 100:.1f}% < {umbral_confianza * 100:.0f}%)",
        )

    # 4. Sujeto autorizado con confianza suficiente
    return DecisionAcceso(
        estado_puerta="ABIERTA",
        es_autorizado=True,
        persona_identificada=nom_limpio,
        confianza=confianza,
        umbral=umbral_confianza,
        motivo_cierre="",
        mensaje=f"ACCESO CONCEDIDO: {nom_limpio} ({confianza * 100:.1f}%)",
    )
