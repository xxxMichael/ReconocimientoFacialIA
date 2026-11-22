"""
Módulo canónico de regla de decisión para control de acceso físico simulado.
Garantiza coherencia estricta entre la interfaz web en vivo y el auditor/evaluador externo.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

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


def es_valor_probabilidad_valido(v: Any) -> bool:
    """
    Verifica de forma estricta si un valor numérico es finito, no nulo y pertenece
    al intervalo cerrado continuo [0.0, 1.0].
    Rechaza None, NaN, infinitos, tipos no numéricos y valores fuera de rango.
    """
    if v is None:
        return False
    if isinstance(v, (bool,)):  # En Python, bool es subclase de int
        return False
    if not isinstance(v, (int, float, np.floating, np.integer)):
        return False
    try:
        val = float(v)
    except (ValueError, TypeError):
        return False
    if math.isnan(val) or math.isinf(val):
        return False
    return 0.0 <= val <= 1.0


def validar_vector_probabilidades(
    probs: Any,
    n_clases_esperadas: int = 4,
    tolerancia_suma: float = 1e-3,
) -> Tuple[bool, str]:
    """
    Valida estrictamente un vector de probabilidades predicho por el clasificador.
    Requisitos:
      - Array o secuencia 1D con exactamente n_clases_esperadas elementos.
      - Cada elemento debe ser numérico, finito y pertenecer a [0.0, 1.0].
      - La suma de todas las probabilidades debe ser coherente con 1.0 dentro de tolerancia_suma.
    Retorna (True, "") si es válido, o (False, motivo_del_rechazo) si es inválido.
    No realiza recortes silenciosos ni normalizaciones anómalas.
    """
    if probs is None:
        return False, "Vector de probabilidades es None"

    try:
        arr = np.asarray(probs, dtype=np.float64)
    except Exception as e:
        return False, f"No se pudo convertir a array numérico: {e}"

    if arr.ndim != 1 or arr.shape[0] != n_clases_esperadas:
        return False, f"Dimensión del vector de probabilidades {arr.shape} no coincide con ({n_clases_esperadas},)"

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False, "El vector de probabilidades contiene valores NaN o infinitos"

    if np.any(arr < 0.0) or np.any(arr > 1.0):
        return False, f"Valores de probabilidad fuera del rango [0.0, 1.0]: {arr.tolist()}"

    suma = float(np.sum(arr))
    if abs(suma - 1.0) > tolerancia_suma:
        return False, f"La suma de probabilidades ({suma:.6f}) difiere de 1.0 más de la tolerancia ({tolerancia_suma})"

    return True, ""


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
    distancia: Optional[float] = None        # Distancia calculada a referencias de la clase
    umbral_distancia: Optional[float] = None # Umbral de distancia máxima tolerada


def clave_ordenamiento_jerarquico(candidato: Dict[str, Any]) -> Tuple[float, float, int, float]:
    """
    Retorna la tupla de ordenación jerárquica para selección y desempate de umbrales:
    1. Menor FAR de desconocidos (far_total_pct).
    2. Menor FRR de puerta (frr_puerta_pct).
    3. Mayor cantidad de aperturas con identidad correcta (-aperturas_correctas).
    4. Mayor umbral en caso de empate completo (-umbral).
    """
    return (
        round(float(candidato.get("far_total_pct", 100.0)), 4),
        round(float(candidato.get("frr_puerta_pct", 100.0)), 4),
        -int(candidato.get("aperturas_correctas", 0)),
        -float(candidato.get("umbral", 0.0)),
    )


def seleccionar_umbral_optimo_jerarquico(
    candidatos_viables: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Selecciona el umbral óptimo de una lista de candidatos viables que cumplen
    con la cota FRR_p <= 30% para todas las personas autorizadas.

    Aplica el orden jerárquico estricto:
      1. Menor FAR
      2. Menor FRR de puerta
      3. Mayor aperturas correctas
      4. Mayor umbral
    Retorna None si la lista de candidatos está vacía.
    """
    if not candidatos_viables:
        return None
    ordenados = sorted(candidatos_viables, key=clave_ordenamiento_jerarquico)
    return ordenados[0]


def evaluar_decision_acceso(
    nombre_predicho: str,
    confianza: float,
    umbral_confianza: float,
    personas_autorizadas: List[str],
) -> DecisionAcceso:
    """
    Función canónica de decisión de acceso físico para predicciones puntuales.
    Utilizada tanto por el streamer web en vivo como por el evaluador externo/auditor.

    Regla:
      - Umbral y score deben ser numéricos, finitos y pertenecer a [0.0, 1.0].
      - Si la clase predicha corresponde a un desconocido (explícito o ausente de autorizados),
        la puerta se mantiene estrictamente CERRADA.
      - Si la confianza es inferior al umbral, la puerta se mantiene CERRADA.
      - Solo si el sujeto está autorizado Y la confianza es >= umbral, la puerta se ABRE.
    """
    # 0. Validación de integridad numérica del umbral
    if not es_valor_probabilidad_valido(umbral_confianza):
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=confianza if es_valor_probabilidad_valido(confianza) else 0.0,
            umbral=0.0,
            motivo_cierre=f"UMBRAL_INVALIDO ({umbral_confianza})",
            mensaje=f"ACCESO DENEGADO: Configuración de umbral inválida ({umbral_confianza})",
        )

    # 0.1 Validación de integridad numérica de la confianza/score
    if not es_valor_probabilidad_valido(confianza):
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=0.0,
            umbral=float(umbral_confianza),
            motivo_cierre=f"SCORE_INVALIDO ({confianza})",
            mensaje=f"ACCESO DENEGADO: Score de predicción inválido ({confianza})",
        )

    conf_val = float(confianza)
    umb_val = float(umbral_confianza)
    nom_limpio = (nombre_predicho or "").strip()
    nom_lower = nom_limpio.lower()

    # 1. Comprobación de clase desconocida explícita
    if nom_lower in ETIQUETAS_DESCONOCIDO_RESERVADAS:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=conf_val,
            umbral=umb_val,
            motivo_cierre=f"CLASE_DESCONOCIDO ({nom_limpio})",
            mensaje=f"ACCESO DENEGADO: Sujeto desconocido ({conf_val * 100:.1f}%)",
        )

    # 2. Comprobación de lista de autorizados
    if not personas_autorizadas or nom_limpio not in personas_autorizadas:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=conf_val,
            umbral=umb_val,
            motivo_cierre=f"CLASE_NO_AUTORIZADA ({nom_limpio})",
            mensaje=f"ACCESO DENEGADO: Sujeto no autorizado ({nom_limpio})",
        )

    # 3. Comprobación de umbral de confianza
    if conf_val < umb_val:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=conf_val,
            umbral=umb_val,
            motivo_cierre=f"RECHAZO_UMBRAL ({conf_val:.4f} < {umb_val:.4f})",
            mensaje=f"ACCESO DENEGADO: Desconocido ({conf_val * 100:.1f}% < {umb_val * 100:.0f}%)",
        )

    # 4. Sujeto autorizado con confianza suficiente
    return DecisionAcceso(
        estado_puerta="ABIERTA",
        es_autorizado=True,
        persona_identificada=nom_limpio,
        confianza=conf_val,
        umbral=umb_val,
        motivo_cierre="",
        mensaje=f"ACCESO CONCEDIDO: {nom_limpio} ({conf_val * 100:.1f}%)",
    )


def evaluar_regla_acceso_completa(
    estado_deteccion: str,
    procesamiento_exitoso: bool,
    clase_predicha: Optional[str],
    score: Optional[float],
    umbral: float,
    personas_autorizadas: List[str],
    error_procesamiento: Optional[str] = None,
) -> DecisionAcceso:
    """
    Regla canónica de control de acceso físico de extremo a extremo:
    Abrir únicamente si:
      - Se detecta exactamente un rostro (estado_deteccion == 'UN_ROSTRO').
      - El procesamiento finaliza correctamente (procesamiento_exitoso == True).
      - La clase predicha pertenece a personas_autorizadas (desconocido nunca abre).
      - Su score y umbral son finitos, numéricos y pertenecen a [0.0, 1.0].
      - Su score es mayor o igual a umbral (score >= umbral).

    En cualquier otro caso, la puerta permanece estrictamente CERRADA.
    """
    # 0. Verificación estricta del umbral
    if not es_valor_probabilidad_valido(umbral):
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=0.0,
            umbral=0.0,
            motivo_cierre=f"UMBRAL_INVALIDO ({umbral})",
            mensaje=f"ACCESO DENEGADO: Umbral inválido ({umbral})",
        )

    umb_val = float(umbral)

    # 1. Verificación del detector facial
    if estado_deteccion != "UN_ROSTRO":
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=0.0,
            umbral=umb_val,
            motivo_cierre=f"FALLO_DETECCION ({estado_deteccion})",
            mensaje=f"ACCESO DENEGADO: Fallo de detección ({estado_deteccion})",
        )

    # 2. Verificación de procesamiento técnico
    if not procesamiento_exitoso:
        detalle = error_procesamiento or "Error no especificado"
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=0.0,
            umbral=umb_val,
            motivo_cierre=f"ERROR_PROCESAMIENTO ({detalle})",
            mensaje=f"ACCESO DENEGADO: Error técnico de procesamiento ({detalle})",
        )

    # 3. Verificación de clase predicha
    if not clase_predicha:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=0.0,
            umbral=umb_val,
            motivo_cierre="SIN_PREDICCION",
            mensaje="ACCESO DENEGADO: Sin predicción disponible",
        )

    # 4. Verificación estricta del score/confianza
    if not es_valor_probabilidad_valido(score):
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=0.0,
            umbral=umb_val,
            motivo_cierre=f"SCORE_INVALIDO ({score})",
            mensaje=f"ACCESO DENEGADO: Score inválido ({score})",
        )

    return evaluar_decision_acceso(
        nombre_predicho=clase_predicha,
        confianza=float(score),
        umbral_confianza=umb_val,
        personas_autorizadas=personas_autorizadas,
    )

