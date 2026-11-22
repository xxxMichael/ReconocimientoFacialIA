"""Detector común de captura, evaluación e inferencia. Haar queda para comparación."""
from typing import Optional, Tuple
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet


def crear_detector(
    orientaciones: Optional[Tuple[int, ...]] = None,
    modo_multiorientacion: str = "estandar",
    alinear_rostro: bool = False,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
):
    # Error explícito si falta ONNX: nunca sustituir silenciosamente por Haar.
    return DetectorYuNet(
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        orientaciones=orientaciones,
        modo_multiorientacion=modo_multiorientacion,
        alinear_rostro=alinear_rostro,
    )
