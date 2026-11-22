"""
Compuerta opcional de calidad facial previa a la clasificación (Fase 1).
Permite rechazar fotogramas con oscuridad severa, bajo contraste, oclusión inferior
o tamaño facial insuficiente sin modificar los modelos ni el preprocesamiento existente.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class ConfiguracionCalidad:
    habilitado: bool = False
    min_luminancia_media: float = 58.0
    min_contraste_std: float = 18.0
    max_fraccion_oscura_inferior: float = 0.35
    min_gradiente_inferior: float = 4.2
    min_tamano_rostro_px: int = 36
    margen_minimo: float = 0.25

    @classmethod
    def desde_diccionario(cls, data: Optional[Dict[str, Any]]) -> "ConfiguracionCalidad":
        if not data or not data.get("habilitado", False):
            return cls(habilitado=False)
        return cls(
            habilitado=True,
            min_luminancia_media=float(data.get("min_luminancia_media", 58.0)),
            min_contraste_std=float(data.get("min_contraste_std", 18.0)),
            max_fraccion_oscura_inferior=float(data.get("max_fraccion_oscura_inferior", 0.35)),
            min_gradiente_inferior=float(data.get("min_gradiente_inferior", 4.2)),
            min_tamano_rostro_px=int(data.get("min_tamano_rostro_px", 36)),
            margen_minimo=float(data.get("margen_minimo", 0.25)),
        )


@dataclass
class ResultadoCalidad:
    aprobada: bool
    codigo: str
    motivo: str
    metricas: Dict[str, float] = field(default_factory=dict)


def evaluar_calidad_rostro(
    recorte_bgr: np.ndarray,
    recuadro: Optional[Tuple[int, int, int, int]] = None,
    config: Optional[ConfiguracionCalidad] = None,
) -> ResultadoCalidad:
    """Evalúa iluminación, contraste, oclusión inferior y tamaño mínimo del rostro."""
    cfg = config or ConfiguracionCalidad(habilitado=True)
    if recorte_bgr is None or recorte_bgr.size == 0:
        return ResultadoCalidad(False, "CALIDAD_VACIA", "Recorte facial vacío.")

    if recuadro is not None and len(recuadro) == 4:
        min_dim = min(int(recuadro[2]), int(recuadro[3]))
        if min_dim < cfg.min_tamano_rostro_px:
            return ResultadoCalidad(
                False,
                "ROSTRO_MUY_PEQUENO",
                f"Calidad insuficiente: rostro demasiado pequeño ({min_dim}px < {cfg.min_tamano_rostro_px}px).",
                {"min_dim_px": float(min_dim)},
            )

    if len(recorte_bgr.shape) == 3:
        gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gris = recorte_bgr.astype(np.float32)

    if gris.shape[:2] != (64, 64):
        gris = cv2.resize(gris, (64, 64), interpolation=cv2.INTER_AREA)

    lum = float(np.mean(gris))
    std = float(np.std(gris))
    inf = gris[42:, :]
    frac_osc_inf = float(np.mean(inf < 15.0))
    grad_x = float(np.mean(np.abs(np.diff(inf, axis=1))))
    grad_y = float(np.mean(np.abs(np.diff(inf, axis=0))))
    grad_inf = grad_x + grad_y

    metricas = {
        "luminancia_media": round(lum, 2),
        "contraste_std": round(std, 2),
        "fraccion_oscura_inferior": round(frac_osc_inf, 4),
        "gradiente_inferior": round(grad_inf, 2),
    }

    if lum < cfg.min_luminancia_media:
        return ResultadoCalidad(
            False,
            "ILUMINACION_INSUFICIENTE",
            f"Calidad insuficiente: poca iluminación (lum={lum:.1f} < {cfg.min_luminancia_media:.1f}).",
            metricas,
        )

    if std < cfg.min_contraste_std:
        return ResultadoCalidad(
            False,
            "CONTRASTE_INSUFICIENTE",
            f"Calidad insuficiente: contraste bajo (std={std:.1f} < {cfg.min_contraste_std:.1f}).",
            metricas,
        )

    if frac_osc_inf > cfg.max_fraccion_oscura_inferior:
        return ResultadoCalidad(
            False,
            "OCLUSION_INFERIOR",
            f"Calidad insuficiente: rostro parcialmente cubierto en zona inferior ({frac_osc_inf*100:.0f}%).",
            metricas,
        )

    if grad_inf < cfg.min_gradiente_inferior:
        return ResultadoCalidad(
            False,
            "DETALLE_INFERIOR_INSUFICIENTE",
            f"Calidad insuficiente: zona inferior sin detalle facial suficiente (grad={grad_inf:.1f}).",
            metricas,
        )

    return ResultadoCalidad(True, "OK", "Calidad facial adecuada.", metricas)
