"""
Evaluación comparativa offline de Fase 1 (compuerta de calidad sin reentrenar):
Compara 'CNN Estándar Portal Completo — 0.61' contra 'CNN Estándar 0.61 + Filtro Calidad (Fase 1)'
sobre fotos limpias y degradadas (portal 75, sesion_04 64, sesion_05 63).
"""
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.detector_yunet import EstadoDeteccion
from scripts.diagnosticar_confusiones_piloto import (
    _aplicar_perturbacion,
    _cargar_bgr_original,
    _clasificar_autorizado,
    _clasificar_desconocido,
)


def main() -> int:
    ahora = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_out = RAIZ / "resultados" / f"evaluacion_fase1_calidad_{ahora}"
    dir_out.mkdir(parents=True, exist_ok=True)

    p_std = PipelineCandidatoCongelado(RAIZ / "configuracion_candidato_estandar_portal_061.json", raiz_proyecto=RAIZ)
    p_cal = PipelineCandidatoCongelado(RAIZ / "configuracion_candidato_estandar_portal_061_calidad.json", raiz_proyecto=RAIZ)

    portal = sorted((RAIZ / "data/recepcion_externa/fotos").glob("*/*"))
    s04 = sorted((RAIZ / "data/propio/imagenes").glob("persona_*/sesion_04/**/*.jpg"))
    s05 = sorted((RAIZ / "data/propio/imagenes").glob("persona_*/sesion_05/**/*.jpg"))

    cache_port = {p: _cargar_bgr_original(p) for p in portal}
    cache_s04 = {p: _cargar_bgr_original(p) for p in s04}
    cache_s05 = {p: _cargar_bgr_original(p) for p in s05}
    cajas_port = {p: p_std.detector.detectar(img).recuadro for p, img in cache_port.items()}
    cajas_s04 = {p: p_std.detector.detectar(img).recuadro for p, img in cache_s04.items()}

    perts_clave = ["limpia", "oscura", "oscura_ruido", "oclusion_inferior", "oclusion_lateral", "rotacion_+15", "desenfoque"]
    rng = np.random.default_rng(42)

    resumen = {"portal": {}, "sesion_04": {}, "sesion_05_limpia": {}}
    for pert in perts_clave:
        resumen["portal"][pert] = {}
        resumen["sesion_04"][pert] = {}
        for nom, pipe in [("estandar_061", p_std), ("estandar_061_calidad", p_cal)]:
            fa = 0
            rech_cal = 0
            rech_otros = 0
            for p in portal:
                img_p, ok_c = _aplicar_perturbacion(cache_port[p], pert, cajas_port[p], rng)
                if not ok_c:
                    rech_otros += 1
                    continue
                det, dec, probs = pipe.evaluar_fotograma_completo(img_p)
                if probs is None and det.estado == EstadoDeteccion.UN_ROSTRO:
                    rech_cal += 1
                elif dec.estado_puerta == "ABIERTA":
                    fa += 1
                else:
                    rech_otros += 1
            resumen["portal"][pert][nom] = {
                "intentos": len(portal),
                "falsa_aceptacion": fa,
                "far_pct": round(100.0 * fa / len(portal), 2),
                "rechazo_calidad": rech_cal,
                "rechazo_modelo_o_det": rech_otros,
            }

            ac = conf = bloq_cal = bloq_otros = 0
            for p in s04:
                persona = p.parts[-4]
                img_p, ok_c = _aplicar_perturbacion(cache_s04[p], pert, cajas_s04[p], rng)
                if not ok_c:
                    bloq_otros += 1
                    continue
                det, dec, probs = pipe.evaluar_fotograma_completo(img_p)
                if probs is None and det.estado == EstadoDeteccion.UN_ROSTRO:
                    bloq_cal += 1
                else:
                    c_a = _clasificar_autorizado(persona, det, dec, probs, pipe)
                    if c_a["categoria"] == "acierto":
                        ac += 1
                    elif c_a["categoria"] == "confusion":
                        conf += 1
                    else:
                        bloq_otros += 1
            resumen["sesion_04"][pert][nom] = {
                "intentos": len(s04),
                "aciertos": ac,
                "confusiones": conf,
                "bloqueo_calidad": bloq_cal,
                "bloqueo_o_fallo_det": bloq_otros,
            }

    for nom, pipe in [("estandar_061", p_std), ("estandar_061_calidad", p_cal)]:
        ac = conf = bloq_cal = bloq_otros = 0
        for p in s05:
            persona = p.parts[-4]
            det, dec, probs = pipe.evaluar_fotograma_completo(cache_s05[p])
            if probs is None and det.estado == EstadoDeteccion.UN_ROSTRO:
                bloq_cal += 1
            else:
                c_a = _clasificar_autorizado(persona, det, dec, probs, pipe)
                if c_a["categoria"] == "acierto":
                    ac += 1
                elif c_a["categoria"] == "confusion":
                    conf += 1
                else:
                    bloq_otros += 1
        resumen["sesion_05_limpia"][nom] = {
            "intentos": len(s05),
            "aciertos": ac,
            "confusiones": conf,
            "bloqueo_calidad": bloq_cal,
            "bloqueo_o_fallo_det": bloq_otros,
        }

    (dir_out / "resumen_fase1.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
