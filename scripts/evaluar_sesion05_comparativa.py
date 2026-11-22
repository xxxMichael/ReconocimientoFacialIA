"""
Evaluación comparativa sobre Sesión 05 (prueba histórica de autorizados) entre:
- CNN Estándar Portal Completo — 0.61 (configuracion_candidato_estandar_portal_061.json)
- CNN Antialiasing — 0.52 (configuracion_candidato_antialiasing_052.json)

Audita el inventario de Sesión 05 primero, ejecuta el pipeline automático completo
(sin sustituir detecciones por cajas humanas), registra decisiones individuales,
probabilidades, matriz de confusión, tabla pareada y latencias.
No evalúa reserva LFW ni las 9 identidades universitarias históricas.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.adquisicion.propio import ManifiestoDataset, calcular_hash_archivo
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.detector import EstadoDeteccion


def ejecutar_evaluacion_sesion05():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"evaluacion_sesion05_antialiasing_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" EVALUACIÓN SOBRE SESIÓN 05 (PRUEBA HISTÓRICA DE AUTORIZADOS)")
    print("=" * 75)
    print(f" Directorio de salida: {dir_salida}")

    # 1. Cargar y verificar ambos candidatos congelados
    ruta_cfg_std = RAIZ / "configuracion_candidato_estandar_portal_061.json"
    ruta_cfg_aa = RAIZ / "configuracion_candidato_antialiasing_052.json"
    pipe_std = PipelineCandidatoCongelado(ruta_cfg_std, raiz_proyecto=RAIZ)
    pipe_aa = PipelineCandidatoCongelado(ruta_cfg_aa, raiz_proyecto=RAIZ)

    hashes_iniciales = {
        "modelo_estandar": calcular_hash_archivo(pipe_std.ruta_modelo),
        "modelo_antialiasing": calcular_hash_archivo(pipe_aa.ruta_modelo),
        "detector_yunet": calcular_hash_archivo(pipe_std.ruta_detector),
        "cfg_estandar": calcular_hash_archivo(ruta_cfg_std),
        "cfg_antialiasing": calcular_hash_archivo(ruta_cfg_aa),
    }

    # 2. Auditar inventario de Sesión 05 en manifiesto_local.csv (o manifiesto.csv)
    ruta_man = RAIZ / "data" / "propio" / "manifiesto_local.csv"
    if not ruta_man.exists():
        ruta_man = RAIZ / "data" / "propio" / "manifiesto.csv"
    man = ManifiestoDataset(ruta_man)

    todos_s05 = [
        r for r in man.registros
        if r.sesion == "sesion_05" and r.tipo_sujeto == "autorizado"
    ]
    excluidos_s05 = [r for r in todos_s05 if (not r.consentimiento or r.decision_humana == "excluida")]
    elegibles_s05 = [
        r for r in todos_s05
        if r.consentimiento and r.decision_humana != "excluida"
    ]
    elegibles_s05.sort(key=lambda r: (r.persona, r.pose, r.identificador))

    print(f" Auditoría Sesión 05 ({ruta_man.name}):")
    print(f"   Total filas sesion_05: {len(todos_s05)} | Excluidos protocolo: {len(excluidos_s05)} | Elegibles: {len(elegibles_s05)}")
    conteo_persona = {}
    for r in elegibles_s05:
        conteo_persona[r.persona] = conteo_persona.get(r.persona, 0) + 1
    print(f"   Por persona: {conteo_persona}")

    # 3. Evaluar cada imagen con ambos pipelines automáticos
    res_std, res_aa, pareados, filas_csv = [], [], [], []

    for r in elegibles_s05:
        ruta_img = RAIZ / r.ruta
        frame_bgr = cv2.imread(str(ruta_img))
        if frame_bgr is None:
            raise ValueError(f"Error técnico leyendo {ruta_img}")
        h_img = calcular_hash_archivo(ruta_img)

        def evaluar_uno(pipe: PipelineCandidatoCongelado, etiqueta: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            det = pipe.detector.detectar(frame_bgr, es_pre_recortada=False)
            t_det_prep = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            if det.estado != EstadoDeteccion.UN_ROSTRO or det.rostro_recortado is None:
                from reconocimiento.clasificacion.regla_acceso import evaluar_regla_acceso_completa
                dec = evaluar_regla_acceso_completa(
                    estado_deteccion=det.estado.value, procesamiento_exitoso=False,
                    clase_predicha=None, score=0.0, umbral=pipe.umbral_score,
                    personas_autorizadas=pipe.personas_autorizadas,
                )
                probs = None
                t_prep_clf = 0.0
                cat = "fallo_deteccion"
                pred_nom = None
                score = 0.0
                pvec = [None, None, None, None]
            else:
                rec = det.recorte_alineado if (pipe.usar_alineacion and det.recorte_alineado is not None) else det.rostro_recortado
                dec, probs = pipe.evaluar_recorte(rec)
                t_prep_clf = (time.perf_counter() - t1) * 1000.0
                pred_idx = int(np.argmax(probs))
                pred_nom = pipe.mapeo_id_clase.get(pred_idx, "desconocido")
                score = float(probs[pred_idx])
                pvec = [round(float(x), 4) for x in probs]

                if dec.estado_puerta == "ABIERTA":
                    cat = "apertura_correcta" if pred_nom == r.persona else "confusion_autorizado"
                elif pred_nom == "desconocido":
                    cat = "bloqueo_desconocido"
                else:
                    cat = "bloqueo_umbral"

            t_total = (time.perf_counter() - t0) * 1000.0
            motivo = dec.motivo_cierre if dec.estado_puerta == "CERRADA" else dec.mensaje

            return {
                "identificador": r.identificador,
                "persona_real": r.persona,
                "pose": r.pose,
                "hash_imagen": h_img,
                "modelo": etiqueta,
                "umbral": pipe.umbral_score,
                "estado_deteccion": det.estado.value,
                "clase_argmax": pred_nom,
                "score_argmax": round(score, 4) if probs is not None else None,
                "prob_persona_1": pvec[0],
                "prob_persona_2": pvec[1],
                "prob_persona_3": pvec[2],
                "prob_desconocido": pvec[3],
                "decision_puerta": dec.estado_puerta,
                "categoria": cat,
                "motivo": motivo,
                "tiempo_det_alin_ms": round(t_det_prep, 2),
                "tiempo_prep_cls_ms": round(t_prep_clf, 2),
                "tiempo_total_ms": round(t_total, 2),
            }

        it_std = evaluar_uno(pipe_std, "CNN Estándar Portal Completo — 0.61")
        it_aa = evaluar_uno(pipe_aa, "CNN Antialiasing — 0.52")
        res_std.append(it_std)
        res_aa.append(it_aa)
        filas_csv.extend([it_std, it_aa])

        ok_std = (it_std["categoria"] == "apertura_correcta")
        ok_aa = (it_aa["categoria"] == "apertura_correcta")
        pareados.append({
            "identificador": r.identificador,
            "persona_real": r.persona,
            "pose": r.pose,
            "ambas_correctas": ok_std and ok_aa,
            "solo_estandar_correcta": ok_std and (not ok_aa),
            "solo_antialiasing_correcta": (not ok_std) and ok_aa,
            "ninguna_correcta": (not ok_std) and (not ok_aa),
            "std_cat": it_std["categoria"],
            "std_pred": it_std["clase_argmax"],
            "std_score": it_std["score_argmax"],
            "aa_cat": it_aa["categoria"],
            "aa_pred": it_aa["clase_argmax"],
            "aa_score": it_aa["score_argmax"],
        })

    # Guardar CSV y JSON individuales
    ruta_csv = dir_salida / "predicciones_sesion05.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
        writer.writeheader()
        writer.writerows(filas_csv)

    # Resumir métricas
    def resumir(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        tot = len(items)
        aciertos = sum(1 for x in items if x["categoria"] == "apertura_correcta")
        conf = sum(1 for x in items if x["categoria"] == "confusion_autorizado")
        bloq_desc = sum(1 for x in items if x["categoria"] == "bloqueo_desconocido")
        bloq_umb = sum(1 for x in items if x["categoria"] == "bloqueo_umbral")
        fallo_det = sum(1 for x in items if x["categoria"] == "fallo_deteccion")
        bloqueos = bloq_desc + bloq_umb + fallo_det
        assert tot == aciertos + conf + bloqueos, "Debe cumplirse intentos == aciertos + confusiones + bloqueos"

        por_persona = {}
        for p in ("persona_1", "persona_2", "persona_3"):
            sub = [x for x in items if x["persona_real"] == p]
            s_ok = sum(1 for x in sub if x["categoria"] == "apertura_correcta")
            s_conf = sum(1 for x in sub if x["categoria"] == "confusion_autorizado")
            s_bloq = len(sub) - s_ok - s_conf
            por_persona[p] = {
                "intentos": len(sub), "correctas": s_ok, "confusiones": s_conf,
                "bloqueos": s_bloq, "frr_pct": round(100.0 * s_bloq / len(sub), 2) if sub else 0.0,
            }

        por_pose = {}
        for pose in ("frontal", "izquierda", "derecha", "arriba", "abajo"):
            sub = [x for x in items if x["pose"] == pose]
            s_ok = sum(1 for x in sub if x["categoria"] == "apertura_correcta")
            s_conf = sum(1 for x in sub if x["categoria"] == "confusion_autorizado")
            s_bloq = len(sub) - s_ok - s_conf
            por_pose[pose] = {"intentos": len(sub), "correctas": s_ok, "confusiones": s_conf, "bloqueos": s_bloq}

        # Matriz de confusión 3x4 (real -> predicha)
        matriz = {p: {"persona_1": 0, "persona_2": 0, "persona_3": 0, "desconocido_o_bloqueo": 0} for p in ("persona_1", "persona_2", "persona_3")}
        for x in items:
            if x["decision_puerta"] == "ABIERTA":
                matriz[x["persona_real"]][x["clase_argmax"]] += 1
            else:
                matriz[x["persona_real"]]["desconocido_o_bloqueo"] += 1

        return {
            "intentos_totales": tot,
            "aperturas_correctas": aciertos,
            "tasa_identificacion_correcta_pct": round(100.0 * aciertos / tot, 2),
            "confusiones_autorizados": conf,
            "bloqueos_totales": bloqueos,
            "bloqueos_desconocido": bloq_desc,
            "bloqueos_umbral": bloq_umb,
            "fallos_deteccion": fallo_det,
            "frr_puerta_pct": round(100.0 * bloqueos / tot, 2),
            "por_persona": por_persona,
            "por_pose": por_pose,
            "matriz_confusion": matriz,
            "latencia_det_alin_media_ms": round(float(np.mean([x["tiempo_det_alin_ms"] for x in items])), 2),
            "latencia_total_media_ms": round(float(np.mean([x["tiempo_total_ms"] for x in items])), 2),
        }

    met_std = resumir(res_std)
    met_aa = resumir(res_aa)
    tabla_pareada = {
        "ambas_correctas": sum(1 for p in pareados if p["ambas_correctas"]),
        "solo_estandar_correcta": sum(1 for p in pareados if p["solo_estandar_correcta"]),
        "solo_antialiasing_correcta": sum(1 for p in pareados if p["solo_antialiasing_correcta"]),
        "ninguna_correcta": sum(1 for p in pareados if p["ninguna_correcta"]),
    }

    # Verificar hashes finales
    hashes_finales = {
        "modelo_estandar": calcular_hash_archivo(pipe_std.ruta_modelo),
        "modelo_antialiasing": calcular_hash_archivo(pipe_aa.ruta_modelo),
        "detector_yunet": calcular_hash_archivo(pipe_std.ruta_detector),
        "cfg_estandar": calcular_hash_archivo(ruta_cfg_std),
        "cfg_antialiasing": calcular_hash_archivo(ruta_cfg_aa),
    }
    assert hashes_iniciales == hashes_finales, "Los hashes cambiaron durante la evaluación"

    resumen = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "alcance": "Comparación sobre prueba histórica Sesión 05 (únicamente autorizados; no permite calcular FAR de desconocidos).",
        "inventario_sesion_05": {
            "total_filas": len(todos_s05),
            "excluidos": len(excluidos_s05),
            "elegibles": len(elegibles_s05),
            "por_persona": conteo_persona,
        },
        "hashes_verificados": hashes_finales,
        "estandar_portal_completo_061": met_std,
        "antialiasing_052": met_aa,
        "tabla_pareada": tabla_pareada,
        "diferencias_pareadas": [p for p in pareados if not p["ambas_correctas"]],
    }
    (dir_salida / "resumen_sesion05.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 75)
    print(" RESULTADOS SOBRE SESIÓN 05 (63 INTENTOS AUTORIZADOS)")
    print("=" * 75)
    print(f" Estándar (0.61)    : Correctas={met_std['aperturas_correctas']}/{met_std['intentos_totales']} ({met_std['tasa_identificacion_correcta_pct']}%) | Conf={met_std['confusiones_autorizados']} | Bloq={met_std['bloqueos_totales']} (FRR={met_std['frr_puerta_pct']}%) | DetFallo={met_std['fallos_deteccion']}")
    print(f" Antialiasing (0.52): Correctas={met_aa['aperturas_correctas']}/{met_aa['intentos_totales']} ({met_aa['tasa_identificacion_correcta_pct']}%) | Conf={met_aa['confusiones_autorizados']} | Bloq={met_aa['bloqueos_totales']} (FRR={met_aa['frr_puerta_pct']}%) | DetFallo={met_aa['fallos_deteccion']}")
    print(f" Pareado: Ambas OK={tabla_pareada['ambas_correctas']} | Solo Estándar={tabla_pareada['solo_estandar_correcta']} | Solo Antialiasing={tabla_pareada['solo_antialiasing_correcta']} | Ninguna={tabla_pareada['ninguna_correcta']}")
    print("=" * 75)
    return resumen, dir_salida


if __name__ == "__main__":
    ejecutar_evaluacion_sesion05()
