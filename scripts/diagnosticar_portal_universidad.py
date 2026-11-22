"""
Diagnóstico comparativo sobre las 75 capturas del portal (david, externo_piloto, paul, sonia, stevenl)
entre la CNN alineada anterior (th=0.84) y la nueva CNN universitaria (th=0.50).

Separa explícitamente:
- 15 capturas utilizadas en el entrenamiento del nuevo modelo (frontal/media, izquierda/media, derecha/media)
- 60 capturas restantes del portal (no vistas en entrenamiento, pero de las mismas 5 identidades)
- Total (75 capturas)

No modifica modelos, umbrales ni el modelo activo de la web. No abre ni evalúa las 9 imágenes
universitarias reservadas, Sesión 05 ni la reserva LFW.
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
from scripts.experimentar_4clases_universidad import POSES_PORTAL_SELECCIONADAS


def ejecutar_diagnostico_portal():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"diagnostico_portal_universidad_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" DIAGNÓSTICO DE DESARROLLO SOBRE LAS 75 CAPTURAS DEL PORTAL")
    print("=" * 75)
    print(f" Directorio de salida: {dir_salida}")

    # 1. Cargar los dos candidatos congelados/propuestos con su propio preprocesamiento
    ruta_cfg_ant = RAIZ / "configuracion_candidato_alineado.json"
    ruta_cfg_nvo = (
        RAIZ
        / "resultados"
        / "exp_4clases_universidad_20261120_192345"
        / "configuracion_candidato_universidad_propuesto.json"
    )

    pipe_ant = PipelineCandidatoCongelado(ruta_cfg_ant, raiz_proyecto=RAIZ)
    pipe_nvo = PipelineCandidatoCongelado(ruta_cfg_nvo, raiz_proyecto=RAIZ)

    print(f" [Anterior] {pipe_ant.nombre} | Umbral: {pipe_ant.umbral_score}")
    print(f" [Nuevo]    {pipe_nvo.nombre} | Umbral: {pipe_nvo.umbral_score}")

    # 2. Cargar las 75 capturas del portal desde manifiesto_local.csv
    ruta_manifiesto = RAIZ / "data" / "propio" / "manifiesto_local.csv"
    manifiesto = ManifiestoDataset(ruta_manifiesto)
    regs_portal = [
        r for r in manifiesto.registros
        if r.procedencia == "portal_web_externo" and r.particion == "prueba"
    ]
    regs_portal.sort(key=lambda r: (r.persona, r.pose, r.distancia or ""))

    if len(regs_portal) != 75:
        raise RuntimeError(f"Se esperaban 75 capturas del portal, encontradas: {len(regs_portal)}")

    # Identificar exactamente qué 15 capturas fueron usadas en entrenamiento
    # (frontal/media, izquierda/media, derecha/media por cada uno de los 5 sujetos)
    def fue_usada_en_entrenamiento(reg) -> bool:
        return (reg.pose, reg.distancia) in POSES_PORTAL_SELECCIONADAS

    res_ant = []
    res_nvo = []
    pareados = []
    filas_csv = []

    for reg in regs_portal:
        ruta_img = RAIZ / reg.ruta
        frame_bgr = cv2.imread(str(ruta_img))
        if frame_bgr is None:
            raise ValueError(f"Error técnico leyendo imagen: {ruta_img}")
        sha256 = calcular_hash_archivo(ruta_img)
        usada_train = fue_usada_en_entrenamiento(reg)
        grupo_train = "utilizadas_15" if usada_train else "restantes_60"

        def evaluar_con_pipeline(pipe: PipelineCandidatoCongelado, etiqueta_modelo: str):
            t0 = time.perf_counter()
            det, dec, probs = pipe.evaluar_fotograma_completo(frame_bgr)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            if det.estado != EstadoDeteccion.UN_ROSTRO:
                cat = "fallo_deteccion"
                pred_nombre = None
                score = 0.0
                pvec = [0.0, 0.0, 0.0, 0.0]
            elif dec.motivo_cierre and "error" in dec.motivo_cierre.lower():
                cat = "error_tecnico"
                pred_nombre = None
                score = 0.0
                pvec = [0.0, 0.0, 0.0, 0.0]
            else:
                pred_idx = int(np.argmax(probs))
                pred_nombre = pipe.mapeo_id_clase.get(pred_idx, "desconocido")
                score = float(probs[pred_idx])
                pvec = [float(x) for x in probs]
                if dec.estado_puerta == "ABIERTA":
                    cat = "falsa_aceptacion"
                elif pred_nombre == "desconocido":
                    cat = "clasificacion_desconocido"
                else:
                    cat = "rechazo_umbral"

            motivo = dec.motivo_cierre if dec.estado_puerta == "CERRADA" else f"Acceso concedido ({pred_nombre}, score={score:.4f} >= {pipe.umbral_score})"
            item = {
                "identificador": reg.identificador,
                "hash_sha256": sha256,
                "participante": reg.persona,
                "pose": reg.pose,
                "distancia": reg.distancia,
                "usada_en_entrenamiento": usada_train,
                "grupo_entrenamiento": grupo_train,
                "modelo": etiqueta_modelo,
                "umbral": pipe.umbral_score,
                "estado_deteccion": det.estado.value,
                "clase_argmax": pred_nombre,
                "score_argmax": round(score, 4),
                "prob_persona_1": round(pvec[0], 4),
                "prob_persona_2": round(pvec[1], 4),
                "prob_persona_3": round(pvec[2], 4),
                "prob_desconocido": round(pvec[3], 4),
                "decision_final": dec.estado_puerta,
                "categoria": cat,
                "motivo": motivo,
                "latencia_ms": round(lat_ms, 2),
            }
            return item

        item_ant = evaluar_con_pipeline(pipe_ant, "anterior_cnn_alineado_084")
        item_nvo = evaluar_con_pipeline(pipe_nvo, "nueva_cnn_universidad_050")
        res_ant.append(item_ant)
        res_nvo.append(item_nvo)
        filas_csv.extend([item_ant, item_nvo])

        ab_ant = item_ant["decision_final"] == "ABIERTA"
        ab_nvo = item_nvo["decision_final"] == "ABIERTA"
        pareados.append({
            "identificador": reg.identificador,
            "participante": reg.persona,
            "pose": reg.pose,
            "distancia": reg.distancia,
            "usada_en_entrenamiento": usada_train,
            "ambos_rechazan": (not ab_ant) and (not ab_nvo),
            "solo_anterior_acepta": ab_ant and (not ab_nvo),
            "solo_nuevo_acepta": (not ab_ant) and ab_nvo,
            "ambos_aceptan": ab_ant and ab_nvo,
        })

    # Comprobaciones de integridad
    ids_util = {r["identificador"] for r in res_nvo if r["usada_en_entrenamiento"]}
    ids_rest = {r["identificador"] for r in res_nvo if not r["usada_en_entrenamiento"]}
    assert len(ids_util) == 15, f"Se esperaban 15 utilizadas, encontradas {len(ids_util)}"
    assert len(ids_rest) == 60, f"Se esperaban 60 restantes, encontradas {len(ids_rest)}"
    assert len(ids_util & ids_rest) == 0, "Solapamiento entre utilizadas y restantes"
    assert len(ids_util | ids_rest) == 75, "La unión debe sumar 75"

    # Guardar CSV de decisiones individuales
    ruta_csv = dir_salida / "decisiones_individuales_portal.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
        writer.writeheader()
        writer.writerows(filas_csv)

    # Función auxiliar de métricas
    def resumir_subconjunto(lista_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        tot = len(lista_items)
        fa = [x for x in lista_items if x["categoria"] == "falsa_aceptacion"]
        desc = sum(1 for x in lista_items if x["categoria"] == "clasificacion_desconocido")
        umb = sum(1 for x in lista_items if x["categoria"] == "rechazo_umbral")
        det_f = sum(1 for x in lista_items if x["categoria"] == "fallo_deteccion")
        err_t = sum(1 for x in lista_items if x["categoria"] == "error_tecnico")
        assert len(fa) + desc + umb + det_f + err_t == tot

        clases_fa = {}
        for x in fa:
            c = x["clase_argmax"] or "desconocida"
            clases_fa[c] = clases_fa.get(c, 0) + 1

        return {
            "intentos": tot,
            "falsas_aceptaciones": len(fa),
            "far_pct": round(100.0 * len(fa) / tot, 2) if tot > 0 else 0.0,
            "clasif_desconocido": desc,
            "rechazo_umbral": umb,
            "fallo_deteccion": det_f,
            "error_tecnico": err_t,
            "clases_asignadas_en_fa": clases_fa,
        }

    def construir_reporte_modelo(lista_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        rep = {
            "global_75": resumir_subconjunto(lista_items),
            "utilizadas_15": resumir_subconjunto([x for x in lista_items if x["usada_en_entrenamiento"]]),
            "restantes_60": resumir_subconjunto([x for x in lista_items if not x["usada_en_entrenamiento"]]),
            "por_participante": {},
            "por_pose": {},
            "por_distancia": {},
        }
        for p in sorted({x["participante"] for x in lista_items}):
            rep["por_participante"][p] = resumir_subconjunto([x for x in lista_items if x["participante"] == p])
        for pose in ("frontal", "izquierda", "derecha", "arriba", "abajo"):
            rep["por_pose"][pose] = resumir_subconjunto([x for x in lista_items if x["pose"] == pose])
        for dist in ("cercana", "media", "lejana"):
            rep["por_distancia"][dist] = resumir_subconjunto([x for x in lista_items if x["distancia"] == dist])
        return rep

    def tabla_pareada_sub(lista_p: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            "total": len(lista_p),
            "ambos_rechazan": sum(1 for x in lista_p if x["ambos_rechazan"]),
            "solo_anterior_acepta": sum(1 for x in lista_p if x["solo_anterior_acepta"]),
            "solo_nuevo_acepta": sum(1 for x in lista_p if x["solo_nuevo_acepta"]),
            "ambos_aceptan": sum(1 for x in lista_p if x["ambos_aceptan"]),
        }

    diagnostico = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "etiqueta_metodologica": "Diagnóstico de desarrollo sobre identidades conocidas durante el entrenamiento. Ni las 15 utilizadas ni las 60 restantes constituyen una prueba independiente de rechazo de personas nuevas.",
        "anterior_cnn_alineado_084": construir_reporte_modelo(res_ant),
        "nueva_cnn_universidad_050": construir_reporte_modelo(res_nvo),
        "comparacion_pareada": {
            "global_75": tabla_pareada_sub(pareados),
            "utilizadas_15": tabla_pareada_sub([x for x in pareados if x["usada_en_entrenamiento"]]),
            "restantes_60": tabla_pareada_sub([x for x in pareados if not x["usada_en_entrenamiento"]]),
        },
        "hashes_configuraciones": {
            "anterior": calcular_hash_archivo(ruta_cfg_ant),
            "nuevo": calcular_hash_archivo(ruta_cfg_nvo),
        },
    }

    with open(dir_salida / "diagnostico_portal_resumen.json", "w", encoding="utf-8") as f:
        json.dump(diagnostico, f, indent=2, ensure_ascii=False)

    # Imprimir resumen en consola
    for grp_key, grp_label in [("global_75", "GLOBAL (75)"), ("utilizadas_15", "UTILIZADAS EN TRAIN (15)"), ("restantes_60", "RESTANTES NO VISTAS (60)")]:
        a = diagnostico["anterior_cnn_alineado_084"][grp_key]
        n = diagnostico["nueva_cnn_universidad_050"][grp_key]
        tp = diagnostico["comparacion_pareada"][grp_key]
        print(f"\n--- {grp_label} ---")
        print(f"  Anterior (0.84): FAR={a['falsas_aceptaciones']}/{a['intentos']} ({a['far_pct']}%) | Desc={a['clasif_desconocido']} | Umb={a['rechazo_umbral']} | DetFallo={a['fallo_deteccion']} | Clases FA={a['clases_asignadas_en_fa']}")
        print(f"  Nuevo    (0.50): FAR={n['falsas_aceptaciones']}/{n['intentos']} ({n['far_pct']}%) | Desc={n['clasif_desconocido']} | Umb={n['rechazo_umbral']} | DetFallo={n['fallo_deteccion']} | Clases FA={n['clases_asignadas_en_fa']}")
        print(f"  Pareado: Ambos rechazan={tp['ambos_rechazan']} | Solo anterior={tp['solo_anterior_acepta']} | Solo nuevo={tp['solo_nuevo_acepta']} | Ambos aceptan={tp['ambos_aceptan']}")

    print(f"\nDiagnóstico guardado en: {dir_salida}")
    return diagnostico, dir_salida


if __name__ == "__main__":
    ejecutar_diagnostico_portal()
