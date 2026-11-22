"""
Evaluación final independiente sobre las 9 identidades universitarias reservadas
(particion = prueba_reservada en data/universidad_desc/manifiesto_universidad.csv).

Compara:
- Candidato Nuevo Congelado: CNN 4 Clases (Portal Completo + Balance 2 Niveles), umbral 0.61
- Referencia Congelada: CNN 4 Clases (Alineado Anterior), umbral 0.84

No reentrena, no ajusta umbrales, no cambia el modelo activo de la web y no evalúa
Sesión 05 ni la reserva LFW.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.adquisicion.propio import calcular_hash_archivo
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.detector import EstadoDeteccion


def ejecutar_evaluacion_final_universidad():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"evaluacion_final_universidad_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_laminas = dir_salida / "laminas"
    dir_laminas.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" EVALUACIÓN FINAL: 9 IDENTIDADES UNIVERSITARIAS RESERVADAS")
    print("=" * 75)
    print(f" Directorio de salida: {dir_salida}")

    # 1. Fijar y verificar ambos candidatos antes de evaluar
    ruta_cfg_nuevo_src = (
        RAIZ
        / "resultados"
        / "exp_4clases_portal_completo_20261120_200452"
        / "configuracion_candidato_portal_completo_propuesto.json"
    )
    ruta_cfg_ref_src = RAIZ / "configuracion_candidato_alineado.json"

    pipe_nuevo = PipelineCandidatoCongelado(ruta_cfg_nuevo_src, raiz_proyecto=RAIZ)
    pipe_ref = PipelineCandidatoCongelado(ruta_cfg_ref_src, raiz_proyecto=RAIZ)

    assert abs(pipe_nuevo.umbral_score - 0.61) < 1e-9, f"Umbral inesperado nuevo: {pipe_nuevo.umbral_score}"
    assert abs(pipe_ref.umbral_score - 0.84) < 1e-9, f"Umbral inesperado ref: {pipe_ref.umbral_score}"

    # Guardar especificaciones congeladas en el directorio de evaluación
    entorno_info = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }

    def construir_especificacion_congelada(pipe: PipelineCandidatoCongelado, justificacion: str) -> Dict[str, Any]:
        spec = dict(pipe.config)
        spec["congelamiento_evaluacion_final"] = {
            "fecha_utc": datetime.now(timezone.utc).isoformat(),
            "sha256_modelo_verificado": calcular_hash_archivo(pipe.ruta_modelo),
            "sha256_detector_verificado": calcular_hash_archivo(pipe.ruta_detector),
            "umbral_congelado": pipe.umbral_score,
            "orden_clases": pipe.orden_clases,
            "personas_autorizadas": pipe.personas_autorizadas,
            "justificacion_seleccion": justificacion,
            "versiones_entorno": entorno_info,
        }
        return spec

    spec_nuevo = construir_especificacion_congelada(
        pipe_nuevo,
        "Seleccionado tras lograr mayor cobertura de entrenamiento (72/75 capturas del portal con balance por identidad y clase) sin deterioro en validación (FRR 1.56%, FAR 0.00% empatado con el candidato alineado anterior; las cifras sobre el portal son rendimiento de entrenamiento).",
    )
    spec_ref = construir_especificacion_congelada(
        pipe_ref,
        "Candidato alineado anterior de referencia congelado con umbral 0.84.",
    )

    (dir_salida / "configuracion_congelada_nuevo.json").write_text(
        json.dumps(spec_nuevo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dir_salida / "configuracion_congelada_referencia.json").write_text(
        json.dumps(spec_ref, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2. Cargar y auditar exclusivamente particion = prueba_reservada
    ruta_manifiesto = RAIZ / "data" / "universidad_desc" / "manifiesto_universidad.csv"
    with ruta_manifiesto.open(encoding="utf-8", newline="") as f:
        todas_filas = list(csv.DictReader(f))

    filas_dev = [r for r in todas_filas if r["particion"] in ("entrenamiento", "validacion")]
    filas_reserva = [r for r in todas_filas if r["particion"] == "prueba_reservada"]

    if len(filas_reserva) != 9:
        raise RuntimeError(f"Se esperaban exactamente 9 filas en prueba_reservada, encontradas: {len(filas_reserva)}")

    ids_reserva = {r["identificador"] for r in filas_reserva}
    ids_dev = {r["identificador"] for r in filas_dev}
    hashes_dev = {r["hash_sha256"] for r in filas_dev}

    if len(ids_reserva) != 9:
        raise RuntimeError("Los identificadores de prueba_reservada no son 9 distintos.")
    if ids_reserva & ids_dev:
        raise RuntimeError(f"Solapamiento de IDs con desarrollo: {ids_reserva & ids_dev}")

    for r in filas_reserva:
        ruta_img = RAIZ / r["ruta"]
        if not ruta_img.exists():
            raise FileNotFoundError(f"Imagen de reserva no encontrada: {ruta_img}")
        h_real = calcular_hash_archivo(ruta_img)
        if h_real != r["hash_sha256"]:
            raise ValueError(f"Discrepancia SHA-256 en {r['identificador']}")
        if h_real in hashes_dev:
            raise ValueError(f"Solapamiento de hash con desarrollo en {r['identificador']}")

    # Guardar manifiesto evaluado
    with open(dir_salida / "manifiesto_evaluado_prueba_reservada.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_reserva[0].keys()))
        writer.writeheader()
        writer.writerows(filas_reserva)

    print(f" Verificación de reserva: {len(filas_reserva)} identidades únicas sin solapamiento.")

    # 3. Evaluar las mismas 9 imágenes con ambos candidatos congelados
    res_ref = []
    res_nuevo = []
    pareados = []
    filas_csv = []

    for r in filas_reserva:
        ruta_img = RAIZ / r["ruta"]
        frame_bgr = cv2.imread(str(ruta_img))
        if frame_bgr is None:
            raise ValueError(f"No se pudo decodificar imagen {ruta_img}")

        def evaluar_intento(pipe: PipelineCandidatoCongelado, etiqueta: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            det, dec, probs = pipe.evaluar_fotograma_completo(frame_bgr)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            if det.estado != EstadoDeteccion.UN_ROSTRO:
                cat = "fallo_deteccion"
                pred_nom = None
                score = 0.0
                pvec = [0.0, 0.0, 0.0, 0.0]
            elif dec.motivo_cierre and "error" in dec.motivo_cierre.lower():
                cat = "error_tecnico"
                pred_nom = None
                score = 0.0
                pvec = [0.0, 0.0, 0.0, 0.0]
            else:
                pred_idx = int(np.argmax(probs))
                pred_nom = pipe.mapeo_id_clase.get(pred_idx, "desconocido")
                score = float(probs[pred_idx])
                pvec = [float(x) for x in probs]
                if dec.estado_puerta == "ABIERTA":
                    cat = "falsa_aceptacion"
                elif pred_nom == "desconocido":
                    cat = "clasificacion_desconocido"
                else:
                    cat = "rechazo_umbral"

            motivo = dec.motivo_cierre if dec.estado_puerta == "CERRADA" else f"Acceso concedido ({pred_nom}, score={score:.4f} >= {pipe.umbral_score})"
            recorte_usado = (
                det.recorte_alineado
                if (pipe.usar_alineacion and det.recorte_alineado is not None)
                else det.rostro_recortado
            )
            return {
                "identificador": r["identificador"],
                "ruta": r["ruta"],
                "hash_sha256": r["hash_sha256"],
                "modelo": etiqueta,
                "umbral": pipe.umbral_score,
                "estado_deteccion": det.estado.value,
                "total_rostros": det.total_rostros_detectados,
                "recuadro": list(det.recuadro) if det.recuadro else None,
                "clase_argmax": pred_nom,
                "score_argmax": round(score, 4),
                "prob_persona_1": round(pvec[0], 4),
                "prob_persona_2": round(pvec[1], 4),
                "prob_persona_3": round(pvec[2], 4),
                "prob_desconocido": round(pvec[3], 4),
                "decision_final": dec.estado_puerta,
                "categoria": cat,
                "motivo": motivo,
                "tiempo_ms": round(lat_ms, 2),
                "_recorte": recorte_usado,
            }

        item_ref = evaluar_intento(pipe_ref, "referencia_cnn_alineado_084")
        item_nuevo = evaluar_intento(pipe_nuevo, "nuevo_cnn_portal_completo_061")
        res_ref.append(item_ref)
        res_nuevo.append(item_nuevo)

        for it in (item_ref, item_nuevo):
            filas_csv.append({k: v for k, v in it.items() if k != "_recorte"})

        ab_ref = item_ref["decision_final"] == "ABIERTA"
        ab_nuevo = item_nuevo["decision_final"] == "ABIERTA"
        pareados.append({
            "identificador": r["identificador"],
            "ambos_rechazan": (not ab_ref) and (not ab_nuevo),
            "solo_referencia_acepta": ab_ref and (not ab_nuevo),
            "solo_nuevo_acepta": (not ab_ref) and ab_nuevo,
            "ambos_aceptan": ab_ref and ab_nuevo,
            "ref_cat": item_ref["categoria"],
            "ref_pred": item_ref["clase_argmax"],
            "ref_score": item_ref["score_argmax"],
            "nuevo_cat": item_nuevo["categoria"],
            "nuevo_pred": item_nuevo["clase_argmax"],
            "nuevo_score": item_nuevo["score_argmax"],
        })

    # Exportar predicciones CSV y JSON
    ruta_csv = dir_salida / "predicciones_prueba_reservada.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
        writer.writeheader()
        writer.writerows(filas_csv)

    (dir_salida / "predicciones_prueba_reservada.json").write_text(
        json.dumps(filas_csv, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 4. Métricas comparativas
    def resumir_modelo(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        fa = sum(1 for x in items if x["categoria"] == "falsa_aceptacion")
        desc = sum(1 for x in items if x["categoria"] == "clasificacion_desconocido")
        umb = sum(1 for x in items if x["categoria"] == "rechazo_umbral")
        det_f = sum(1 for x in items if x["categoria"] == "fallo_deteccion")
        err_t = sum(1 for x in items if x["categoria"] == "error_tecnico")
        assert fa + desc + umb + det_f + err_t == n

        clasificados = fa + desc + umb
        far_e2e_pct = round(100.0 * fa / n, 2) if n > 0 else None
        far_cls_pct = round(100.0 * fa / clasificados, 2) if clasificados > 0 else None
        latencias = [x["tiempo_ms"] for x in items]

        return {
            "intentos_totales": n,
            "clasificados_pipeline": clasificados,
            "falsas_aceptaciones": fa,
            "far_e2e_conteo": f"{fa}/{n}",
            "far_e2e_pct": far_e2e_pct if far_e2e_pct is not None else "no calculable",
            "far_clasificador_conteo": f"{fa}/{clasificados}",
            "far_clasificador_pct": far_cls_pct if far_cls_pct is not None else "no calculable",
            "clasificaciones_desconocido": desc,
            "rechazos_umbral": umb,
            "fallos_deteccion": det_f,
            "errores_tecnicos": err_t,
            "latencia_media_ms": round(float(np.mean(latencias)), 2),
        }

    met_ref = resumir_modelo(res_ref)
    met_nuevo = resumir_modelo(res_nuevo)

    tabla_pareada = {
        "ambos_rechazan": sum(1 for p in pareados if p["ambos_rechazan"]),
        "solo_anterior_acepta": sum(1 for p in pareados if p["solo_referencia_acepta"]),
        "solo_nuevo_acepta": sum(1 for p in pareados if p["solo_nuevo_acepta"]),
        "ambos_aceptan": sum(1 for p in pareados if p["ambos_aceptan"]),
    }

    # 5. Lámina local de errores (falsas aceptaciones o fallos de detección)
    incidencias = []
    for ir, inu in zip(res_ref, res_nuevo):
        if ir["categoria"] in ("falsa_aceptacion", "fallo_deteccion", "error_tecnico") or inu["categoria"] in ("falsa_aceptacion", "fallo_deteccion", "error_tecnico"):
            incidencias.append((ir, inu))

    ruta_lamina = None
    if incidencias:
        n_inc = len(incidencias)
        fig, axes = plt.subplots(n_inc, 3, figsize=(14, 3.8 * n_inc))
        if n_inc == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, (ir, inu) in enumerate(incidencias):
            frame_rgb = cv2.cvtColor(cv2.imread(str(RAIZ / ir["ruta"])), cv2.COLOR_BGR2RGB)
            if ir["recuadro"]:
                x, y, w_b, h_b = ir["recuadro"]
                cv2.rectangle(frame_rgb, (x, y), (x + w_b, y + h_b), (0, 255, 0), 3)
            axes[idx, 0].imshow(frame_rgb)
            axes[idx, 0].set_title(f"{ir['identificador']}\n({ir['estado_deteccion']})", fontsize=10, fontweight="bold")
            axes[idx, 0].axis("off")

            for col_i, it, nom in [(1, ir, "Ref (0.84)"), (2, inu, "Nuevo (0.61)")]:
                rec = it["_recorte"]
                if rec is not None and rec.size > 0:
                    axes[idx, col_i].imshow(cv2.cvtColor(rec, cv2.COLOR_BGR2RGB) if len(rec.shape) == 3 else rec, cmap="gray" if len(rec.shape) == 2 else None)
                else:
                    axes[idx, col_i].text(0.5, 0.5, f"SIN RECORTE\n({it['estado_deteccion']})", ha="center", va="center", color="red")
                col_txt = "red" if it["decision_final"] == "ABIERTA" else "green"
                axes[idx, col_i].set_title(
                    f"{nom}: {it['decision_final']}\n({it['categoria']})\nPred: {it['clase_argmax']} ({it['score_argmax']:.3f})",
                    fontsize=9, color=col_txt
                )
                axes[idx, col_i].axis("off")

        plt.tight_layout()
        ruta_lamina = dir_laminas / "inspeccion_errores_reserva_universidad.png"
        plt.savefig(ruta_lamina, dpi=150, bbox_inches="tight")
        plt.close()

    resumen = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "particion_evaluada": "prueba_reservada",
        "total_identidades": len(filas_reserva),
        "referencia_cnn_alineado_084": met_ref,
        "nuevo_cnn_portal_completo_061": met_nuevo,
        "tabla_pareada": tabla_pareada,
        "detalle_pareado": pareados,
        "lamina_errores": str(ruta_lamina) if ruta_lamina else None,
    }
    (dir_salida / "resumen_evaluacion_final_universidad.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print(" RESULTADOS FINALES SOBRE 9 IDENTIDADES UNIVERSITARIAS RESERVADAS")
    print("=" * 75)
    print(f" Referencia (0.84): FAR_e2e={met_ref['far_e2e_conteo']} ({met_ref['far_e2e_pct']}%) | FAR_cls={met_ref['far_clasificador_conteo']} ({met_ref['far_clasificador_pct']}%) | Desc={met_ref['clasificaciones_desconocido']} | Umb={met_ref['rechazos_umbral']} | DetFallo={met_ref['fallos_deteccion']}")
    print(f" Nuevo      (0.61): FAR_e2e={met_nuevo['far_e2e_conteo']} ({met_nuevo['far_e2e_pct']}%) | FAR_cls={met_nuevo['far_clasificador_conteo']} ({met_nuevo['far_clasificador_pct']}%) | Desc={met_nuevo['clasificaciones_desconocido']} | Umb={met_nuevo['rechazos_umbral']} | DetFallo={met_nuevo['fallos_deteccion']}")
    print(f" Pareado: Ambos rechazan={tabla_pareada['ambos_rechazan']} | Solo anterior={tabla_pareada['solo_anterior_acepta']} | Solo nuevo={tabla_pareada['solo_nuevo_acepta']} | Ambos aceptan={tabla_pareada['ambos_aceptan']}")
    print(f" Incidencias graficadas: {len(incidencias)}")
    print("=" * 75)
    return resumen, dir_salida


if __name__ == "__main__":
    ejecutar_evaluacion_final_universidad()
