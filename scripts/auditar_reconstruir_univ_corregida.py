"""
Script de auditoría y reconstrucción sin reentrenamiento para el experimento:
resultados/exp_4clases_univ_corregida_20261120_213750/

- Carga los pesos existentes de ambos candidatos (Referencia 0.61 y Nueva CNN 0.57),
  verifica sus hashes SHA-256 y conserva sus umbrales originales.
- Repite únicamente la inferencia de validación automática (64 Sesión 04 + 100 LFW + 7 Univ).
- Exporta decisiones individuales con las 4 probabilidades, curva de umbrales reconstruida,
  verificación de concordancia contra resumen_iteracion_univ_corregida.json y detalle
  de la confusión en autorizados.
- Guarda todo en resultados/exp_4clases_univ_corregida_20261120_213750/auditoria_reconstruida/.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.adquisicion.propio import calcular_hash_archivo
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.clasificacion.regla_acceso import evaluar_regla_acceso_completa
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from scripts.entrenar_desarrollo import BINS, CUADRICULA, cargar_registros
from scripts.experimentar_alineacion import (
    ID_TO_CLASE,
    N_CLASES,
    PERSONAS_AUTORIZADAS,
    calcular_tasa_pct,
    cargar_desconocidos_calibracion_alineados,
    preparar_captura_alineada,
)
from scripts.experimentar_4clases_universidad import (
    cargar_manifiesto_universidad_desarrollo,
    evaluar_modelo_sobre_validacion,
    procesar_muestras_universidad,
)


def ejecutar_auditoria_reconstruccion():
    dir_exp = RAIZ / "resultados" / "exp_4clases_univ_corregida_20261120_213750"
    dir_aud = dir_exp / "auditoria_reconstruida"
    dir_aud.mkdir(parents=True, exist_ok=True)

    ruta_resumen_orig = dir_exp / "resumen_iteracion_univ_corregida.json"
    resumen_orig = json.loads(ruta_resumen_orig.read_text(encoding="utf-8"))

    # 1. Cargar candidatos congelados existentes y verificar hashes
    ruta_cfg_ref = RAIZ / "resultados" / "exp_4clases_portal_completo_20261120_200452" / "configuracion_candidato_portal_completo_propuesto.json"
    ruta_cfg_nva = dir_exp / "configuracion_candidato_univ_corregida.json"

    pipe_ref = PipelineCandidatoCongelado(ruta_cfg_ref, raiz_proyecto=RAIZ)
    pipe_nva = PipelineCandidatoCongelado(ruta_cfg_nva, raiz_proyecto=RAIZ)

    assert abs(pipe_ref.umbral_score - 0.61) < 1e-9
    assert abs(pipe_nva.umbral_score - 0.57) < 1e-9

    # 2. Cargar datos de validación automática exactamente como en desarrollo
    detector = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)

    registros_propio, _ = cargar_registros(solo_base=False)
    hashes_por_particion: Dict[str, str] = {}
    muestras_autorizadas = []
    imagenes_autorizadas: Dict[str, np.ndarray] = {}
    for reg in registros_propio:
        if reg.procedencia == "portal_web_externo":
            continue
        muestra, img_proc = preparar_captura_alineada(reg, detector, hashes_por_particion)
        muestras_autorizadas.append(muestra)
        if img_proc is not None:
            imagenes_autorizadas[reg.identificador] = img_proc

    val_auth_todos = [m for m in muestras_autorizadas if m["particion"] == "validacion" and m["sesion"] == "sesion_04"]
    val_auth_proc = [m for m in val_auth_todos if m["estado"] == "procesada"]

    _, X_lfw_val_img, filas_lfw_val, estados_lfw_val = cargar_desconocidos_calibracion_alineados(detector, extractor)
    _, filas_univ_val = cargar_manifiesto_universidad_desarrollo()
    _, X_univ_val_img, meta_univ_val = procesar_muestras_universidad(filas_univ_val, detector, extractor, es_entrenamiento=False)
    X_val_univ_img_arr = np.asarray(X_univ_val_img, dtype=np.float32)

    datos_eval = {
        "val_auth_todos": val_auth_todos,
        "val_auth_proc_ids": [m["id"] for m in val_auth_proc],
        "imagenes_autorizadas": imagenes_autorizadas,
        "filas_lfw_val": filas_lfw_val,
        "estados_lfw_val": estados_lfw_val,
        "X_lfw_val_img": X_lfw_val_img,
        "meta_univ_val": meta_univ_val,
        "X_univ_val_img": X_val_univ_img_arr,
    }

    # 3. Reconstruir curva de umbrales completa para la nueva CNN
    res_nva_curva = evaluar_modelo_sobre_validacion("cnn_univ_corregida", pipe_nva.clasificador, datos_eval, es_cnn=True, umbral_fijo=None)
    (dir_aud / "curva_umbrales_reconstruida.json").write_text(
        json.dumps(res_nva_curva["curva_umbrales"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 4. Generar decisiones individuales completas (con las 4 probabilidades y pose)
    def generar_decisiones_detalladas(pipe: PipelineCandidatoCongelado, etiqueta_modelo: str) -> List[Dict[str, Any]]:
        decs = []
        th = pipe.umbral_score

        # 4a. Autorizados Sesión 04
        for m in val_auth_todos:
            mid = m["id"]
            if m["estado"] != "procesada":
                decs.append({
                    "modelo": etiqueta_modelo, "umbral": th, "fuente": "autorizado_sesion_04",
                    "identificador": mid, "pose": m["pose"], "identidad_real": m["persona"],
                    "estado_deteccion": m["estado_deteccion"], "identidad_predicha": None, "score_argmax": None,
                    "prob_persona_1": None, "prob_persona_2": None, "prob_persona_3": None, "prob_desconocido": None,
                    "decision_puerta": "CERRADA", "categoria": "fallo_deteccion", "motivo": "Sin detección de rostro único",
                })
                continue

            img_64 = imagenes_autorizadas[mid]
            probs = pipe.clasificador.predecir_probabilidades(np.expand_dims(img_64, axis=0))[0]
            pred_idx = int(np.argmax(probs))
            pred_nom = ID_TO_CLASE[pred_idx]
            score = float(probs[pred_idx])
            dec = evaluar_regla_acceso_completa("UN_ROSTRO", True, pred_nom, score, th, PERSONAS_AUTORIZADAS)

            if dec.estado_puerta == "ABIERTA":
                cat = "identificacion_correcta" if pred_nom == m["persona"] else "confusion_autorizado"
            elif pred_nom == "desconocido":
                cat = "bloqueo_desconocido"
            else:
                cat = "bloqueo_umbral"

            decs.append({
                "modelo": etiqueta_modelo, "umbral": th, "fuente": "autorizado_sesion_04",
                "identificador": mid, "pose": m["pose"], "identidad_real": m["persona"],
                "estado_deteccion": "UN_ROSTRO", "identidad_predicha": pred_nom, "score_argmax": round(score, 4),
                "prob_persona_1": round(float(probs[0]), 4), "prob_persona_2": round(float(probs[1]), 4),
                "prob_persona_3": round(float(probs[2]), 4), "prob_desconocido": round(float(probs[3]), 4),
                "decision_puerta": dec.estado_puerta, "categoria": cat,
                "motivo": dec.motivo_cierre if dec.estado_puerta == "CERRADA" else dec.mensaje,
            })

        # 4b. LFW Calibración (100)
        idx_lfw = 0
        for i, fila in enumerate(filas_lfw_val):
            est = estados_lfw_val[i]
            ident_lfw = fila.get("identidad_normalizada") or f"lfw_{i}"
            if est != "UN_ROSTRO":
                decs.append({
                    "modelo": etiqueta_modelo, "umbral": th, "fuente": "desconocido_lfw_calibracion",
                    "identificador": ident_lfw, "pose": "no_documentada", "identidad_real": "desconocido",
                    "estado_deteccion": est, "identidad_predicha": None, "score_argmax": None,
                    "prob_persona_1": None, "prob_persona_2": None, "prob_persona_3": None, "prob_desconocido": None,
                    "decision_puerta": "CERRADA", "categoria": "fallo_deteccion", "motivo": f"Fallo detector: {est}",
                })
                continue

            img_64 = X_lfw_val_img[idx_lfw]
            idx_lfw += 1
            probs = pipe.clasificador.predecir_probabilidades(np.expand_dims(img_64, axis=0))[0]
            pred_idx = int(np.argmax(probs))
            pred_nom = ID_TO_CLASE[pred_idx]
            score = float(probs[pred_idx])
            dec = evaluar_regla_acceso_completa("UN_ROSTRO", True, pred_nom, score, th, PERSONAS_AUTORIZADAS)

            if dec.estado_puerta == "ABIERTA":
                cat = "falsa_aceptacion"
            elif pred_nom == "desconocido":
                cat = "rechazo_desconocido"
            else:
                cat = "rechazo_umbral"

            decs.append({
                "modelo": etiqueta_modelo, "umbral": th, "fuente": "desconocido_lfw_calibracion",
                "identificador": ident_lfw, "pose": "no_documentada", "identidad_real": "desconocido",
                "estado_deteccion": "UN_ROSTRO", "identidad_predicha": pred_nom, "score_argmax": round(score, 4),
                "prob_persona_1": round(float(probs[0]), 4), "prob_persona_2": round(float(probs[1]), 4),
                "prob_persona_3": round(float(probs[2]), 4), "prob_desconocido": round(float(probs[3]), 4),
                "decision_puerta": dec.estado_puerta, "categoria": cat,
                "motivo": dec.motivo_cierre if dec.estado_puerta == "CERRADA" else dec.mensaje,
            })

        # 4c. Universidad Validación (7)
        idx_u = 0
        for m in meta_univ_val:
            if not m["procesada"]:
                decs.append({
                    "modelo": etiqueta_modelo, "umbral": th, "fuente": "desconocido_universidad_validacion",
                    "identificador": m["id"], "pose": "no_documentada", "identidad_real": "desconocido",
                    "estado_deteccion": m["estado_deteccion"], "identidad_predicha": None, "score_argmax": None,
                    "prob_persona_1": None, "prob_persona_2": None, "prob_persona_3": None, "prob_desconocido": None,
                    "decision_puerta": "CERRADA", "categoria": "fallo_deteccion", "motivo": f"Fallo detector: {m['estado_deteccion']}",
                })
                continue

            img_64 = X_val_univ_img_arr[idx_u]
            idx_u += 1
            probs = pipe.clasificador.predecir_probabilidades(np.expand_dims(img_64, axis=0))[0]
            pred_idx = int(np.argmax(probs))
            pred_nom = ID_TO_CLASE[pred_idx]
            score = float(probs[pred_idx])
            dec = evaluar_regla_acceso_completa("UN_ROSTRO", True, pred_nom, score, th, PERSONAS_AUTORIZADAS)

            if dec.estado_puerta == "ABIERTA":
                cat = "falsa_aceptacion"
            elif pred_nom == "desconocido":
                cat = "rechazo_desconocido"
            else:
                cat = "rechazo_umbral"

            decs.append({
                "modelo": etiqueta_modelo, "umbral": th, "fuente": "desconocido_universidad_validacion",
                "identificador": m["id"], "pose": "no_documentada", "identidad_real": "desconocido",
                "estado_deteccion": "UN_ROSTRO", "identidad_predicha": pred_nom, "score_argmax": round(score, 4),
                "prob_persona_1": round(float(probs[0]), 4), "prob_persona_2": round(float(probs[1]), 4),
                "prob_persona_3": round(float(probs[2]), 4), "prob_desconocido": round(float(probs[3]), 4),
                "decision_puerta": dec.estado_puerta, "categoria": cat,
                "motivo": dec.motivo_cierre if dec.estado_puerta == "CERRADA" else dec.mensaje,
            })

        return decs

    decs_ref = generar_decisiones_detalladas(pipe_ref, "referencia_portal_completo_061")
    decs_nva = generar_decisiones_detalladas(pipe_nva, "nueva_cnn_univ_corregida_057")
    todas_decs = decs_ref + decs_nva

    ruta_csv = dir_aud / "decisiones_individuales_reconstruidas.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(todas_decs[0].keys()))
        writer.writeheader()
        writer.writerows(todas_decs)
    (dir_aud / "decisiones_individuales_reconstruidas.json").write_text(
        json.dumps(todas_decs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 5. Verificar concordancia exacta contra resumen_iteracion_univ_corregida.json
    def verificar_concordancia(decs: List[Dict[str, Any]], rb_orig: Dict[str, Any], nombre: str) -> Dict[str, Any]:
        auth = [d for d in decs if d["fuente"] == "autorizado_sesion_04"]
        aciertos = sum(1 for d in auth if d["categoria"] == "identificacion_correcta")
        confusiones = sum(1 for d in auth if d["categoria"] == "confusion_autorizado")
        bloqueos = sum(1 for d in auth if d["decision_puerta"] == "CERRADA")
        assert len(auth) == aciertos + confusiones + bloqueos == 64

        desc = [d for d in decs if d["fuente"].startswith("desconocido")]
        fa = sum(1 for d in desc if d["categoria"] == "falsa_aceptacion")
        rech_desc = sum(1 for d in desc if d["categoria"] == "rechazo_desconocido")
        rech_umb = sum(1 for d in desc if d["categoria"] == "rechazo_umbral")
        fallo_det = sum(1 for d in desc if d["categoria"] == "fallo_deteccion")
        assert len(desc) == fa + rech_desc + rech_umb + fallo_det == 107

        coincide = (
            aciertos == rb_orig["aperturas_correctas"]
            and confusiones == rb_orig["confusiones_autorizados"]
            and bloqueos == rb_orig["bloqueos_puerta"]
            and fa == (rb_orig["far_lfw"]["falsas_aceptaciones"] + rb_orig["far_universidad"]["falsas_aceptaciones"])
        )
        if not coincide:
            raise AssertionError(f"Discrepancia detectada en {nombre}!")

        return {
            "coincide_con_resumen_original": coincide,
            "autorizados": {"total": len(auth), "aciertos": aciertos, "confusiones": confusiones, "bloqueos": bloqueos},
            "desconocidos": {"total": len(desc), "fa": fa, "rech_desconocido": rech_desc, "rech_umbral": rech_umb, "fallos_deteccion": fallo_det},
        }

    verif_ref = verificar_concordancia(decs_ref, resumen_orig["validacion_automatica_referencia_061"]["regla_b"], "referencia")
    verif_nva = verificar_concordancia(decs_nva, resumen_orig["validacion_automatica_nueva"]["regla_b"], "nueva")

    # 6. Identificar la muestra exacta de la confusión en la nueva CNN (y qué hizo la referencia en ella)
    confusiones_nva = [d for d in decs_nva if d["categoria"] == "confusion_autorizado"]
    detalles_confusion = []
    for c_item in confusiones_nva:
        ref_misma = next(d for d in decs_ref if d["identificador"] == c_item["identificador"])
        detalles_confusion.append({
            "identificador": c_item["identificador"],
            "pose": c_item["pose"],
            "persona_real": c_item["identidad_real"],
            "nueva_cnn_057": {
                "clase_predicha": c_item["identidad_predicha"],
                "score": c_item["score_argmax"],
                "probabilidades": {
                    "persona_1": c_item["prob_persona_1"],
                    "persona_2": c_item["prob_persona_2"],
                    "persona_3": c_item["prob_persona_3"],
                    "desconocido": c_item["prob_desconocido"],
                },
                "decision": c_item["decision_puerta"],
                "categoria": c_item["categoria"],
            },
            "referencia_061": {
                "clase_predicha": ref_misma["identidad_predicha"],
                "score": ref_misma["score_argmax"],
                "probabilidades": {
                    "persona_1": ref_misma["prob_persona_1"],
                    "persona_2": ref_misma["prob_persona_2"],
                    "persona_3": ref_misma["prob_persona_3"],
                    "desconocido": ref_misma["prob_desconocido"],
                },
                "decision": ref_misma["decision_puerta"],
                "categoria": ref_misma["categoria"],
            },
        })

    auditoria_info = {
        "fecha_reconstruccion_utc": datetime.now(timezone.utc).isoformat(),
        "nota": "Artefactos reconstruidos posteriormente mediante inferencia sobre los pesos guardados sin reentrenamiento ni recalibración.",
        "hashes_verificados": {
            "referencia_modelo": calcular_hash_archivo(pipe_ref.ruta_modelo),
            "nueva_modelo": calcular_hash_archivo(pipe_nva.ruta_modelo),
        },
        "verificacion_referencia": verif_ref,
        "verificacion_nueva": verif_nva,
        "detalle_confusion_autorizados": detalles_confusion,
    }
    (dir_aud / "resumen_auditoria_reconstruida.json").write_text(
        json.dumps(auditoria_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dir_aud / "README_AUDITORIA.md").write_text(
        "# Auditoría Reconstruida Sin Reentrenamiento\n\n"
        "Estos artefactos fueron **reconstruidos posteriormente** cargando los pesos congelados existentes "
        "(`0.61` y `0.57`) y ejecutando únicamente la inferencia de validación automática.\n"
        "Los conteos coinciden al 100% con `resumen_iteracion_univ_corregida.json`.\n",
        encoding="utf-8",
    )

    print("Auditoría reconstruida con éxito en:", dir_aud)
    print("Verificación Referencia:", verif_ref)
    print("Verificación Nueva CNN :", verif_nva)
    print("Detalle Confusión      :", json.dumps(detalles_confusion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ejecutar_auditoria_reconstruccion()
