"""
Diagnóstico cuantitativo y visual del bajo rendimiento de persona_1 en desarrollo.
Analiza sistemáticamente:
1. Desglose detallado por modelo, persona y pose (identificación previa vs rechazo).
2. Propiedades de imagen (brillo, contraste, nitidez, tamaño y posición de rostro).
3. Comparación asistida: recorte automático YuNet vs recorte humano aprobado.
4. Vecinos de entrenamiento más cercanos en espacio de características.
5. Viabilidad de umbrales con restricción por persona (FRR_p <= 30%).
6. Generación de láminas visuales locales para inspección.

Restricciones estrictas:
- Únicamente sesiones 01-04 y desconocidos de calibración LFW (100 identidades).
- No accede a sesión 05 ni a reserva de prueba.
- No modifica modelos, capturas ni manifiestos existentes.
- Guarda todos los resultados en un directorio independiente nuevo.
"""
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.entrenar_desarrollo import (
    cargar_registros, preparar_captura, matriz_datos,
    normalizar_resolucion, normalizar_intensidades,
    suavizado_gaussiano, a_escala_de_grises,
    RESOLUCION, SIGMA, BINS, CUADRICULA
)
from scripts.calibrar_desarrollo import CALIBRACION_LFW, preparar_negativos
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.rechazo_distancia import BancoReferencias
from reconocimiento.preprocesamiento.fabrica import crear_detector
from scripts.experimentar_rechazo_distancia import MODELOS_MAP, calcular_tasa_pct, formatear_tasa

RESULTADOS = RAIZ / "resultados"
BASE_YUNET = RESULTADOS / "desarrollo_yunet_20261118_185200_833277"
BASE_RECHAZO = RESULTADOS / "desarrollo_rechazo_distancia_20261118_195533_620169"


def estadisticas_vector(arr: List[float]) -> Dict[str, Optional[float]]:
    if not arr:
        return {"media": None, "std": None, "min": None, "mediana": None, "max": None}
    a = np.array(arr, dtype=np.float64)
    return {
        "media": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "mediana": float(np.median(a)),
        "max": float(np.max(a)),
    }


def calcular_metricas_imagen(img_bgr: np.ndarray, caja: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brillo = float(np.mean(gray))
    contraste = float(np.std(gray))
    nitidez = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    h_img, w_img = img_bgr.shape[:2]
    caja_ancho = float(caja[2]) if caja else 0.0
    caja_alto = float(caja[3]) if caja else 0.0
    area_caja = caja_ancho * caja_alto
    area_img = float(w_img * h_img)
    pct_area = (area_caja / area_img * 100.0) if area_img > 0 else 0.0
    aspect_ratio = (caja_ancho / caja_alto) if caja_alto > 0 else 0.0

    return {
        "brillo": brillo,
        "contraste": contraste,
        "nitidez": nitidez,
        "caja_ancho": caja_ancho,
        "caja_alto": caja_alto,
        "area_caja": area_caja,
        "pct_area_imagen": pct_area,
        "aspect_ratio": aspect_ratio,
    }


def ejecutar_diagnostico():
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    dir_salida = RESULTADOS / f"diagnostico_persona_1_{timestamp}"
    dir_laminas = dir_salida / "laminas"
    dir_laminas.mkdir(parents=True, exist_ok=True)
    print(f"Iniciando diagnóstico en: {dir_salida}")

    detector = crear_detector()
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS, rango_intensidad=(0.0, 1.0))
    registros, _ = cargar_registros()

    # Separar particiones y registros
    train_regs = [r for r in registros if r.particion == "entrenamiento"]
    val_regs = [r for r in registros if r.particion == "validacion"]

    muestras = []
    imagenes = {}
    hashes_part = {}
    for r in registros:
        m, img = preparar_captura(r, detector, hashes_part)
        muestras.append(m)
        if img is not None:
            imagenes[r.identificador] = img

    entrenamiento = [m for m in muestras if m["particion"] == "entrenamiento"]
    intentos_val = [m for m in muestras if m["particion"] == "validacion" and m["intento"]]
    personas = sorted({m["persona"] for m in entrenamiento if m["estado"] == "procesada"})
    p2id = {p: i for i, p in enumerate(personas)}

    img_train, y_train, ids_train = matriz_datos("entrenamiento", muestras, imagenes, p2id)
    img_val, y_val, ids_val = matriz_datos("validacion", muestras, imagenes, p2id)
    hist_train, _ = extractor.extraer_lote(img_train)
    hist_val, _ = extractor.extraer_lote(img_val)

    # 1. Cargar bancos y calibraciones actuales de los 5 modelos
    calibs_base = {}
    modelos = {}
    bancos = {}
    for nom, (cls, arch) in MODELOS_MAP.items():
        dir_mod_rech = BASE_RECHAZO / nom
        calib_file = dir_mod_rech / "calibracion_rechazo.json"
        calibs_base[nom] = json.loads(calib_file.read_text(encoding="utf-8"))
        modelos[nom] = cls.cargar(BASE_YUNET / nom / arch)
        bancos[nom] = BancoReferencias.cargar_npz(dir_mod_rech / "referencias.npz")

    # 2. Análisis cuantitativo por modelo, persona y pose
    analisis_modelos = {}
    preds_detalle_por_modelo = {}
    for nom in MODELOS_MAP.keys():
        preds_file = BASE_RECHAZO / nom / "predicciones_detalle.json"
        preds_detalle_por_modelo[nom] = json.loads(preds_file.read_text(encoding="utf-8"))["validacion"]

    for nom in MODELOS_MAP.keys():
        preds = preds_detalle_por_modelo[nom]
        calib = calibs_base[nom]["regla_B_puntaje_y_distancia"]
        s_th = calib["umbral_score"]
        d_th = calib["umbral_distancia"]
        met = calib["metrica_distancia"]

        por_persona_pose = {}
        for p in personas:
            por_persona_pose[p] = {}
            p_preds = [v for v in preds if v["persona_real"] == p]
            poses_p = sorted({v["pose"] for v in p_preds})
            for pose in poses_p:
                sub = [v for v in p_preds if v["pose"] == pose]
                tot = len(sub)
                corr_id_previa = sum(1 for v in sub if v["persona_predicha"] == p)
                confusiones = [v["persona_predicha"] for v in sub if v["persona_predicha"] != p and v["persona_predicha"] != "DESCONOCIDO"]
                conteo_conf = dict(Counter(confusiones))

                ap_corr = sum(1 for v in sub if v["regla_B"]["estado_puerta"] == "ABIERTA" and v["persona_predicha"] == p)
                ap_conf = sum(1 for v in sub if v["regla_B"]["estado_puerta"] == "ABIERTA" and v["persona_predicha"] != p)

                rech_umbral = sum(1 for v in sub if v["regla_B"]["motivo_cierre"] == "RECHAZO_UMBRAL")
                rech_dist = sum(1 for v in sub if v["regla_B"]["motivo_cierre"] == "RECHAZO_DISTANCIA")
                rech_ambos = sum(1 for v in sub if v["regla_B"]["motivo_cierre"] == "RECHAZO_AMBOS")
                fallos_det = sum(1 for v in sub if v["regla_B"]["motivo_cierre"].startswith("DETECCION_"))
                fallos_proc = sum(1 for v in sub if v["regla_B"]["motivo_cierre"].startswith("PROCESAMIENTO_"))

                scores_pose = [v["confianza"] for v in sub if v.get("confianza") is not None]
                dists_pose = [v["distancia"] for v in sub if v.get("distancia") is not None]

                por_persona_pose[p][pose] = {
                    "total_intentos": tot,
                    "identificacion_correcta_previa": corr_id_previa,
                    "identificacion_correcta_previa_pct": calcular_tasa_pct(corr_id_previa, tot),
                    "confusiones_previa": conteo_conf,
                    "aperturas_correctas": ap_corr,
                    "aperturas_confundidas": ap_conf,
                    "rechazos_por_umbral": rech_umbral,
                    "rechazos_por_distancia": rech_dist,
                    "rechazos_por_ambos": rech_ambos,
                    "fallos_deteccion": fallos_det,
                    "fallos_procesamiento": fallos_proc,
                    "total_rechazos": tot - (ap_corr + ap_conf),
                    "frr_puerta_pct": calcular_tasa_pct(tot - (ap_corr + ap_conf), tot),
                    "distribucion_scores": estadisticas_vector(scores_pose),
                    "distribucion_distancias": estadisticas_vector(dists_pose),
                }

        # Totales agregados por persona
        resumen_personas = {}
        for p in personas:
            sub = [v for v in preds if v["persona_real"] == p]
            tot = len(sub)
            corr_id = sum(1 for v in sub if v["persona_predicha"] == p)
            confs = dict(Counter(v["persona_predicha"] for v in sub if v["persona_predicha"] != p and v["persona_predicha"] != "DESCONOCIDO"))
            ap_corr = sum(1 for v in sub if v["regla_B"]["estado_puerta"] == "ABIERTA" and v["persona_predicha"] == p)
            ap_conf = sum(1 for v in sub if v["regla_B"]["estado_puerta"] == "ABIERTA" and v["persona_predicha"] != p)
            rech_tot = tot - (ap_corr + ap_conf)
            resumen_personas[p] = {
                "total": tot,
                "identificaciones_correctas_previas": corr_id,
                "identificaciones_correctas_previas_pct": calcular_tasa_pct(corr_id, tot),
                "confusiones_previas": confs,
                "aperturas_correctas": ap_corr,
                "aperturas_confundidas": ap_conf,
                "rechazos_totales": rech_tot,
                "frr_puerta_pct": calcular_tasa_pct(rech_tot, tot),
                "rechazos_umbral": sum(1 for v in sub if v["regla_B"]["motivo_cierre"] == "RECHAZO_UMBRAL"),
                "rechazos_distancia": sum(1 for v in sub if v["regla_B"]["motivo_cierre"] == "RECHAZO_DISTANCIA"),
                "rechazos_ambos": sum(1 for v in sub if v["regla_B"]["motivo_cierre"] == "RECHAZO_AMBOS"),
            }

        analisis_modelos[nom] = {
            "calibracion_activa": {"umbral_score": s_th, "umbral_distancia": d_th, "metrica": met},
            "por_persona": resumen_personas,
            "detalle_pose": por_persona_pose,
        }

    # 3. Propiedades estadísticas de imagen por sesión y persona
    stats_imagenes = {}
    datos_completos_imagenes = []
    for r in registros:
        img_bgr = cv2.imread(r.ruta)
        if img_bgr is None:
            continue
        caja = r.caja_aprobada
        metr = calcular_metricas_imagen(img_bgr, caja)
        metr.update({
            "id": r.identificador,
            "persona": r.persona,
            "sesion": r.sesion,
            "pose": r.pose,
            "particion": r.particion,
        })
        datos_completos_imagenes.append(metr)

    for p in personas:
        stats_imagenes[p] = {}
        sub_p = [d for d in datos_completos_imagenes if d["persona"] == p]
        sesiones_p = sorted({d["sesion"] for d in sub_p})
        for s in sesiones_p:
            sub_s = [d for d in sub_p if d["sesion"] == s]
            stats_imagenes[p][s] = {
                "n": len(sub_s),
                "brillo": estadisticas_vector([d["brillo"] for d in sub_s]),
                "contraste": estadisticas_vector([d["contraste"] for d in sub_s]),
                "nitidez": estadisticas_vector([d["nitidez"] for d in sub_s]),
                "caja_ancho": estadisticas_vector([d["caja_ancho"] for d in sub_s]),
                "caja_alto": estadisticas_vector([d["caja_alto"] for d in sub_s]),
                "area_caja": estadisticas_vector([d["area_caja"] for d in sub_s]),
                "pct_area_imagen": estadisticas_vector([d["pct_area_imagen"] for d in sub_s]),
            }

    # Comparación de cajas YuNet vs Humanas en sesión 04
    comparativa_cajas_val = []
    for r in val_regs:
        img_bgr = cv2.imread(r.ruta)
        det = detector.detectar(img_bgr)
        c_humana = r.caja_aprobada
        c_yunet = det.recuadro
        iou = None
        iomin = None
        if c_humana and c_yunet:
            iou, iomin = detector.calcular_iou_y_iomin(c_yunet, c_humana)
        comparativa_cajas_val.append({
            "id": r.identificador,
            "persona": r.persona,
            "pose": r.pose,
            "caja_humana": c_humana,
            "caja_yunet": c_yunet,
            "estado_yunet": det.estado.value,
            "iou": iou,
            "iomin": iomin,
            "ancho_humana": c_humana[2] if c_humana else None,
            "alto_humana": c_humana[3] if c_humana else None,
            "ancho_yunet": c_yunet[2] if c_yunet else None,
            "alto_yunet": c_yunet[3] if c_yunet else None,
        })

    # 4. Comparación diagnóstica sin reentrenar: Recorte YuNet vs Recorte Humano Aprobado
    # Evaluado sobre las 64 muestras de validación (sesión 04)
    print("\n--- Ejecutando comparación diagnóstica YuNet vs Humano ---")
    val_p1_regs = [r for r in val_regs if r.persona == "persona_1"]
    imgs_yunet = []
    imgs_humano = []
    for r in val_p1_regs:
        img_bgr = cv2.imread(r.ruta)
        det = detector.detectar(img_bgr)
        rec_y = det.rostro_recortado
        norm_y = suavizado_gaussiano(normalizar_intensidades(normalizar_resolucion(a_escala_de_grises(rec_y), RESOLUCION), "float32_0_1"), SIGMA)
        imgs_yunet.append(norm_y)

        rec_h = detector.recortar_con_margen_seguro(img_bgr, r.caja_aprobada)
        norm_h = suavizado_gaussiano(normalizar_intensidades(normalizar_resolucion(a_escala_de_grises(rec_h), RESOLUCION), "float32_0_1"), SIGMA)
        imgs_humano.append(norm_h)

    imgs_yunet_arr = np.stack(imgs_yunet)
    imgs_humano_arr = np.stack(imgs_humano)
    hists_yunet_arr, _ = extractor.extraer_lote(imgs_yunet_arr)
    hists_humano_arr, _ = extractor.extraer_lote(imgs_humano_arr)

    comparacion_recortes_p1 = {}
    for nom, m in modelos.items():
        if nom == "cnn":
            py = m.predecir_probabilidades(imgs_yunet_arr)
            ph = m.predecir_probabilidades(imgs_humano_arr)
            ry = m.extraer_caracteristicas(imgs_yunet_arr)
            rh = m.extraer_caracteristicas(imgs_humano_arr)
        elif nom == "svm":
            py = m.predecir_probabilidades(hists_yunet_arr)
            ph = m.predecir_probabilidades(hists_humano_arr)
            ry = m.escalador.transform(hists_yunet_arr).astype(np.float32)
            rh = m.escalador.transform(hists_humano_arr).astype(np.float32)
        else:
            py = m.predecir_probabilidades(hists_yunet_arr)
            ph = m.predecir_probabilidades(hists_humano_arr)
            ry = hists_yunet_arr.astype(np.float32)
            rh = hists_humano_arr.astype(np.float32)

        preds_y = [personas[int(np.argmax(p))] for p in py]
        scores_y = [float(np.max(p)) for p in py]
        preds_h = [personas[int(np.argmax(p))] for p in ph]
        scores_h = [float(np.max(p)) for p in ph]

        calib = calibs_base[nom]["regla_B_puntaje_y_distancia"]
        s_th = calib["umbral_score"]
        d_th = calib["umbral_distancia"]
        met = calib["metrica_distancia"]
        banco = bancos[nom]

        dists_y = [banco.calcular_distancia(ry[i], preds_y[i], metrica=met) for i in range(len(ry))]
        dists_h = [banco.calcular_distancia(rh[i], preds_h[i], metrica=met) for i in range(len(rh))]

        corr_y = sum(1 for p in preds_y if p == "persona_1")
        corr_h = sum(1 for p in preds_h if p == "persona_1")

        ap_y = sum(1 for i in range(len(preds_y)) if scores_y[i] >= s_th and dists_y[i] <= d_th)
        ap_h = sum(1 for i in range(len(preds_h)) if scores_h[i] >= s_th and dists_h[i] <= d_th)

        ap_corr_y = sum(1 for i in range(len(preds_y)) if scores_y[i] >= s_th and dists_y[i] <= d_th and preds_y[i] == "persona_1")
        ap_corr_h = sum(1 for i in range(len(preds_h)) if scores_h[i] >= s_th and dists_h[i] <= d_th and preds_h[i] == "persona_1")

        comparacion_recortes_p1[nom] = {
            "yunet": {
                "identificacion_correcta": corr_y,
                "identificacion_correcta_pct": calcular_tasa_pct(corr_y, 24),
                "aperturas_totales": ap_y,
                "aperturas_correctas": ap_corr_y,
                "rechazos": 24 - ap_y,
                "frr_puerta_pct": calcular_tasa_pct(24 - ap_y, 24),
                "score_medio": float(np.mean(scores_y)),
                "distancia_media": float(np.mean(dists_y)),
            },
            "humano_aprobado": {
                "identificacion_correcta": corr_h,
                "identificacion_correcta_pct": calcular_tasa_pct(corr_h, 24),
                "aperturas_totales": ap_h,
                "aperturas_correctas": ap_corr_h,
                "rechazos": 24 - ap_h,
                "frr_puerta_pct": calcular_tasa_pct(24 - ap_h, 24),
                "score_medio": float(np.mean(scores_h)),
                "distancia_media": float(np.mean(dists_h)),
            }
        }

    # 5. Vecinos de entrenamiento más cercanos para errores de persona_1
    print("\n--- Analizando vecinos más cercanos en entrenamiento ---")
    train_meta = [(r.identificador, r.persona, r.sesion, r.pose) for r in train_regs]
    vecinos_errores_p1 = []
    for i, r in enumerate(val_p1_regs):
        vh = hists_yunet_arr[i]
        dists = np.linalg.norm(hist_train - vh, axis=1)
        nn_idx = np.argsort(dists)[:5]
        lista_nn = [
            {
                "id": train_meta[idx][0],
                "persona": train_meta[idx][1],
                "sesion": train_meta[idx][2],
                "pose": train_meta[idx][3],
                "distancia_l2": float(dists[idx]),
            }
            for idx in nn_idx
        ]
        vecinos_errores_p1.append({
            "id_val": r.identificador,
            "pose": r.pose,
            "vecinos_cercanos_entrenamiento": lista_nn,
            "persona_vecino_1": lista_nn[0]["persona"],
            "distancia_vecino_1": lista_nn[0]["distancia_l2"],
        })

    conteo_nn1 = Counter(v["persona_vecino_1"] for v in vecinos_errores_p1)

    # 6. Viabilidad de umbrales por persona (FRR_p <= 30% para todas las personas)
    print("\n--- Analizando viabilidad de umbrales por persona ---")
    filas_neg, estados_neg, img_neg = preparar_negativos(detector)
    hist_neg, _ = extractor.extraer_lote(img_neg)
    idx_neg_unrostro = [i for i, est in enumerate(estados_neg) if est == "UN_ROSTRO"]

    candidatos_score = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00]
    metricas_eval = ["min_cos", "cent_cos", "min_l2", "cent_l2"]
    mapa_idx_val = {id_m: idx for idx, id_m in enumerate(ids_val)}

    viabilidad_por_persona = {}

    for nom, (cls, arch) in MODELOS_MAP.items():
        m = modelos[nom]
        banco = bancos[nom]
        calib_prev = calibs_base[nom]["regla_B_puntaje_y_distancia"]
        u_base = calib_prev["umbral_score"]
        cands_s = sorted(set(candidatos_score + [u_base]))

        if nom == "cnn":
            rep_v = m.extraer_caracteristicas(img_val)
            rep_n = m.extraer_caracteristicas(img_neg)
            probs_v = m.predecir_probabilidades(img_val)
            probs_n = m.predecir_probabilidades(img_neg)
        elif nom == "svm":
            rep_v = m.escalador.transform(hist_val).astype(np.float32)
            rep_n = m.escalador.transform(hist_neg).astype(np.float32)
            probs_v = m.predecir_probabilidades(hist_val)
            probs_n = m.predecir_probabilidades(hist_neg)
        else:
            rep_v = hist_val.astype(np.float32)
            rep_n = hist_neg.astype(np.float32)
            probs_v = m.predecir_probabilidades(hist_val)
            probs_n = m.predecir_probabilidades(hist_neg)

        pv = [personas[int(np.argmax(p))] for p in probs_v]
        sv = [float(np.max(p)) for p in probs_v]
        pn = [personas[int(np.argmax(p))] for p in probs_n]
        sn = [float(np.max(p)) for p in probs_n]

        candidatos_viables_puerta = []
        candidatos_viables_correcta = []

        for met in metricas_eval:
            dv = [banco.calcular_distancia(rep_v[i], pv[i], metrica=met) for i in range(len(rep_v))]
            dn = [banco.calcular_distancia(rep_n[i], pn[i], metrica=met) for i in range(len(rep_n))]
            d_cands = sorted(set(dv))

            for s_th in cands_s:
                for d_th in d_cands:
                    ap_puerta = {p: 0 for p in personas}
                    ap_corr = {p: 0 for p in personas}

                    for item in intentos_val:
                        mid = item["id"]
                        if mid in mapa_idx_val:
                            idx = mapa_idx_val[mid]
                            if sv[idx] >= s_th and dv[idx] <= d_th:
                                ap_puerta[item["persona"]] += 1
                                if pv[idx] == item["persona"]:
                                    ap_corr[item["persona"]] += 1

                    # Criterio A: Puerta abierta FRR <= 30% por persona (>=17 p1, >=14 p2, >=14 p3)
                    if ap_puerta["persona_1"] >= 17 and ap_puerta["persona_2"] >= 14 and ap_puerta["persona_3"] >= 14:
                        ap_neg = sum(sn[j] >= s_th and dn[j] <= d_th for j in range(len(idx_neg_unrostro)))
                        candidatos_viables_puerta.append({
                            "metrica": met,
                            "umbral_score": s_th,
                            "umbral_distancia": d_th,
                            "falsas_aceptaciones": ap_neg,
                            "far_pct": calcular_tasa_pct(ap_neg, 100),
                            "aperturas_por_persona": dict(ap_puerta),
                            "aperturas_correctas_por_persona": dict(ap_corr),
                        })

                    # Criterio B: Acceso correcto FRR_correcta <= 30% por persona
                    if ap_corr["persona_1"] >= 17 and ap_corr["persona_2"] >= 14 and ap_corr["persona_3"] >= 14:
                        ap_neg = sum(sn[j] >= s_th and dn[j] <= d_th for j in range(len(idx_neg_unrostro)))
                        candidatos_viables_correcta.append({
                            "metrica": met,
                            "umbral_score": s_th,
                            "umbral_distancia": d_th,
                            "falsas_aceptaciones": ap_neg,
                            "far_pct": calcular_tasa_pct(ap_neg, 100),
                            "aperturas_correctas_por_persona": dict(ap_corr),
                        })

        if candidatos_viables_puerta:
            candidatos_viables_puerta.sort(key=lambda x: (x["falsas_aceptaciones"], -x["umbral_score"], x["umbral_distancia"]))
            mejor_puerta = candidatos_viables_puerta[0]
        else:
            mejor_puerta = None

        if candidatos_viables_correcta:
            candidatos_viables_correcta.sort(key=lambda x: (x["falsas_aceptaciones"], -x["umbral_score"], x["umbral_distancia"]))
            mejor_correcta = candidatos_viables_correcta[0]
        else:
            mejor_correcta = None

        viabilidad_por_persona[nom] = {
            "viabilidad_frr_puerta": "viable" if mejor_puerta else "sin_configuracion_viable_por_persona",
            "mejor_configuracion_puerta": mejor_puerta,
            "total_candidatos_viables_puerta": len(candidatos_viables_puerta),
            "viabilidad_frr_acceso_correcto": "viable" if mejor_correcta else "sin_configuracion_viable_por_persona",
            "mejor_configuracion_acceso_correcto": mejor_correcta,
            "total_candidatos_viables_acceso_correcto": len(candidatos_viables_correcta),
        }

    # 7. Generación de Láminas Visuales
    print("\n--- Generando láminas visuales para inspección ---")

    # Lámina 1: Comparativa de Escala y Encuadre entre Sesiones (persona_1)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    ejemplos_ses = {
        "sesion_01 (Entr.)": "persona_1_sesion_01_frontal_01",
        "sesion_02 (Entr.)": "persona_1_sesion_02_frontal_01",
        "sesion_03 (Entr.)": "persona_1_sesion_03_frontal_01",
        "sesion_04 (Val.)": "persona_1_sesion_04_frontal_01",
    }
    reg_map = {r.identificador: r for r in registros}
    for idx, (titulo, mid) in enumerate(ejemplos_ses.items()):
        ax = axes[idx]
        if mid in reg_map:
            r = reg_map[mid]
            img = cv2.imread(r.ruta)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            c = r.caja_aprobada
            if c:
                x, y, w, h = c
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y - 5, f"{w}x{h} px", color="lime", fontsize=10, backgroundcolor="black")
            ax.imshow(img_rgb)
            ax.set_title(f"{titulo}\n{mid}", fontsize=10)
            ax.axis("off")
    plt.suptitle("Lámina 1: Variación Drástica de Escala en persona_1 (Sesiones 01-03 vs Sesión 04)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta_lamina1 = dir_laminas / "lamina_1_comparacion_escala_sesiones.png"
    plt.savefig(ruta_lamina1, dpi=150)
    plt.close()

    # Lámina 2: YuNet vs Humano en las 5 poses de sesión 04
    poses_lista = ["frontal", "izquierda", "derecha", "arriba", "abajo"]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for col_idx, pose in enumerate(poses_lista):
        r_p = next(r for r in val_p1_regs if r.pose == pose)
        img_bgr = cv2.imread(r_p.ruta)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        det = detector.detectar(img_bgr)
        cy = det.recuadro
        ch = r_p.caja_aprobada

        # Fila 0: Imagen con cajas
        ax_img = axes[0, col_idx]
        ax_img.imshow(img_rgb)
        if cy:
            ry = plt.Rectangle((cy[0], cy[1]), cy[2], cy[3], fill=False, edgecolor="lime", linewidth=2, label="YuNet")
            ax_img.add_patch(ry)
        if ch:
            rh = plt.Rectangle((ch[0], ch[1]), ch[2], ch[3], fill=False, edgecolor="cyan", linewidth=2, linestyle="--", label="Humano")
            ax_img.add_patch(rh)
        iou_val = detector.calcular_iou_y_iomin(cy, ch)[0] if (cy and ch) else 0.0
        ax_img.set_title(f"Pose: {pose}\nIoU: {iou_val:.2f}", fontsize=10)
        ax_img.axis("off")
        if col_idx == 0:
            ax_img.legend(loc="upper left")

        # Fila 1: Comparativa de recortes normalizados
        ax_rec = axes[1, col_idx]
        rec_y = det.rostro_recortado
        rec_h = detector.recortar_con_margen_seguro(img_bgr, ch) if ch else None
        norm_y = normalizar_resolucion(a_escala_de_grises(rec_y), (64, 64))
        norm_h = normalizar_resolucion(a_escala_de_grises(rec_h), (64, 64)) if rec_h is not None else np.zeros((64, 64))

        comp = np.hstack([norm_y, np.full((64, 4), 255, dtype=np.uint8), norm_h])
        ax_rec.imshow(comp, cmap="gray")
        ax_rec.set_title("Izquierda: YuNet | Derecha: Humano", fontsize=8)
        ax_rec.axis("off")

    plt.suptitle("Lámina 2: Recuadro y Recorte de YuNet (Verde) vs Humano Aprobado (Cian) en Sesión 04", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta_lamina2 = dir_laminas / "lamina_2_recortes_yunet_vs_humano.png"
    plt.savefig(ruta_lamina2, dpi=150)
    plt.close()

    # Lámina 3: Vecinos de entrenamiento más cercanos para muestras de sesión 04
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    muestras_interes = [
        val_p1_regs[0],  # frontal 01
        val_p1_regs[4],  # izquierda 01
        val_p1_regs[9],  # derecha 01
        val_p1_regs[14], # arriba 01
    ]
    for row_idx, r_val in enumerate(muestras_interes):
        # Col 0: muestra de validación
        ax_val = axes[row_idx, 0]
        img_val_bgr = cv2.imread(r_val.ruta)
        det_v = detector.detectar(img_val_bgr)
        rec_v = normalizar_resolucion(a_escala_de_grises(det_v.rostro_recortado), (64, 64))
        ax_val.imshow(rec_v, cmap="gray")
        ax_val.set_title(f"Val: {r_val.identificador}\n({r_val.persona}, {r_val.pose})", fontsize=9, color="blue")
        ax_val.axis("off")

        # Buscar 2 vecinos más cercanos
        norm_v = suavizado_gaussiano(normalizar_intensidades(rec_v, "float32_0_1"), SIGMA)
        hist_v = extractor.extraer(norm_v).histograma if hasattr(extractor.extraer(norm_v), 'histograma') else extractor.extraer_lote(norm_v[np.newaxis, ...])[0][0]
        dists = np.linalg.norm(hist_train - hist_v, axis=1)
        nn_idx = np.argsort(dists)[:2]

        for col_idx, idx_nn in enumerate(nn_idx, start=1):
            ax_nn = axes[row_idx, col_idx]
            r_tr = train_regs[idx_nn]
            img_tr_bgr = cv2.imread(r_tr.ruta)
            rec_tr = detector.recortar_con_margen_seguro(img_tr_bgr, r_tr.caja_aprobada)
            rec_tr_norm = normalizar_resolucion(a_escala_de_grises(rec_tr), (64, 64))
            ax_nn.imshow(rec_tr_norm, cmap="gray")
            color_nn = "green" if r_tr.persona == "persona_1" else "red"
            ax_nn.set_title(f"Vecino {col_idx}: {r_tr.persona}\n({r_tr.sesion} {r_tr.pose}, d={dists[idx_nn]:.2f})", fontsize=9, color=color_nn)
            ax_nn.axis("off")

    plt.suptitle("Lámina 3: Vecinos más Cercanos en Histogramas Regionales para persona_1 (Sesión 04)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta_lamina3 = dir_laminas / "lamina_3_vecinos_cercanos_entrenamiento.png"
    plt.savefig(ruta_lamina3, dpi=150)
    plt.close()

    # Lámina 4: Comparativa cuantitativa de métricas de imagen
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    # Ancho de caja
    sesiones_labels = ["sesion_01", "sesion_02", "sesion_03", "sesion_04"]
    for p in personas:
        medias_ancho = [stats_imagenes[p][s]["caja_ancho"]["media"] for s in sesiones_labels]
        ax1.plot(sesiones_labels, medias_ancho, marker="o", label=p)
    ax1.set_title("Ancho Promedio de Caja Facial (px)")
    ax1.set_ylabel("Píxeles")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Nitidez
    for p in personas:
        medias_nit = [stats_imagenes[p][s]["nitidez"]["media"] for s in sesiones_labels]
        ax2.plot(sesiones_labels, medias_nit, marker="s", label=p)
    ax2.set_title("Nitidez Promedio (Varianza Laplaciana)")
    ax2.set_ylabel("Varianza Laplaciana")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # Brillo
    for p in personas:
        medias_bri = [stats_imagenes[p][s]["brillo"]["media"] for s in sesiones_labels]
        ax3.plot(sesiones_labels, medias_bri, marker="^", label=p)
    ax3.set_title("Brillo Promedio (Intensidad en Escala de Grises)")
    ax3.set_ylabel("Nivel de Gris (0-255)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # Contraste
    for p in personas:
        medias_con = [stats_imagenes[p][s]["contraste"]["media"] for s in sesiones_labels]
        ax4.plot(sesiones_labels, medias_con, marker="d", label=p)
    ax4.set_title("Contraste Promedio (Desviación Estándar de Gris)")
    ax4.set_ylabel("Desviación Estándar")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    plt.suptitle("Lámina 4: Evolución de Propiedades de Imagen por Persona y Sesión", fontsize=13, fontweight="bold")
    plt.tight_layout()
    ruta_lamina4 = dir_laminas / "lamina_4_distribucion_metricas_imagen.png"
    plt.savefig(ruta_lamina4, dpi=150)
    plt.close()

    # 8. Guardar diagnóstico en JSON
    informe_json = {
        "timestamp": timestamp,
        "directorios": {
            "salida": str(dir_salida),
            "desarrollo_base_yunet": str(BASE_YUNET),
            "desarrollo_base_rechazo": str(BASE_RECHAZO),
            "laminas": {
                "lamina_1": str(ruta_lamina1),
                "lamina_2": str(ruta_lamina2),
                "lamina_3": str(ruta_lamina3),
                "lamina_4": str(ruta_lamina4),
            }
        },
        "conteo_ejemplos": {
            p: {
                "entrenamiento": len([r for r in train_regs if r.persona == p]),
                "sesiones_entrenamiento": dict(Counter(r.sesion for r in train_regs if r.persona == p)),
                "validacion_sesion_04": len([r for r in val_regs if r.persona == p]),
                "poses_validacion": dict(Counter(r.pose for r in val_regs if r.persona == p)),
            }
            for p in personas
        },
        "estadisticas_imagenes": stats_imagenes,
        "analisis_modelos": analisis_modelos,
        "comparacion_recortes_asistida_p1": comparacion_recortes_p1,
        "vecinos_cercanos_p1_resumen": dict(conteo_nn1),
        "viabilidad_por_persona": viabilidad_por_persona,
    }

    ruta_json = dir_salida / "diagnostico.json"
    ruta_json.write_text(json.dumps(informe_json, indent=2), encoding="utf-8")
    print(f"Diagnóstico guardado en: {ruta_json}")

    return informe_json, dir_salida


if __name__ == "__main__":
    ejecutar_diagnostico()
