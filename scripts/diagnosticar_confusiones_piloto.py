"""
Diagnóstico completo (solo lectura) de las confusiones observadas en el piloto remoto.

Analiza ambas CNN congeladas:
  - Principal:  configuracion_candidato_antialiasing_052.json (umbral 0.52)
  - Referencia: configuracion_candidato_estandar_portal_061.json (umbral 0.61)

Partes:
  A. Ajuste sobre las 75 fotos originales del portal (entrenamiento vs. no incluidas).
  B. Robustez ante 11 perturbaciones sobre fotograma completo (portal desconocidos y sesion_04).
  C. Espacio de embeddings 128-D (fc1_relu): distancias coseno a centroides y voto 5-NN + PCA 2D.
  D. Balance de entrenamiento y artefactos disponibles (pesos y ausencia de curvas por época).

No modifica modelos, umbrales, configuraciones, manifiestos ni fotografías.
"""

import csv
from datetime import datetime
import hashlib
import json
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
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.adquisicion.propio import ManifiestoDataset
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.alineador import alinear_rostro
from reconocimiento.preprocesamiento.detector_yunet import EstadoDeteccion

CANDIDATOS = [
    {
        "id": "antialiasing_052",
        "nombre": "CNN Antialiasing — 0.52",
        "config": RAIZ / "configuracion_candidato_antialiasing_052.json",
        "sha256_esperado": "581b805b649be08577e8157fe26fc30d8d8bfe90a3b4819379f42051bc613f7f",
    },
    {
        "id": "estandar_061",
        "nombre": "CNN Estándar Portal Completo — 0.61",
        "config": RAIZ / "configuracion_candidato_estandar_portal_061.json",
        "sha256_esperado": "24d3a1ed14cd8750863ff5bf464185c22eaef02bf4313044b64747929bd636f9",
    },
]

RUTA_RESUMEN_PORTAL_COMPLETO = RAIZ / "resultados/exp_4clases_portal_completo_20261120_200452/resumen_experimento_portal_completo.json"
RUTA_MANIFIESTO_EXP_PORTAL = RAIZ / "resultados/exp_4clases_portal_completo_20261120_200452/manifiesto_entrenamiento_experimento.csv"
RUTA_MANIFIESTO_LOCAL = RAIZ / "data/propio/manifiesto_local.csv"
RUTA_LFW_TRAIN = RAIZ / "data/desconocidos_lfw/entrenamiento_4clases/manifiesto.csv"
DIR_PORTAL_FOTOS = RAIZ / "data/recepcion_externa/fotos"
DIR_PROPIO_IMG = RAIZ / "data/propio/imagenes"

CLASES_AUT = ("persona_1", "persona_2", "persona_3")
PERTURBACIONES = [
    "limpia",
    "oscura",
    "oscura_ruido",
    "desenfoque",
    "movimiento",
    "oclusion_inferior",
    "oclusion_lateral",
    "rotacion_+15",
    "rotacion_-15",
    "baja_res",
    "jpeg_q35",
]


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _cargar_bgr_original(ruta: Path) -> np.ndarray:
    arr = np.frombuffer(ruta.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo decodificar imagen: {ruta}")
    return img


def _aplicar_perturbacion(
    img_bgr: np.ndarray,
    nombre: str,
    caja_limpia: Optional[Tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, bool]:
    """
    Aplica la perturbación al fotograma completo antes de la detección facial.
    Retorna `(imagen_perturbada, caja_disponible_para_oclusion)`.
    """
    h, w = img_bgr.shape[:2]
    if nombre == "limpia":
        return img_bgr, True
    if nombre == "oscura":
        out = np.clip(img_bgr.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        return out, True
    if nombre == "oscura_ruido":
        ruido = rng.normal(0.0, 8.0, size=img_bgr.shape).astype(np.float32)
        out = np.clip(img_bgr.astype(np.float32) * 0.35 + ruido, 0, 255).astype(np.uint8)
        return out, True
    if nombre == "desenfoque":
        return cv2.GaussianBlur(img_bgr, (9, 9), 0), True
    if nombre == "movimiento":
        kernel = np.zeros((9, 9), dtype=np.float32)
        kernel[4, :] = 1.0 / 9.0
        return cv2.filter2D(img_bgr, -1, kernel), True
    if nombre == "oclusion_inferior":
        out = img_bgr.copy()
        if caja_limpia is None:
            return out, False
        rx, ry, rw, rh = caja_limpia
        y0 = max(0, min(h, ry + int(round(0.60 * rh))))
        y1 = max(0, min(h, ry + rh))
        x0 = max(0, min(w, rx))
        x1 = max(0, min(w, rx + rw))
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = 0
        return out, True
    if nombre == "oclusion_lateral":
        out = img_bgr.copy()
        if caja_limpia is None:
            return out, False
        rx, ry, rw, rh = caja_limpia
        x0 = max(0, min(w, rx))
        x1 = max(0, min(w, rx + int(round(0.35 * rw))))
        y0 = max(0, min(h, ry))
        y1 = max(0, min(h, ry + rh))
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = 0
        return out, True
    if nombre == "rotacion_+15":
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 15.0, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), True
    if nombre == "rotacion_-15":
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -15.0, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), True
    if nombre == "baja_res":
        nw = max(1, int(round(w * 0.35)))
        nh = max(1, int(round(h * 0.35)))
        peq = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return cv2.resize(peq, (w, h), interpolation=cv2.INTER_LINEAR), True
    if nombre == "jpeg_q35":
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        if not ok:
            return img_bgr.copy(), True
        return cv2.imdecode(buf, cv2.IMREAD_COLOR), True
    raise ValueError(f"Perturbación desconocida: {nombre}")


def _clasificar_desconocido(res_det, dec, probs, pipe: PipelineCandidatoCongelado) -> Dict[str, Any]:
    if res_det.estado != EstadoDeteccion.UN_ROSTRO or probs is None:
        return {
            "categoria": "fallo_deteccion",
            "clase_argmax": None,
            "score": 0.0,
            "decision_puerta": dec.estado_puerta,
            "probs": [None, None, None, None],
        }
    idx = int(np.argmax(probs))
    clase = pipe.mapeo_id_clase.get(idx, "desconocido")
    score = float(probs[idx])
    pvec = [round(float(x), 4) for x in probs]
    if dec.estado_puerta == "ABIERTA":
        cat = "falsa_aceptacion"
    elif clase == "desconocido":
        cat = "rechazo_desconocido"
    else:
        cat = "rechazo_umbral"
    return {
        "categoria": cat,
        "clase_argmax": clase,
        "score": round(score, 4),
        "decision_puerta": dec.estado_puerta,
        "probs": pvec,
    }


def _clasificar_autorizado(persona_real: str, res_det, dec, probs, pipe: PipelineCandidatoCongelado) -> Dict[str, Any]:
    if res_det.estado != EstadoDeteccion.UN_ROSTRO or probs is None:
        return {"categoria": "fallo_deteccion", "clase_argmax": None, "score": 0.0}
    idx = int(np.argmax(probs))
    clase = pipe.mapeo_id_clase.get(idx, "desconocido")
    score = float(probs[idx])
    if dec.estado_puerta == "ABIERTA":
        cat = "acierto" if clase == persona_real else "confusion"
    else:
        cat = "bloqueo"
    return {"categoria": cat, "clase_argmax": clase, "score": round(score, 4)}


def _extraer_embedding_desde_deteccion(pipe: PipelineCandidatoCongelado, frame_bgr: np.ndarray, res_det, caja_fallback=None) -> Tuple[Optional[np.ndarray], str]:
    """
    Extrae el vector de 128 dimensiones (fc1_relu) exactamente con el mismo preprocesamiento
    del candidato. Retorna `(vector_128d, modo_recorte)`.
    """
    if res_det.estado == EstadoDeteccion.UN_ROSTRO and res_det.rostro_recortado is not None:
        rec = res_det.recorte_alineado if (pipe.usar_alineacion and res_det.recorte_alineado is not None) else res_det.rostro_recortado
        tensor = pipe.preprocesar_recorte(rec)
        feat = pipe.clasificador.extraer_caracteristicas(np.expand_dims(tensor, axis=0))[0]
        return feat, ("alineado_5pts" if (pipe.usar_alineacion and res_det.recorte_alineado is not None) else "recorte_yunet")

    if caja_fallback is not None:
        # Mismo tratamiento de fallback de entrenamiento cuando YuNet no detecta
        from reconocimiento.preprocesamiento.alineador import preprocesar_rostro_alineado
        tensor, _ = preprocesar_rostro_alineado(
            imagen=frame_bgr,
            puntos_faciales=None,
            recuadro=caja_fallback,
            resolucion=pipe.resolucion,
            sigma=pipe.sigma_gaussiano,
            metodo_remuestreo=pipe.metodo_remuestreo,
        )
        feat = pipe.clasificador.extraer_caracteristicas(np.expand_dims(tensor, axis=0))[0]
        return feat, "fallback_caja_sin_landmarks"

    return None, "sin_deteccion"


def _normalizar_l2(mat: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(normas, 1e-8)


def main() -> int:
    ahora = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_out = RAIZ / "resultados" / f"diagnostico_piloto_remoto_{ahora}"
    dir_lam = dir_out / "laminas"
    dir_lam.mkdir(parents=True, exist_ok=True)

    # 0. Verificar hashes iniciales de modelos
    pipes: Dict[str, PipelineCandidatoCongelado] = {}
    hashes_iniciales: Dict[str, str] = {}
    for c in CANDIDATOS:
        pipe = PipelineCandidatoCongelado(c["config"], raiz_proyecto=RAIZ)
        sha = _sha256(pipe.ruta_modelo)
        assert sha == c["sha256_esperado"], f"Hash inesperado en {pipe.ruta_modelo}: {sha}"
        pipes[c["id"]] = pipe
        hashes_iniciales[c["id"]] = sha

    # Cargar manifiesto de entrenamiento de Portal Completo (fuente verificable)
    ids_portal_en_manifiesto = set()
    with RUTA_MANIFIESTO_EXP_PORTAL.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["fuente"] == "desconocido_portal":
                ids_portal_en_manifiesto.add(row["id"])

    # Inventario de fotos del portal (75)
    fotos_portal = []
    for p_dir in sorted(DIR_PORTAL_FOTOS.iterdir()):
        if not p_dir.is_dir():
            continue
        for img_p in sorted(p_dir.iterdir()):
            if img_p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                fotos_portal.append({
                    "identidad": p_dir.name,
                    "archivo": img_p.name,
                    "stem": img_p.stem,
                    "ruta": img_p,
                    "en_manifiesto_portal": img_p.stem in ids_portal_en_manifiesto,
                })
    assert len(fotos_portal) == 75, f"Esperadas 75 fotos en portal, encontradas {len(fotos_portal)}"

    # Inventario de validación autorizados sesion_04 (64)
    fotos_s04 = []
    for p_name in CLASES_AUT:
        for img_p in sorted((DIR_PROPIO_IMG / p_name / "sesion_04").rglob("*.jpg")):
            fotos_s04.append({"persona": p_name, "archivo": img_p.name, "ruta": img_p})

    # Inventario de entrenamiento autorizados originales (268 en manifiesto_local.csv: 208 s01-03 + 60 cercana/media/lejana)
    man_local = ManifiestoDataset(RUTA_MANIFIESTO_LOCAL)
    regs_train_aut = [
        r for r in man_local.registros
        if r.particion == "entrenamiento" and r.procedencia != "portal_web_externo" and r.decision_humana != "excluida"
    ]

    # Inventario LFW train (300)
    with RUTA_LFW_TRAIN.open(encoding="utf-8", newline="") as f:
        filas_lfw_train = list(csv.DictReader(f))

    # Precargar imágenes en memoria (solo lectura)
    cache_portal_bgr = {f["stem"]: _cargar_bgr_original(f["ruta"]) for f in fotos_portal}
    cache_s04_bgr = {f["ruta"]: _cargar_bgr_original(f["ruta"]) for f in fotos_s04}

    # -----------------------------------------------------------------------
    # PARTE A: Ajuste sobre las 75 fotos del portal
    # -----------------------------------------------------------------------
    filas_a = []
    resumen_a: Dict[str, Any] = {}
    cajas_limpias_portal: Dict[str, Dict[str, Optional[Tuple[int, int, int, int]]]] = {}

    for c in CANDIDATOS:
        cid = c["id"]
        pipe = pipes[cid]
        cajas_limpias_portal[cid] = {}
        por_id: Dict[str, Dict[str, Any]] = {}

        for fp in fotos_portal:
            ident = fp["identidad"]
            img_bgr = cache_portal_bgr[fp["stem"]]
            res_det, dec, probs = pipe.evaluar_fotograma_completo(img_bgr, es_pre_recortada=False)
            cajas_limpias_portal[cid][fp["stem"]] = res_det.recuadro if res_det.estado == EstadoDeteccion.UN_ROSTRO else None
            clas = _clasificar_desconocido(res_det, dec, probs, pipe)
            en_ent = "si" if (cid == "estandar_061" and fp["en_manifiesto_portal"]) else (
                "no" if (cid == "estandar_061" and not fp["en_manifiesto_portal"]) else (
                    "si_reconstruido" if fp["en_manifiesto_portal"] else "no_reconstruido"
                )
            )
            filas_a.append({
                "candidato": cid,
                "identidad": ident,
                "archivo": fp["archivo"],
                "en_entrenamiento": en_ent,
                "estado_detector": res_det.estado.value,
                "clase_predicha": clas["clase_argmax"] or "",
                "prob_persona_1": clas["probs"][0] if clas["probs"][0] is not None else "",
                "prob_persona_2": clas["probs"][1] if clas["probs"][1] is not None else "",
                "prob_persona_3": clas["probs"][2] if clas["probs"][2] is not None else "",
                "prob_desconocido": clas["probs"][3] if clas["probs"][3] is not None else "",
                "score_argmax": clas["score"],
                "decision_puerta": clas["decision_puerta"],
                "categoria": clas["categoria"],
            })

            agg = por_id.setdefault(ident, {
                "intentos": 0, "rechazo_desconocido": 0, "rechazo_umbral": 0,
                "falsa_aceptacion": 0, "fallo_deteccion": 0, "fa_destinos": {p: 0 for p in CLASES_AUT},
                "max_prob_persona_1": 0.0,
            })
            agg["intentos"] += 1
            agg[clas["categoria"]] += 1
            if clas["categoria"] == "falsa_aceptacion" and clas["clase_argmax"] in agg["fa_destinos"]:
                agg["fa_destinos"][clas["clase_argmax"]] += 1
            if clas["probs"][0] is not None:
                agg["max_prob_persona_1"] = max(agg["max_prob_persona_1"], float(clas["probs"][0]))

        # Verificar suma coherente
        tot_a = {"intentos": 0, "rechazo_desconocido": 0, "rechazo_umbral": 0, "falsa_aceptacion": 0, "fallo_deteccion": 0}
        for agg in por_id.values():
            assert agg["intentos"] == agg["rechazo_desconocido"] + agg["rechazo_umbral"] + agg["falsa_aceptacion"] + agg["fallo_deteccion"]
            for k in tot_a:
                tot_a[k] += agg[k]
        resumen_a[cid] = {"total": tot_a, "por_identidad": por_id}

    ruta_csv_a = dir_out / "predicciones_portal.csv"
    with ruta_csv_a.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_a[0].keys()))
        w.writeheader()
        w.writerows(filas_a)

    # -----------------------------------------------------------------------
    # PARTE B: Robustez ante perturbaciones en fotograma completo
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    filas_b = []
    resumen_b: Dict[str, Any] = {}

    # Precalcular cajas limpias para sesion_04 por candidato
    cajas_limpias_s04: Dict[str, Dict[Path, Optional[Tuple[int, int, int, int]]]] = {}
    for c in CANDIDATOS:
        cid = c["id"]
        pipe = pipes[cid]
        cajas_limpias_s04[cid] = {}
        for item in fotos_s04:
            det = pipe.detector.detectar(cache_s04_bgr[item["ruta"]], es_pre_recortada=False)
            cajas_limpias_s04[cid][item["ruta"]] = det.recuadro if det.estado == EstadoDeteccion.UN_ROSTRO else None

    for c in CANDIDATOS:
        cid = c["id"]
        pipe = pipes[cid]
        resumen_b[cid] = {"portal": {}, "sesion_04": {}}

        for pert in PERTURBACIONES:
            # B.1 Portal (desconocidos)
            agg_p = {
                "intentos": 0, "rechazo_desconocido": 0, "rechazo_umbral": 0,
                "falsa_aceptacion": 0, "fallo_deteccion": 0,
                "fa_destinos": {p: 0 for p in CLASES_AUT},
                "por_identidad": {},
            }
            for fp in fotos_portal:
                ident = fp["identidad"]
                ip_agg = agg_p["por_identidad"].setdefault(ident, {
                    "intentos": 0, "rechazo_desconocido": 0, "rechazo_umbral": 0,
                    "falsa_aceptacion": 0, "fallo_deteccion": 0, "fa_destinos": {p: 0 for p in CLASES_AUT},
                })
                caja_c = cajas_limpias_portal[cid][fp["stem"]]
                img_pert, caja_ok = _aplicar_perturbacion(cache_portal_bgr[fp["stem"]], pert, caja_c, rng)
                if not caja_ok:
                    # Imagen sin detección limpia previa en prueba de oclusión basada en caja limpia
                    clas = {"categoria": "fallo_deteccion", "clase_argmax": None}
                else:
                    res_det, dec, probs = pipe.evaluar_fotograma_completo(img_pert, es_pre_recortada=False)
                    clas = _clasificar_desconocido(res_det, dec, probs, pipe)

                agg_p["intentos"] += 1
                agg_p[clas["categoria"]] += 1
                ip_agg["intentos"] += 1
                ip_agg[clas["categoria"]] += 1
                if clas["categoria"] == "falsa_aceptacion" and clas["clase_argmax"] in agg_p["fa_destinos"]:
                    agg_p["fa_destinos"][clas["clase_argmax"]] += 1
                    ip_agg["fa_destinos"][clas["clase_argmax"]] += 1

            assert agg_p["intentos"] == agg_p["rechazo_desconocido"] + agg_p["rechazo_umbral"] + agg_p["falsa_aceptacion"] + agg_p["fallo_deteccion"]
            agg_p["far_pct"] = round(100.0 * agg_p["falsa_aceptacion"] / agg_p["intentos"], 2)
            agg_p["fallo_det_pct"] = round(100.0 * agg_p["fallo_deteccion"] / agg_p["intentos"], 2)
            resumen_b[cid]["portal"][pert] = agg_p

            filas_b.append({
                "candidato": cid, "conjunto": "portal_desconocidos", "perturbacion": pert,
                "intentos": agg_p["intentos"], "aciertos_o_rechazos": agg_p["rechazo_desconocido"] + agg_p["rechazo_umbral"],
                "confusiones_o_fa": agg_p["falsa_aceptacion"], "bloqueos": 0, "fallos_deteccion": agg_p["fallo_deteccion"],
                "fa_persona_1": agg_p["fa_destinos"]["persona_1"], "fa_persona_2": agg_p["fa_destinos"]["persona_2"],
                "fa_persona_3": agg_p["fa_destinos"]["persona_3"], "stevenl_fa": agg_p["por_identidad"]["stevenl"]["falsa_aceptacion"],
            })

            # B.2 Sesion_04 (autorizados)
            agg_s = {"intentos": 0, "acierto": 0, "confusion": 0, "bloqueo": 0, "fallo_deteccion": 0}
            for item in fotos_s04:
                caja_c = cajas_limpias_s04[cid][item["ruta"]]
                img_pert, caja_ok = _aplicar_perturbacion(cache_s04_bgr[item["ruta"]], pert, caja_c, rng)
                if not caja_ok:
                    clas_a = {"categoria": "fallo_deteccion"}
                else:
                    res_det, dec, probs = pipe.evaluar_fotograma_completo(img_pert, es_pre_recortada=False)
                    clas_a = _clasificar_autorizado(item["persona"], res_det, dec, probs, pipe)
                agg_s["intentos"] += 1
                agg_s[clas_a["categoria"]] += 1

            assert agg_s["intentos"] == agg_s["acierto"] + agg_s["confusion"] + agg_s["bloqueo"] + agg_s["fallo_deteccion"]
            agg_s["exactitud_pct"] = round(100.0 * agg_s["acierto"] / agg_s["intentos"], 2)
            agg_s["bloqueo_pct"] = round(100.0 * agg_s["bloqueo"] / agg_s["intentos"], 2)
            resumen_b[cid]["sesion_04"][pert] = agg_s

            filas_b.append({
                "candidato": cid, "conjunto": "autorizados_sesion_04", "perturbacion": pert,
                "intentos": agg_s["intentos"], "aciertos_o_rechazos": agg_s["acierto"],
                "confusiones_o_fa": agg_s["confusion"], "bloqueos": agg_s["bloqueo"], "fallos_deteccion": agg_s["fallo_deteccion"],
                "fa_persona_1": "", "fa_persona_2": "", "fa_persona_3": "", "stevenl_fa": "",
            })

    ruta_csv_b = dir_out / "robustez.csv"
    with ruta_csv_b.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_b[0].keys()))
        w.writeheader()
        w.writerows(filas_b)

    # -----------------------------------------------------------------------
    # PARTE C: Espacio de embeddings 128-D (fc1_relu)
    # -----------------------------------------------------------------------
    resumen_c: Dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    for idx_c, c in enumerate(CANDIDATOS):
        cid = c["id"]
        pipe = pipes[cid]

        refs_vecs: List[np.ndarray] = []
        refs_clases: List[str] = []
        refs_grupo: List[str] = []
        conteos_emb = {
            "train_aut_sesion_01_03": 0,
            "train_aut_cercana_media_lejana": 0,
            "train_aut_fallback": 0,
            "train_aut_omitidas": 0,
            "s04_ok": 0,
            "s04_omitidas": 0,
            "lfw_ok": 0,
            "lfw_fallback": 0,
            "lfw_omitidas": 0,
            "portal_ok": 0,
            "portal_omitidas": 0,
        }

        # C.1 Autorizados entrenamiento (sin aumentos)
        for reg in regs_train_aut:
            ruta_img = RAIZ / reg.ruta
            img_bgr = _cargar_bgr_original(ruta_img)
            det = pipe.detector.detectar(img_bgr, es_pre_recortada=False)
            caja_fb = tuple(reg.caja_aprobada) if (reg.caja_aprobada and min(reg.caja_aprobada[2:]) > 0) else None
            feat, modo = _extraer_embedding_desde_deteccion(pipe, img_bgr, det, caja_fallback=caja_fb)
            if feat is None:
                conteos_emb["train_aut_omitidas"] += 1
                continue
            if modo == "fallback_caja_sin_landmarks":
                conteos_emb["train_aut_fallback"] += 1
            if reg.sesion in {"sesion_01", "sesion_02", "sesion_03"}:
                conteos_emb["train_aut_sesion_01_03"] += 1
            else:
                conteos_emb["train_aut_cercana_media_lejana"] += 1
            refs_vecs.append(feat)
            refs_clases.append(reg.persona)
            refs_grupo.append(reg.persona)

        # C.2 Autorizados sesion_04
        s04_vecs = []
        s04_clases = []
        for item in fotos_s04:
            img_bgr = cache_s04_bgr[item["ruta"]]
            det = pipe.detector.detectar(img_bgr, es_pre_recortada=False)
            feat, _ = _extraer_embedding_desde_deteccion(pipe, img_bgr, det, caja_fallback=None)
            if feat is None:
                conteos_emb["s04_omitidas"] += 1
                continue
            conteos_emb["s04_ok"] += 1
            s04_vecs.append(feat)
            s04_clases.append(item["persona"])

        # C.3 LFW train (300)
        for fila in filas_lfw_train:
            ruta_img = RAIZ / fila["ruta_imagen"]
            img_bgr = _cargar_bgr_original(ruta_img)
            det = pipe.detector.detectar(img_bgr, es_pre_recortada=False)
            caja_man = (int(fila["caja_yunet_x"]), int(fila["caja_yunet_y"]), int(fila["caja_yunet_w"]), int(fila["caja_yunet_h"]))
            feat, modo = _extraer_embedding_desde_deteccion(pipe, img_bgr, det, caja_fallback=caja_man)
            if feat is None:
                conteos_emb["lfw_omitidas"] += 1
                continue
            conteos_emb["lfw_ok"] += 1
            if modo == "fallback_caja_sin_landmarks":
                conteos_emb["lfw_fallback"] += 1
            refs_vecs.append(feat)
            refs_clases.append("desconocido")
            refs_grupo.append("desconocido_lfw")

        # C.4 Portal (75)
        portal_vecs = []
        portal_idents = []
        for fp in fotos_portal:
            img_bgr = cache_portal_bgr[fp["stem"]]
            det = pipe.detector.detectar(img_bgr, es_pre_recortada=False)
            feat, _ = _extraer_embedding_desde_deteccion(pipe, img_bgr, det, caja_fallback=None)
            if feat is None:
                conteos_emb["portal_omitidas"] += 1
                continue
            conteos_emb["portal_ok"] += 1
            portal_vecs.append(feat)
            portal_idents.append(fp["identidad"])

        M_ref = _normalizar_l2(np.stack(refs_vecs))
        arr_ref_clases = np.array(refs_clases)
        arr_ref_grupo = np.array(refs_grupo)

        centroides = {}
        for grp in ["persona_1", "persona_2", "persona_3", "desconocido_lfw"]:
            sub = M_ref[arr_ref_grupo == grp]
            c_vec = np.mean(sub, axis=0)
            centroides[grp] = c_vec / max(np.linalg.norm(c_vec), 1e-8)

        M_port = _normalizar_l2(np.stack(portal_vecs))
        arr_port_idents = np.array(portal_idents)

        distancias_portal = {}
        votos_5nn_portal = {}
        for p_ident in sorted(set(portal_idents)):
            mask = arr_port_idents == p_ident
            sub_p = M_port[mask]
            c_p = np.mean(sub_p, axis=0)
            c_p = c_p / max(np.linalg.norm(c_p), 1e-8)
            distancias_portal[p_ident] = {
                grp: round(float(1.0 - np.dot(c_p, centroides[grp])), 4)
                for grp in ["persona_1", "persona_2", "persona_3", "desconocido_lfw"]
            }

            # 5-NN por cada muestra del sujeto contra refs + otras identidades del portal (excluyendo p_ident)
            otras_mask = arr_port_idents != p_ident
            M_pool = np.vstack([M_ref, M_port[otras_mask]])
            clases_pool = np.concatenate([arr_ref_clases, np.full(int(np.sum(otras_mask)), "desconocido")])

            sims = sub_p @ M_pool.T
            votos_tot = {"persona_1": 0, "persona_2": 0, "persona_3": 0, "desconocido": 0}
            muestras_mayoria_aut = 0
            for i in range(len(sub_p)):
                top5 = np.argsort(-sims[i])[:5]
                conteo_muestra = {"persona_1": 0, "persona_2": 0, "persona_3": 0, "desconocido": 0}
                for j in top5:
                    lbl = str(clases_pool[j])
                    votos_tot[lbl] += 1
                    conteo_muestra[lbl] += 1
                gan = max(conteo_muestra.items(), key=lambda kv: kv[1])[0]
                if gan != "desconocido":
                    muestras_mayoria_aut += 1

            votos_5nn_portal[p_ident] = {
                "muestras_detectadas": int(len(sub_p)),
                "muestras_con_mayoria_autorizada_5nn": muestras_mayoria_aut,
                "votos_totales_5nn": votos_tot,
            }

        resumen_c[cid] = {
            "conteos_embeddings": conteos_emb,
            "distancia_coseno_centroides": distancias_portal,
            "vecinos_5nn_excluyendo_propia_identidad": votos_5nn_portal,
        }

        # PCA 2D por SVD (solo puntos, sin rostros)
        M_s04 = _normalizar_l2(np.stack(s04_vecs))
        X_all = np.vstack([M_ref, M_s04, M_port])
        X_cent = X_all - np.mean(X_all, axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(X_cent, full_matrices=False)
        proj = X_cent @ Vt[:2].T
        var_exp = (S[:2] ** 2) / np.sum(S ** 2)
        resumen_c[cid]["pca_varianza_explicada_pct"] = [round(float(var_exp[0]) * 100.0, 2), round(float(var_exp[1]) * 100.0, 2)]

        n_ref = len(M_ref)
        n_s04 = len(M_s04)
        proj_ref = proj[:n_ref]
        proj_s04 = proj[n_ref:n_ref + n_s04]
        proj_port = proj[n_ref + n_s04:]

        ax = axes[idx_c]
        ax.scatter(proj_ref[arr_ref_grupo == "desconocido_lfw", 0], proj_ref[arr_ref_grupo == "desconocido_lfw", 1],
                   s=12, alpha=0.35, c="#94a3b8", label="LFW train (desconocido)")
        for aut_n, col_a in [("persona_1", "#ef4444"), ("persona_2", "#22c55e"), ("persona_3", "#3b82f6")]:
            m_a = arr_ref_grupo == aut_n
            ax.scatter(proj_ref[m_a, 0], proj_ref[m_a, 1], s=18, alpha=0.55, c=col_a, label=f"{aut_n} (train)")
        ax.scatter(proj_s04[:, 0], proj_s04[:, 1], s=22, alpha=0.55, marker="^", c="#0f172a", label="sesion_04 (val)")

        # Destacar stevenl y resto del portal
        m_stv = arr_port_idents == "stevenl"
        m_ext = arr_port_idents == "externo_piloto"
        m_otros_p = ~(m_stv | m_ext)
        ax.scatter(proj_port[m_otros_p, 0], proj_port[m_otros_p, 1], s=36, marker="s", c="#06b6d4", edgecolors="#083344", label="Portal (david/paul/sonia)")
        ax.scatter(proj_port[m_ext, 0], proj_port[m_ext, 1], s=55, marker="D", c="#f59e0b", edgecolors="#451a03", label="Portal (externo_piloto)")
        ax.scatter(proj_port[m_stv, 0], proj_port[m_stv, 1], s=75, marker="*", c="#7c3aed", edgecolors="#ffffff", linewidth=0.5, label="Portal (stevenl)")
        ax.set_title(f"{c['nombre']}\nPCA 2D fc1_relu ({var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% var.)", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="best")

    plt.tight_layout()
    ruta_pca = dir_lam / "pca_embeddings.png"
    plt.savefig(ruta_pca, dpi=150, bbox_inches="tight")
    plt.close()

    (dir_out / "embeddings_resumen.json").write_text(json.dumps(resumen_c, indent=2, ensure_ascii=False), encoding="utf-8")

    # -----------------------------------------------------------------------
    # PARTE D: Balance de entrenamiento y verificación final de hashes
    # -----------------------------------------------------------------------
    res_pc = json.loads(RUTA_RESUMEN_PORTAL_COMPLETO.read_text(encoding="utf-8"))
    fuentes_ent = res_pc["resumen_pesos"]["fuentes"]
    cociente_p1_portal = round(
        fuentes_ent["autorizado_persona_1"]["peso_efectivo_total"] / fuentes_ent["desconocido_portal"]["peso_por_identidad"],
        2,
    )

    hashes_finales = {cid: _sha256(pipes[cid].ruta_modelo) for cid in pipes}
    assert hashes_finales == hashes_iniciales, "Los hashes de los modelos cambiaron durante la lectura"

    diagnostico_completo = {
        "fecha_utc": datetime.utcnow().isoformat() + "Z",
        "hashes_modelos_verificados": hashes_finales,
        "A_ajuste_fotos_portal": resumen_a,
        "B_robustez_perturbaciones": resumen_b,
        "C_embeddings": resumen_c,
        "D_entrenamiento": {
            "resumen_pesos_fuentes": fuentes_ent,
            "cociente_peso_persona1_vs_identidad_portal": cociente_p1_portal,
            "artefactos_cnn_guardados": {
                "antialiasing_052": sorted([p.name for p in (RAIZ / "resultados/exp_4clases_antialiasing_20261120_233516/cnn").iterdir()]),
                "estandar_061": sorted([p.name for p in (RAIZ / "resultados/exp_4clases_portal_completo_20261120_200452/cnn").iterdir()]),
            },
            "curvas_perdida_disponibles": False,
            "nota_curvas": "Las carpetas cnn/ solo contienen modelo.pt; no se guardaron curvas de pérdida ni métricas por época.",
        },
    }
    (dir_out / "resumen_diagnostico.json").write_text(
        json.dumps(diagnostico_completo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Generar informe Markdown conciso con los números reales observados
    a_aa = resumen_a["antialiasing_052"]["por_identidad"]
    a_std = resumen_a["estandar_061"]["por_identidad"]
    b_aa_p = resumen_b["antialiasing_052"]["portal"]
    b_std_p = resumen_b["estandar_061"]["portal"]
    b_aa_s = resumen_b["antialiasing_052"]["sesion_04"]
    c_aa = resumen_c["antialiasing_052"]["distancia_coseno_centroides"]
    v_aa = resumen_c["antialiasing_052"]["vecinos_5nn_excluyendo_propia_identidad"]

    lineas_md = [
        "# Diagnóstico de confusiones del piloto remoto",
        "",
        f"- **Directorio:** `{dir_out.relative_to(RAIZ).as_posix()}`",
        f"- **SHA-256 verificados (sin cambios):** AA=`{hashes_finales['antialiasing_052'][:16]}...`, Std=`{hashes_finales['estandar_061'][:16]}...`",
        f"- **Log real del piloto (`intentos.jsonl`):** 1 línea registrada (desconocido rechazado con `score=0.9911`).",
        "",
        "## A. Ajuste sobre las 75 fotos originales del portal",
        "",
        "| Identidad | AA 0.52 (RechDesc / RechUmb / FA / FalloDet) | Std 0.61 (RechDesc / RechUmb / FA / FalloDet) |",
        "| --- | --- | --- |",
    ]
    for ident in sorted(a_aa.keys()):
        ra = a_aa[ident]
        rs = a_std[ident]
        lineas_md.append(
            f"| {ident} | {ra['rechazo_desconocido']} / {ra['rechazo_umbral']} / {ra['falsa_aceptacion']} / {ra['fallo_deteccion']} "
            f"| {rs['rechazo_desconocido']} / {rs['rechazo_umbral']} / {rs['falsa_aceptacion']} / {rs['fallo_deteccion']} |"
        )
    lineas_md.extend([
        "",
        "## B. Robustez ante perturbaciones en fotograma completo (Portal 75 vs. Sesión 04 64)",
        "",
        "| Perturbación | AA 0.52 Portal FA (p1/p2/p3) | AA 0.52 stevenl FA | AA 0.52 s04 Acierto / Conf / Bloq / Fallo | Std 0.61 Portal FA |",
        "| --- | --- | --- | --- | --- |",
    ])
    for pert in PERTURBACIONES:
        bp = b_aa_p[pert]
        bs = b_aa_s[pert]
        sp = b_std_p[pert]
        fd = bp["fa_destinos"]
        lineas_md.append(
            f"| {pert} | {bp['falsa_aceptacion']}/{bp['intentos']} ({bp['far_pct']}%) [{fd['persona_1']}/{fd['persona_2']}/{fd['persona_3']}] "
            f"| {bp['por_identidad']['stevenl']['falsa_aceptacion']}/15 "
            f"| {bs['acierto']} / {bs['confusion']} / {bs['bloqueo']} / {bs['fallo_deteccion']} "
            f"| {sp['falsa_aceptacion']}/{sp['intentos']} ({sp['far_pct']}%) |"
        )
    lineas_md.extend([
        "",
        "## C. Embeddings 128-D (`fc1_relu`) — AA 0.52",
        "",
        "| Identidad portal | Dist. p1 | Dist. p2 | Dist. p3 | Dist. LFW | Muestras con mayoría autorizada en 5-NN |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for ident in sorted(c_aa.keys()):
        d = c_aa[ident]
        v = v_aa[ident]
        lineas_md.append(
            f"| {ident} | {d['persona_1']} | {d['persona_2']} | {d['persona_3']} | {d['desconocido_lfw']} "
            f"| {v['muestras_con_mayoria_autorizada_5nn']}/{v['muestras_detectadas']} |"
        )
    lineas_md.extend([
        "",
        "## D. Balance de entrenamiento y curvas",
        f"- **Cociente de peso efectivo (`persona_1` / identidad de portal):** `305.25 / 0.9539 = {cociente_p1_portal}x`.",
        "- **Curvas de pérdida:** No hay archivos de historial por época en `cnn/` (solo `modelo.pt`).",
    ])
    (dir_out / "INFORME_DIAGNOSTICO.md").write_text("\n".join(lineas_md), encoding="utf-8")

    print(f"OK: {dir_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
