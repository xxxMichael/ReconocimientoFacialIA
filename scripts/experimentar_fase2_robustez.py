"""
Fase 2: Iteración controlada de entrenamiento enfocada en robustez frente a
contraluz, oscuridad, oclusión parcial y negativos difíciles del portal.

Preserva 100% intactos los modelos actuales (0.61, 0.61+calidad, 0.52).
Guarda todos los artefactos en resultados/exp_4clases_fase2_robustez_<timestamp>/
y exporta configuracion_candidato_fase2_robustez.json sin reemplazar la predeterminada.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.detector_yunet import EstadoDeteccion
from scripts.experimentar_alineacion import (
    CLASES_NOMBRES,
    CLASE_TO_ID,
    ID_TO_CLASE,
    N_CLASES,
    PERSONAS_AUTORIZADAS,
    calcular_sha256,
)
from scripts.experimentar_4clases_universidad import evaluar_modelo_sobre_validacion
from scripts.experimentar_antialiasing_alineacion import SEMILLA, preparar_datos_con_remuestreo
from scripts.diagnosticar_confusiones_piloto import (
    _aplicar_perturbacion,
    _cargar_bgr_original,
    _clasificar_autorizado,
)


def _aplicar_contraluz_frame(img_bgr: np.ndarray) -> np.ndarray:
    """Simula contraluz en fotograma completo: borde superior/fondo muy brillante y rostro subexpuesto."""
    h, w = img_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.52, w * 0.50
    dist = np.sqrt(((yy - cy) / (h * 0.45)) ** 2 + ((xx - cx) / (w * 0.40)) ** 2)
    halo = np.clip((dist - 0.55) / 0.85, 0.0, 1.0)
    # Oscurecer centro (0.42x) y añadir resplandor en periferia/parte superior
    out = img_bgr.astype(np.float32) * (0.42 + 0.38 * halo[..., None]) + (85.0 * halo[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _aug_oscuridad_contraluz(img_64: np.ndarray, rng: np.random.RandomState, es_autorizado: bool) -> np.ndarray:
    """Genera variación de oscuridad o contraluz sobre tensor 64x64 [0, 1]."""
    out = img_64.astype(np.float32).copy()
    if rng.rand() < 0.5:
        # Oscuridad uniforme + ruido leve
        factor = rng.uniform(0.42, 0.78) if es_autorizado else rng.uniform(0.30, 0.65)
        out = out * factor
    else:
        # Contraluz: parte superior/bordes más brillantes, centro del rostro más oscuro
        yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
        dist = np.sqrt(((yy - 34.0) / 28.0) ** 2 + ((xx - 32.0) / 26.0) ** 2)
        halo = np.clip((dist - 0.45) / 0.8, 0.0, 1.0)
        factor_centro = rng.uniform(0.45, 0.70) if es_autorizado else rng.uniform(0.32, 0.58)
        out = out * (factor_centro + 0.35 * halo) + rng.uniform(0.15, 0.32) * halo

    ruido = rng.normal(0.0, rng.uniform(0.008, 0.025), size=out.shape).astype(np.float32)
    out = np.clip(out + ruido, 0.0, 1.0)
    return cv2.GaussianBlur(out, (5, 5), 0.5).astype(np.float32)


def _aug_oclusion_o_desenfoque(img_64: np.ndarray, rng: np.random.RandomState, es_autorizado: bool) -> np.ndarray:
    """
    Para desconocidos: oclusión inferior/lateral marcada (refuerza rechazo ante tapas/oclusiones).
    Para autorizados: oclusión leve de mentón o desenfoque/rotación suave.
    """
    out = img_64.astype(np.float32).copy()
    if not es_autorizado:
        if rng.rand() < 0.7:
            # Oclusión inferior (30%–42%) o lateral (30%)
            val = float(rng.uniform(0.02, 0.50))
            if rng.rand() < 0.7:
                y0 = int(rng.randint(37, 45))
                out[y0:, :] = val
            else:
                x1 = int(rng.randint(18, 24))
                out[:, :x1] = val
        else:
            out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.8, 1.4))
    else:
        if rng.rand() < 0.45:
            # Sombra / cobertura muy leve solo en mentón inferior (<= 18%)
            out[52:, :] = out[52:, :] * rng.uniform(0.25, 0.55)
        else:
            out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.6, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _calcular_pesos_fase2(y_train: np.ndarray, fuentes: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Mantiene 25% por clase (persona_1/2/3 y desconocido), pero dentro de desconocido
    asigna 30% a desconocido_portal, 15% a desconocido_universidad y 55% a desconocido_lfw
    para evitar la subrepresentación 1/320 de los negativos difíciles conocidos.
    """
    n_total = len(y_train)
    peso_por_clase = n_total / 4.0
    pesos = np.ones(n_total, dtype=np.float32)

    for cid, nombre in ID_TO_CLASE.items():
        if nombre != "desconocido":
            mask = (y_train == cid)
            cnt = int(np.sum(mask))
            if cnt > 0:
                pesos[mask] = peso_por_clase / cnt

    sub_dist = {
        "desconocido_portal": 0.30 * peso_por_clase,
        "desconocido_universidad": 0.15 * peso_por_clase,
        "desconocido_lfw": 0.55 * peso_por_clase,
    }
    arr_f = np.array(fuentes)
    resumen_f = {}
    for f_nom, w_tot in sub_dist.items():
        m_f = (arr_f == f_nom)
        cnt = int(np.sum(m_f))
        if cnt > 0:
            pesos[m_f] = w_tot / cnt
            resumen_f[f_nom] = {"muestras": cnt, "peso_total": round(float(w_tot), 2), "peso_por_muestra": round(float(w_tot / cnt), 4)}

    return pesos, resumen_f


def main() -> int:
    ahora = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_out = RAIZ / "resultados" / f"exp_4clases_fase2_robustez_{ahora}"
    dir_out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Cargando particiones base idénticas (estandar)...")
    datos = preparar_datos_con_remuestreo("estandar")
    X_base = datos["X_train_img"]
    y_base = datos["y_train"]
    assert len(y_base) == 1221, f"Esperadas 1221 muestras base, obtenidas {len(y_base)}"

    # Cargar manifiesto original para etiquetar fuente y variante exacta de las 1221 filas
    ruta_man_orig = RAIZ / "resultados/exp_4clases_portal_completo_20261120_200452/manifiesto_entrenamiento_experimento.csv"
    with ruta_man_orig.open(encoding="utf-8", newline="") as f:
        filas_man = list(csv.DictReader(f))
    assert len(filas_man) == 1221

    rng = np.random.RandomState(SEMILLA)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    fuentes_list: List[str] = []

    # Seleccionar 100 LFW deterministas para aumentos de robustez
    idx_lfw = [i for i, r in enumerate(filas_man) if r["fuente"] == "desconocido_lfw"]
    idx_lfw_aug = set(rng.choice(idx_lfw, size=min(100, len(idx_lfw)), replace=False).tolist())

    conteo_fase2 = {"base": 0, "aug_oscuridad_contraluz": 0, "aug_oclusion_desenfoque": 0}
    for i, r in enumerate(filas_man):
        img = X_base[i]
        cid = int(y_base[i])
        fuente = r["fuente"]
        variante = r["variante"]

        X_list.append(img)
        y_list.append(cid)
        fuentes_list.append(fuente)
        conteo_fase2["base"] += 1

        es_orig = (variante == "original")
        aplicar_extra = (
            (fuente.startswith("autorizado_") and es_orig)
            or (fuente in {"desconocido_portal", "desconocido_universidad"} and es_orig)
            or (i in idx_lfw_aug)
        )
        if aplicar_extra:
            es_aut = fuente.startswith("autorizado_")
            X_list.append(_aug_oscuridad_contraluz(img, rng, es_aut))
            y_list.append(cid)
            fuentes_list.append(fuente)
            conteo_fase2["aug_oscuridad_contraluz"] += 1

            X_list.append(_aug_oclusion_o_desenfoque(img, rng, es_aut))
            y_list.append(cid)
            fuentes_list.append(fuente)
            conteo_fase2["aug_oclusion_desenfoque"] += 1

    X_train_f2 = np.stack(X_list).astype(np.float32)
    y_train_f2 = np.asarray(y_list, dtype=np.int64)
    pesos_f2, resumen_pesos_desc = _calcular_pesos_fase2(y_train_f2, fuentes_list)
    print(f"  Total entrenamiento Fase 2: {len(y_train_f2)} muestras ({conteo_fase2})")

    print("[2/5] Entrenando CNN Fase 2 (misma arquitectura, semilla=42)...")
    cnn_f2 = ClasificadorCNN(n_clases=N_CLASES, semilla=SEMILLA)
    cnn_f2.entrenar_con_validacion(
        X_train_f2,
        y_train_f2,
        datos["X_val_img"],
        datos["y_val"],
        pesos_clases=None,
        pesos_muestras=pesos_f2,
        metrica_checkpoint="macro_recall",
    )
    ruta_mod_f2 = dir_out / "cnn" / "modelo.pt"
    ruta_mod_f2.parent.mkdir(parents=True, exist_ok=True)
    cnn_f2.guardar(ruta_mod_f2)

    print("[3/5] Calibrando umbral exclusivamente sobre validación...")
    res_val_f2 = evaluar_modelo_sobre_validacion("cnn_fase2_robustez", cnn_f2, datos, es_cnn=True, umbral_fijo=None)
    th_f2 = res_val_f2["umbral_recomendado"]
    if th_f2 is None:
        # Fallback si no hubo umbral viable automático (no esperado)
        th_f2 = 0.61

    pipe_ref = PipelineCandidatoCongelado(RAIZ / "configuracion_candidato_estandar_portal_061.json", raiz_proyecto=RAIZ)
    cfg_f2 = {
        "candidato": {
            "identificador": "cnn_4clases_fase2_robustez",
            "nombre": f"CNN Fase 2 Robustez — {th_f2:.2f}",
            "tipo_arquitectura": "CNN",
            "ruta_modelo": str(ruta_mod_f2.relative_to(RAIZ)).replace("\\", "/"),
            "sha256_modelo": calcular_sha256(ruta_mod_f2),
            "umbral_score": round(float(th_f2), 2),
            "orden_clases": CLASES_NOMBRES,
            "mapeo_clases": {str(k): v for k, v in ID_TO_CLASE.items()},
            "personas_autorizadas": PERSONAS_AUTORIZADAS,
            "fecha_propuesta": datetime.now().isoformat(),
            "estado": "fase2_experimental_sin_reemplazar_default",
        },
        "detector_facial": pipe_ref.config["detector_facial"],
        "preprocesamiento": pipe_ref.config["preprocesamiento"],
        "regla_acceso": pipe_ref.config["regla_acceso"],
    }
    ruta_cfg_f2 = RAIZ / "configuracion_candidato_fase2_robustez.json"
    ruta_cfg_f2.write_text(json.dumps(cfg_f2, indent=2, ensure_ascii=False), encoding="utf-8")
    (dir_out / "configuracion_candidato_fase2_robustez.json").write_text(
        json.dumps(cfg_f2, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[4/5] Evaluando comparación controlada vs CNN Estándar 0.61 en sesión_05 y robustez...")
    pipe_f2 = PipelineCandidatoCongelado(ruta_cfg_f2, raiz_proyecto=RAIZ)

    # Evaluación en sesion_05 (63)
    s05 = sorted((RAIZ / "data/propio/imagenes").glob("persona_*/sesion_05/**/*.jpg"))
    res_s05 = {}
    for nom, pipe in [("estandar_061", pipe_ref), ("fase2_robustez", pipe_f2)]:
        ac = conf = bloq = fd = 0
        for p in s05:
            persona = p.parts[-4]
            det, dec, probs = pipe.evaluar_fotograma_completo(_cargar_bgr_original(p))
            c_a = _clasificar_autorizado(persona, det, dec, probs, pipe)
            if c_a["categoria"] == "acierto":
                ac += 1
            elif c_a["categoria"] == "confusion":
                conf += 1
            elif c_a["categoria"] == "bloqueo":
                bloq += 1
            else:
                fd += 1
        res_s05[nom] = {"intentos": len(s05), "aciertos": ac, "confusiones": conf, "bloqueos": bloq, "fallos_det": fd}

    # Evaluación en robustez (portal 75 + sesion_04 64) incluyendo contraluz
    portal = sorted((RAIZ / "data/recepcion_externa/fotos").glob("*/*"))
    s04 = sorted((RAIZ / "data/propio/imagenes").glob("persona_*/sesion_04/**/*.jpg"))
    cache_port = {p: _cargar_bgr_original(p) for p in portal}
    cache_s04 = {p: _cargar_bgr_original(p) for p in s04}
    cajas_port = {p: pipe_ref.detector.detectar(img).recuadro for p, img in cache_port.items()}
    cajas_s04 = {p: pipe_ref.detector.detectar(img).recuadro for p, img in cache_s04.items()}

    perts = ["limpia", "oscura", "oscura_ruido", "contraluz", "oclusion_inferior", "oclusion_lateral", "rotacion_+15", "desenfoque"]
    rng_eval = np.random.default_rng(42)
    comp_rob = {"portal": {}, "sesion_04": {}}

    for pert in perts:
        comp_rob["portal"][pert] = {}
        comp_rob["sesion_04"][pert] = {}
        for nom, pipe in [("estandar_061", pipe_ref), ("fase2_robustez", pipe_f2)]:
            fa = 0
            for p in portal:
                if pert == "contraluz":
                    img_p, ok_c = _aplicar_contraluz_frame(cache_port[p]), True
                else:
                    img_p, ok_c = _aplicar_perturbacion(cache_port[p], pert, cajas_port[p], rng_eval)
                if not ok_c:
                    continue
                _, dec, _ = pipe.evaluar_fotograma_completo(img_p)
                if dec.estado_puerta == "ABIERTA":
                    fa += 1
            comp_rob["portal"][pert][nom] = {"intentos": len(portal), "falsa_aceptacion": fa, "far_pct": round(100.0 * fa / len(portal), 2)}

            ac = conf = bloq = 0
            for p in s04:
                persona = p.parts[-4]
                if pert == "contraluz":
                    img_p, ok_c = _aplicar_contraluz_frame(cache_s04[p]), True
                else:
                    img_p, ok_c = _aplicar_perturbacion(cache_s04[p], pert, cajas_s04[p], rng_eval)
                if not ok_c:
                    bloq += 1
                    continue
                det, dec, probs = pipe.evaluar_fotograma_completo(img_p)
                c_a = _clasificar_autorizado(persona, det, dec, probs, pipe)
                if c_a["categoria"] == "acierto":
                    ac += 1
                elif c_a["categoria"] == "confusion":
                    conf += 1
                else:
                    bloq += 1
            comp_rob["sesion_04"][pert][nom] = {"intentos": len(s04), "aciertos": ac, "confusiones": conf, "bloqueos_o_det": bloq}

    print("[5/5] Guardando resumen Fase 2...")
    res_f2_limpio = dict(res_val_f2)
    res_f2_limpio.pop("curva_umbrales", None)
    if res_f2_limpio.get("regla_b"):
        res_f2_limpio["regla_b"].pop("decisiones_individuales", None)

    resumen = {
        "fecha_utc": datetime.utcnow().isoformat() + "Z",
        "umbral_calibrado_fase2": th_f2,
        "sha256_modelo_fase2": cfg_f2["candidato"]["sha256_modelo"],
        "conteo_muestras_entrenamiento": conteo_fase2,
        "pesos_desconocido_fase2": resumen_pesos_desc,
        "validacion_fase2": res_f2_limpio,
        "sesion_05": res_s05,
        "robustez_comparativa": comp_rob,
    }
    (dir_out / "resumen_fase2.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
