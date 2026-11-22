"""
Experimento acotado de remuestreo con antialiasing (supersampling + cv2.resize(INTER_AREA))
en la alineación facial de 5 puntos, comparado contra la CNN Portal Completo (umbral 0.61).

Aísla exclusivamente el efecto de `metodo_remuestreo = "supersampling_area"` frente a `"estandar"`:
- Mismo conjunto de entrenamiento exacto de Portal Completo (1,221 muestras, sin las 3 rescatadas).
- Mismo balance por identidad/clase, aumentos, semilla (42), arquitectura CNN e hiperparámetros.
- Misma validación automática (64 Sesión 04 + 100 LFW + 7 Univ), sin tocar Sesión 05, reserva LFW ni 9 Univ.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.adquisicion.propio import ManifiestoDataset, calcular_hash_archivo
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.clasificacion.regla_acceso import evaluar_regla_acceso_completa
from reconocimiento.preprocesamiento.alineador import alinear_rostro, preprocesar_rostro_alineado
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from scripts.calibrar_desarrollo import CALIBRACION_LFW
from scripts.entrenar_desarrollo import (
    BINS,
    CUADRICULA,
    RAIZ_IMAGENES,
    REVISIONES_VALIDAS,
    RESOLUCION,
    SEMILLA,
    SIGMA,
    SOLAPAMIENTO_MINIMO,
    cargar_registros,
)
from scripts.experimentar_alineacion import (
    CLASES_NOMBRES,
    CLASE_TO_ID,
    ID_TO_CLASE,
    N_CLASES,
    PERSONAS_AUTORIZADAS,
    aumentar_muestra_entrenamiento,
    calcular_macro_recall,
    calcular_sha256,
)
from scripts.experimentar_4clases_universidad import (
    cargar_manifiesto_universidad_desarrollo,
    evaluar_modelo_sobre_validacion,
)
from scripts.experimentar_4clases_portal_completo import calcular_pesos_identidad_y_clase


def verificar_antialiasing_sintetico_y_lamina(dir_salida: Path) -> Dict[str, Any]:
    """
    1. Comprueba con patrón sintético de alta frecuencia que supersampling_area reduce aliasing.
    2. Genera lámina comparativa con imágenes reales a distintas escalas.
    """
    H, W = 512, 512
    x = np.arange(W)
    patron = (((x // 2) % 2) * 255).astype(np.uint8)
    img_hf = cv2.cvtColor(np.tile(patron, (H, 1)), cv2.COLOR_GRAY2BGR)
    # Rotación moderada (5°) y escala no entera para probar aliasing real tipo moiré
    pts_hf = np.array([
        [172.0, 185.0], [338.0, 200.0], [250.0, 285.0], [182.0, 376.0], [312.0, 388.0]
    ], dtype=np.float32)

    rec_est, meta_est = alinear_rostro(img_hf, pts_hf, resolucion=(64, 64), metodo_remuestreo="estandar")
    rec_aa, meta_aa = alinear_rostro(img_hf, pts_hf, resolucion=(64, 64), metodo_remuestreo="supersampling_area")

    std_est = float(np.std(rec_est[10:54, 10:54]))
    std_aa = float(np.std(rec_aa[10:54, 10:54]))
    assert std_aa < std_est * 0.1, f"El método antialiasing debe reducir drásticamente el aliasing ({std_aa} vs {std_est})"

    # Lámina real a distintas escalas
    dir_lam = dir_salida / "laminas"
    dir_lam.mkdir(parents=True, exist_ok=True)
    det = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)

    muestras_ejemplo = [
        ("Autorizado cercano", RAIZ / "data/propio/imagenes/persona_1/sesion_cercana/frontal/persona_1_sesion_cercana_frontal_f01.jpg"),
        ("Autorizado lejano", RAIZ / "data/propio/imagenes/persona_1/sesion_lejana/frontal/persona_1_sesion_lejana_frontal_f01.jpg"),
        ("Portal movil", RAIZ / "data/recepcion_externa/fotos/stevenl/stevenl_frontal_cercana.jpg"),
        ("Universidad alta res", RAIZ / "data/recepcion_externa/fotos_desc/image copy 10.png"),
    ]
    validos = [(tit, p) for tit, p in muestras_ejemplo if p.exists()]
    if validos:
        fig, axes = plt.subplots(len(validos), 3, figsize=(11, 3.3 * len(validos)))
        if len(validos) == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, (tit, p_img) in enumerate(validos):
            img_bgr = cv2.imread(str(p_img))
            res_d = det.detectar(img_bgr)
            prep_est, m_e = preprocesar_rostro_alineado(img_bgr, res_d.puntos_faciales, res_d.recuadro, (64, 64), SIGMA, "estandar")
            prep_aa, m_a = preprocesar_rostro_alineado(img_bgr, res_d.puntos_faciales, res_d.recuadro, (64, 64), SIGMA, "supersampling_area")

            axes[idx, 0].imshow(cv2.cvtColor(res_d.rostro_recortado, cv2.COLOR_BGR2RGB) if res_d.rostro_recortado is not None else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            axes[idx, 0].set_title(f"{tit}\n(escala={m_a.escala:.3f}, k={m_a.factor_sobremuestreo})", fontsize=9)
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(prep_est, cmap="gray")
            axes[idx, 1].set_title("Actual (warpAffine INTER_AREA)", fontsize=9)
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(prep_aa, cmap="gray")
            axes[idx, 2].set_title("Antialiasing (supersampling + resize)", fontsize=9)
            axes[idx, 2].axis("off")

        plt.tight_layout()
        plt.savefig(dir_lam / "comparacion_remuestreo_escalas.png", dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "patron_sintetico_std_estandar": round(std_est, 4),
        "patron_sintetico_std_antialiasing": round(std_aa, 4),
        "reduccion_aliasing_pct": round(100.0 * (1.0 - std_aa / max(std_est, 1e-6)), 2),
    }


def preparar_datos_con_remuestreo(metodo_remuestreo: str) -> Dict[str, Any]:
    """
    Prepara exactamente las mismas 1,221 muestras de entrenamiento y las mismas
    muestras de validación automática de Portal Completo, variando únicamente `metodo_remuestreo`.
    """
    detector = DetectorYuNet(
        modo_multiorientacion="cascada",
        alinear_rostro=True,
        metodo_remuestreo=metodo_remuestreo,
    )

    # 1. Autorizados
    registros_propio, _ = cargar_registros(solo_base=False)
    hashes_por_particion: Dict[str, str] = {}
    muestras_autorizadas = []
    imagenes_autorizadas: Dict[str, np.ndarray] = {}

    for reg in registros_propio:
        if reg.procedencia == "portal_web_externo":
            continue
        decision = reg.decision_humana
        entrenamiento = (reg.particion == "entrenamiento")
        caja = reg.caja_aprobada if decision in REVISIONES_VALIDAS else None
        referencia = caja if caja and min(caja[2:]) > 0 else None

        muestra = {
            "id": reg.identificador, "persona": reg.persona, "sesion": reg.sesion,
            "particion": reg.particion, "pose": reg.pose, "intento": False,
            "estado_deteccion": None, "estado": None,
        }
        if reg.tipo_sujeto != "autorizado" or not reg.consentimiento or decision == "excluida":
            muestra["estado"] = "excluida_por_protocolo"
            muestras_autorizadas.append(muestra)
            continue
        if entrenamiento and referencia is None:
            muestra["estado"] = "sin_caja_revisada"
            muestras_autorizadas.append(muestra)
            continue

        muestra["intento"] = True
        contenido = (RAIZ / reg.ruta).read_bytes()
        sha256 = hashlib.sha256(contenido).hexdigest()
        part_prev = hashes_por_particion.setdefault(sha256, reg.particion)
        if part_prev != reg.particion:
            raise ValueError(f"Duplicado entre particiones: {reg.identificador}")

        imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        det = detector.detectar(imagen)
        muestra["estado_deteccion"] = det.estado.value

        if det.estado != EstadoDeteccion.UN_ROSTRO:
            if entrenamiento and referencia:
                img_proc, _ = preprocesar_rostro_alineado(
                    imagen, puntos_faciales=None, recuadro=referencia, resolucion=RESOLUCION, sigma=SIGMA, metodo_remuestreo=metodo_remuestreo
                )
                muestra["estado"] = "procesada"
                imagenes_autorizadas[reg.identificador] = img_proc
                muestras_autorizadas.append(muestra)
                continue
            muestra["estado"] = det.estado.value.lower()
            muestras_autorizadas.append(muestra)
            continue

        if referencia and det.recuadro:
            iou, iomin = detector.calcular_iou_y_iomin(det.recuadro, referencia)
            if max(iou, iomin) < SOLAPAMIENTO_MINIMO and not entrenamiento:
                muestra["estado"] = "localizacion_incorrecta"
                muestras_autorizadas.append(muestra)
                continue

        caja_final = referencia if (referencia and entrenamiento) else det.recuadro
        img_proc, _ = preprocesar_rostro_alineado(
            imagen=imagen, puntos_faciales=det.puntos_faciales, recuadro=caja_final,
            resolucion=RESOLUCION, sigma=SIGMA, metodo_remuestreo=metodo_remuestreo,
        )
        muestra["estado"] = "procesada"
        imagenes_autorizadas[reg.identificador] = img_proc
        muestras_autorizadas.append(muestra)

    train_auth = [m for m in muestras_autorizadas if m["particion"] == "entrenamiento" and m["estado"] == "procesada"]
    val_auth_todos = [m for m in muestras_autorizadas if m["particion"] == "validacion" and m["sesion"] == "sesion_04"]
    val_auth_proc = [m for m in val_auth_todos if m["estado"] == "procesada"]

    # 2. LFW train (300) y val (100)
    with (RAIZ / "data/desconocidos_lfw/entrenamiento_4clases/manifiesto.csv").open(encoding="utf-8", newline="") as f:
        filas_lfw_tr = list(csv.DictReader(f))
    lfw_tr_imgs = []
    lfw_tr_ids = []
    for fila in filas_lfw_tr:
        img_bgr = cv2.imdecode(np.frombuffer((RAIZ / fila["ruta_imagen"]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        caja_man = (int(fila["caja_yunet_x"]), int(fila["caja_yunet_y"]), int(fila["caja_yunet_w"]), int(fila["caja_yunet_h"]))
        det = detector.detectar(img_bgr)
        pts = det.puntos_faciales if det.estado == EstadoDeteccion.UN_ROSTRO else None
        img_proc, _ = preprocesar_rostro_alineado(img_bgr, pts, caja_man, RESOLUCION, SIGMA, metodo_remuestreo)
        lfw_tr_imgs.append(img_proc)
        lfw_tr_ids.append(f"lfw_train_{fila['identidad_normalizada'].replace(' ', '_')}")

    with (CALIBRACION_LFW / "manifiesto.csv").open(encoding="utf-8", newline="") as f:
        filas_lfw_val = list(csv.DictReader(f))
    lfw_val_imgs = []
    estados_lfw_val = []
    for fila in filas_lfw_val:
        img = cv2.imdecode(np.frombuffer((RAIZ / fila["ruta_imagen"]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        det = detector.detectar(img)
        estados_lfw_val.append(det.estado.value)
        if det.estado == EstadoDeteccion.UN_ROSTRO:
            img_proc, _ = preprocesar_rostro_alineado(img, det.puntos_faciales, det.recuadro, RESOLUCION, SIGMA, metodo_remuestreo)
            lfw_val_imgs.append(img_proc)

    # 3. Portal completo (72/75)
    man = ManifiestoDataset(RAIZ / "data/propio/manifiesto_local.csv")
    regs_portal = sorted(
        [r for r in man.registros if r.procedencia == "portal_web_externo" and r.particion == "prueba"],
        key=lambda r: (r.persona, r.pose, r.distancia or ""),
    )
    portal_imgs = []
    portal_ids = []
    for reg in regs_portal:
        img_bgr = cv2.imdecode(np.frombuffer((RAIZ / reg.ruta).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        det = detector.detectar(img_bgr)
        if det.estado == EstadoDeteccion.UN_ROSTRO:
            img_proc, _ = preprocesar_rostro_alineado(img_bgr, det.puntos_faciales, det.recuadro, RESOLUCION, SIGMA, metodo_remuestreo)
            portal_imgs.append(img_proc)
            portal_ids.append(f"portal_{reg.persona}")

    # 4. Universidad (15/18 train, 5/7 val automáticas — exactamente igual que Portal Completo)
    filas_univ_train, filas_univ_val = cargar_manifiesto_universidad_desarrollo()
    univ_tr_imgs = []
    univ_tr_ids = []
    for fila in filas_univ_train:
        img_bgr = cv2.imdecode(np.frombuffer((RAIZ / fila["ruta"]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        det = detector.detectar(img_bgr)
        if det.estado == EstadoDeteccion.UN_ROSTRO:
            img_proc, _ = preprocesar_rostro_alineado(img_bgr, det.puntos_faciales, det.recuadro, RESOLUCION, SIGMA, metodo_remuestreo)
            univ_tr_imgs.append(img_proc)
            univ_tr_ids.append(fila["identificador"])

    univ_val_imgs = []
    meta_univ_val = []
    for fila in filas_univ_val:
        img_bgr = cv2.imdecode(np.frombuffer((RAIZ / fila["ruta"]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        det = detector.detectar(img_bgr)
        if det.estado == EstadoDeteccion.UN_ROSTRO:
            img_proc, _ = preprocesar_rostro_alineado(img_bgr, det.puntos_faciales, det.recuadro, RESOLUCION, SIGMA, metodo_remuestreo)
            univ_val_imgs.append(img_proc)
            meta_univ_val.append({"id": fila["identificador"], "procesada": True, "estado_deteccion": det.estado.value})
        else:
            meta_univ_val.append({"id": fila["identificador"], "procesada": False, "estado_deteccion": det.estado.value})

    # 5. Aumentos EXCLUSIVAMENTE en entrenamiento con misma SEMILLA
    rng = np.random.RandomState(SEMILLA)
    X_train_list, y_train_list, ids_tr, fuentes_tr = [], [], [], []
    for m in train_auth:
        p = m["persona"]
        cid = CLASE_TO_ID[p]
        im_o = imagenes_autorizadas[m["id"]]
        X_train_list.append(im_o)
        y_train_list.append(cid)
        ids_tr.append(p)
        fuentes_tr.append(f"autorizado_{p}")
        for aum in aumentar_muestra_entrenamiento(im_o, rng):
            X_train_list.append(aum)
            y_train_list.append(cid)
            ids_tr.append(p)
            fuentes_tr.append(f"autorizado_{p}")

    cid_desc = CLASE_TO_ID["desconocido"]
    for i, im_o in enumerate(lfw_tr_imgs):
        X_train_list.append(im_o); y_train_list.append(cid_desc); ids_tr.append(lfw_tr_ids[i]); fuentes_tr.append("desconocido_lfw")
    for i, im_o in enumerate(portal_imgs):
        X_train_list.append(im_o); y_train_list.append(cid_desc); ids_tr.append(portal_ids[i]); fuentes_tr.append("desconocido_portal")
    for i, im_o in enumerate(univ_tr_imgs):
        X_train_list.append(im_o); y_train_list.append(cid_desc); ids_tr.append(univ_tr_ids[i]); fuentes_tr.append("desconocido_universidad")
        for aum in aumentar_muestra_entrenamiento(im_o, rng):
            X_train_list.append(aum); y_train_list.append(cid_desc); ids_tr.append(univ_tr_ids[i]); fuentes_tr.append("desconocido_universidad")

    X_train_img = np.asarray(X_train_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)
    pesos_muestras, resumen_pesos = calcular_pesos_identidad_y_clase(y_train, ids_tr, fuentes_tr)

    X_val_auth_img = np.stack([imagenes_autorizadas[m["id"]] for m in val_auth_proc])
    y_val_auth = np.array([CLASE_TO_ID[m["persona"]] for m in val_auth_proc], dtype=np.int64)
    X_lfw_val_arr = np.asarray(lfw_val_imgs, dtype=np.float32)
    X_univ_val_arr = np.asarray(univ_val_imgs, dtype=np.float32)
    X_val_img = np.vstack([X_val_auth_img, X_lfw_val_arr, X_univ_val_arr])
    y_val = np.concatenate([
        y_val_auth,
        np.full(len(X_lfw_val_arr), cid_desc, dtype=np.int64),
        np.full(len(X_univ_val_arr), cid_desc, dtype=np.int64),
    ])

    return {
        "X_train_img": X_train_img, "y_train": y_train, "pesos_muestras": pesos_muestras,
        "resumen_pesos": resumen_pesos, "X_val_img": X_val_img, "y_val": y_val,
        "val_auth_todos": val_auth_todos, "val_auth_proc_ids": [m["id"] for m in val_auth_proc],
        "imagenes_autorizadas": imagenes_autorizadas, "filas_lfw_val": filas_lfw_val,
        "estados_lfw_val": estados_lfw_val, "X_lfw_val_img": X_lfw_val_arr,
        "meta_univ_val": meta_univ_val, "X_univ_val_img": X_univ_val_arr,
    }


def ejecutar_experimento_antialiasing():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"exp_4clases_antialiasing_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de salida: {dir_salida}")

    # 1. Verificación sintética y lámina real
    verif_aa = verificar_antialiasing_sintetico_y_lamina(dir_salida)
    print("Verificación sintética antialiasing:", verif_aa)

    # 2. Preparar datos con supersampling_area
    t_prep0 = time.perf_counter()
    datos_aa = preparar_datos_con_remuestreo("supersampling_area")
    t_prep_aa = time.perf_counter() - t_prep0
    assert len(datos_aa["y_train"]) == 1221, f"Se esperaban exactamente 1221 muestras (como Portal Completo), obtenidas {len(datos_aa['y_train'])}"

    # 3. Entrenar únicamente CNN
    cnn_aa = ClasificadorCNN(n_clases=N_CLASES, semilla=SEMILLA)
    cnn_aa.entrenar_con_validacion(
        datos_aa["X_train_img"], datos_aa["y_train"], datos_aa["X_val_img"], datos_aa["y_val"],
        pesos_clases=None, pesos_muestras=datos_aa["pesos_muestras"], metrica_checkpoint="macro_recall"
    )
    ruta_mod_aa = dir_salida / "cnn" / "modelo.pt"
    ruta_mod_aa.parent.mkdir(parents=True, exist_ok=True)
    cnn_aa.guardar(ruta_mod_aa)

    # 4. Calibrar nueva variante exclusivamente en validación
    res_aa = evaluar_modelo_sobre_validacion("cnn_antialiasing", cnn_aa, datos_aa, es_cnn=True, umbral_fijo=None)
    th_aa = res_aa["umbral_recomendado"]

    # 5. Evaluar referencia Portal Completo (0.61) con su propio preprocesamiento ("estandar")
    datos_ref = preparar_datos_con_remuestreo("estandar")
    ruta_ref_cfg = RAIZ / "resultados/exp_4clases_portal_completo_20261120_200452/configuracion_candidato_portal_completo_propuesto.json"
    pipe_ref = PipelineCandidatoCongelado(ruta_ref_cfg, raiz_proyecto=RAIZ)
    res_ref = evaluar_modelo_sobre_validacion("ref_portal_completo_061", pipe_ref.clasificador, datos_ref, es_cnn=True, umbral_fijo=pipe_ref.umbral_score)

    # Medir latencia de preprocesamiento y pipeline completo en muestra representativa
    img_test = cv2.imread(str(RAIZ / "data/propio/imagenes/persona_1/sesion_04/frontal/persona_1_sesion_04_frontal_f01.jpg"))
    det_est = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True, metodo_remuestreo="estandar")
    det_aa = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True, metodo_remuestreo="supersampling_area")

    t_est_list, t_aa_list = [], []
    for _ in range(15):
        t0 = time.perf_counter()
        _ = det_est.detectar(img_test)
        t_est_list.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        _ = det_aa.detectar(img_test)
        t_aa_list.append((time.perf_counter() - t0) * 1000.0)

    # Guardar configuración propuesta (sin promover a web)
    if th_aa is not None:
        prep_cfg_aa = dict(pipe_ref.config["preprocesamiento"])
        prep_cfg_aa["metodo_remuestreo"] = "supersampling_area"
        prep_cfg_aa["descripcion_remuestreo"] = "Sobremuestreo k=clamp(round(1/escala), 2, 16) antes de cv2.resize(..., INTER_AREA) cuando escala < 0.9"
        cfg_prop = {
            "candidato": {
                "identificador": "cnn_4clases_antialiasing_propuesto",
                "nombre": "CNN 4 Clases (Portal Completo + Antialiasing Supersampling INTER_AREA)",
                "tipo_arquitectura": "CNN",
                "ruta_modelo": str(ruta_mod_aa.relative_to(RAIZ)).replace("\\", "/"),
                "sha256_modelo": calcular_sha256(ruta_mod_aa),
                "umbral_score": th_aa,
                "orden_clases": CLASES_NOMBRES,
                "mapeo_clases": {str(k): v for k, v in ID_TO_CLASE.items()},
                "personas_autorizadas": PERSONAS_AUTORIZADAS,
                "fecha_propuesta": datetime.now().isoformat(),
                "estado": "experimental_antialiasing_sin_promocion_web",
            },
            "detector_facial": pipe_ref.config["detector_facial"],
            "preprocesamiento": prep_cfg_aa,
            "regla_acceso": pipe_ref.config["regla_acceso"],
        }
        (dir_salida / "configuracion_candidato_antialiasing_propuesto.json").write_text(
            json.dumps(cfg_prop, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Guardar decisiones individuales y curva
    decs = []
    for r_m in (res_ref, res_aa):
        robj = r_m.get("regla_b") or r_m.get("regla_a")
        if robj and robj.get("decisiones_individuales"):
            decs.extend(robj["decisiones_individuales"])
    if decs:
        with open(dir_salida / "decisiones_individuales_validacion.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(decs[0].keys()))
            writer.writeheader()
            writer.writerows(decs)
    if res_aa.get("curva_umbrales"):
        (dir_salida / "curva_umbrales.json").write_text(json.dumps(res_aa["curva_umbrales"], indent=2, ensure_ascii=False), encoding="utf-8")

    def limpiar(r):
        rc = dict(r)
        for k in ("regla_a", "regla_b"):
            if rc.get(k):
                rc[k] = {kk: vv for kk, vv in rc[k].items() if kk != "decisiones_individuales"}
        rc.pop("curva_umbrales", None)
        return rc

    resumen = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "verificacion_sintetica_antialiasing": verif_aa,
        "latencia_deteccion_alineacion_ms": {
            "estandar": round(float(np.mean(t_est_list)), 2),
            "supersampling_area": round(float(np.mean(t_aa_list)), 2),
        },
        "referencia_portal_completo_061": limpiar(res_ref),
        "nueva_antialiasing": limpiar(res_aa),
    }
    (dir_salida / "resumen_experimento_antialiasing.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rb_r = res_ref["regla_b"]
    print("\n" + "=" * 75)
    print(" COMPARACIÓN EN VALIDACIÓN AUTOMÁTICA: ESTÁNDAR vs ANTIALIASING")
    print("=" * 75)
    print(f" Referencia (0.61): Correctas={rb_r['aperturas_correctas']}/64 | Conf={rb_r['confusiones_autorizados']} | Bloq={rb_r['bloqueos_puerta']} (FRR={rb_r['frr_puerta_pct']}%) | FAR={rb_r['far_total_pct']}% ({rb_r['far_conteo']})")
    if res_aa["regla_b_habilitada"]:
        rb_a = res_aa["regla_b"]
        print(f" Antialias  ({th_aa:.2f}): Correctas={rb_a['aperturas_correctas']}/64 | Conf={rb_a['confusiones_autorizados']} | Bloq={rb_a['bloqueos_puerta']} (FRR={rb_a['frr_puerta_pct']}%) | FAR={rb_a['far_total_pct']}% ({rb_a['far_conteo']})")
    else:
        print(" Antialias: sin_umbral_viable")
    print(f" Latencia media detección+alineación: Estandar={resumen['latencia_deteccion_alineacion_ms']['estandar']} ms vs Antialiasing={resumen['latencia_deteccion_alineacion_ms']['supersampling_area']} ms")
    print("=" * 75)


if __name__ == "__main__":
    ejecutar_experimento_antialiasing()
