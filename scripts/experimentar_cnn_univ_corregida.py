"""
Iteración controlada de la CNN de 4 clases incorporando las correcciones/aprobaciones
de rostros universitarios (data/universidad_desc/revision_universidad.json) únicamente
en entrenamiento (18 identidades), manteniendo intactos modelos anteriores, particiones
y el modelo activo de la web.

- Rescata las 3 imágenes universitarias de entrenamiento que fallaban por VARIOS_ROSTROS
  (universidad_014, universidad_021, universidad_028) usando la caja aprobada y el
  fallback documentado de preprocesar_rostro_alineado (puntos_faciales=None).
- Mantiene la alineación de 5 puntos cuando los landmarks de YuNet son válidos y están
  dentro de la caja aprobada.
- Evalúa y calibra exclusivamente sobre validación automática (Sesión 04 + 100 LFW + 7 Univ),
  con un diagnóstico separado claramente etiquetado como 'asistido' sobre las 7 Univ.
- No evalúa prueba_reservada (9 imágenes), Sesión 05 ni reserva LFW.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.adquisicion.propio import calcular_hash_archivo
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.alineador import preprocesar_rostro_alineado
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from scripts.entrenar_desarrollo import (
    BINS,
    CUADRICULA,
    RESOLUCION,
    SEMILLA,
    SIGMA,
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
    cargar_desconocidos_calibracion_alineados,
    cargar_desconocidos_entrenamiento_alineados,
    preparar_captura_alineada,
)
from scripts.experimentar_4clases_universidad import (
    MANIFIESTO_UNIVERSIDAD,
    cargar_manifiesto_universidad_desarrollo,
    evaluar_modelo_sobre_validacion,
    procesar_muestras_universidad,
)
from scripts.experimentar_4clases_portal_completo import (
    calcular_pesos_identidad_y_clase,
    cargar_todas_capturas_portal,
)

RUTA_REVISION_UNIV = RAIZ / "data" / "universidad_desc" / "revision_universidad.json"


def cargar_universidad_train_con_revisiones(
    filas_train: List[Dict[str, str]],
    detector: DetectorYuNet,
    extractor: ExtractorHistogramasRegionales,
    ruta_revision: Path = RUTA_REVISION_UNIV,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Procesa las 18 imágenes universitarias de entrenamiento incorporando las cajas aprobadas
    de revision_universidad.json cuando YuNet automático falla o difiere.
    No inventa landmarks: si YuNet no entregó 5 puntos válidos, usa puntos_faciales=None
    con la caja aprobada (fallback_recorte_seguro de preprocesar_rostro_alineado).
    """
    revisiones = json.loads(ruta_revision.read_text(encoding="utf-8")) if ruta_revision.exists() else {}

    imagenes_64 = []
    texturas_256 = []
    trazabilidad = []

    for fila in filas_train:
        ident = fila["identificador"]
        ruta_img = RAIZ / fila["ruta"]
        contenido = ruta_img.read_bytes()
        digest = hashlib.sha256(contenido).hexdigest()
        if digest != fila["hash_sha256"]:
            raise ValueError(f"SHA-256 modificado en {ident}")

        img_bgr = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"No se pudo decodificar {ruta_img}")

        rev_info = revisiones.get(ident, {})
        decision_rev = rev_info.get("decision", "pendiente")
        caja_rev = tuple(rev_info["caja"]) if rev_info.get("caja") else None

        if decision_rev == "excluida":
            trazabilidad.append({
                "id": ident, "procesada": False, "motivo": "excluida_por_revision",
                "estado_yunet": "NO_EVALUADA", "alineado": False, "metodo_prep": "ninguno"
            })
            continue

        det = detector.detectar(img_bgr)
        pts_validos = None
        caja_a_usar = None

        if det.estado == EstadoDeteccion.UN_ROSTRO and det.puntos_faciales is not None:
            # Si hay caja aprobada/corregida, verificar si los landmarks de YuNet caen dentro
            if caja_rev is not None and decision_rev in ("aprobada", "corregida"):
                bx, by, bw, bh = caja_rev
                pts_dentro = all(bx <= px <= bx + bw and by <= py <= by + bh for px, py in det.puntos_faciales)
                if pts_dentro:
                    pts_validos = det.puntos_faciales
                    caja_a_usar = caja_rev
                else:
                    # La caja corregida apunta a otro rostro distinto al de YuNet: no mezclar landmarks
                    pts_validos = None
                    caja_a_usar = caja_rev
            else:
                pts_validos = det.puntos_faciales
                caja_a_usar = det.recuadro
        elif caja_rev is not None and decision_rev in ("aprobada", "corregida"):
            # Rescate mediante caja aprobada sin inventar landmarks
            pts_validos = None
            caja_a_usar = caja_rev
        else:
            trazabilidad.append({
                "id": ident, "procesada": False, "motivo": f"sin_deteccion_ni_caja ({det.estado.value})",
                "estado_yunet": det.estado.value, "alineado": False, "metodo_prep": "ninguno"
            })
            continue

        img_proc, meta = preprocesar_rostro_alineado(
            imagen=img_bgr,
            puntos_faciales=pts_validos,
            recuadro=caja_a_usar,
            resolucion=RESOLUCION,
            sigma=SIGMA,
        )
        feat = extractor.extraer(img_proc).vector_caracteristicas

        imagenes_64.append(img_proc)
        texturas_256.append(feat)
        trazabilidad.append({
            "id": ident,
            "procesada": True,
            "estado_yunet": det.estado.value,
            "decision_revision": decision_rev,
            "caja_usada": list(caja_a_usar) if caja_a_usar else None,
            "alineado": meta.alineado,
            "metodo_prep": meta.metodo,
            "motivo_fallback": meta.motivo_fallback,
            "ruta": fila["ruta"],
            "sha256": digest,
        })

    return texturas_256, imagenes_64, trazabilidad


def ejecutar_iteracion_univ_corregida():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"exp_4clases_univ_corregida_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" ITERACIÓN CONTROLADA CNN: CORRECCIONES UNIVERSITARIAS EN ENTRENAMIENTO")
    print("=" * 75)
    print(f" Directorio de resultados: {dir_salida}")

    detector = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)

    # 1. Autorizados
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

    train_auth = [m for m in muestras_autorizadas if m["particion"] == "entrenamiento" and m["estado"] == "procesada"]
    val_auth_todos = [m for m in muestras_autorizadas if m["particion"] == "validacion" and m["sesion"] == "sesion_04"]
    val_auth_proc = [m for m in val_auth_todos if m["estado"] == "procesada"]

    # 2. Desconocidos LFW
    X_lfw_train_tex, X_lfw_train_img, meta_lfw_train = cargar_desconocidos_entrenamiento_alineados(detector, extractor)
    X_lfw_val_tex, X_lfw_val_img, filas_lfw_val, estados_lfw_val = cargar_desconocidos_calibracion_alineados(detector, extractor)

    # 3. Portal completo (72 válidas / 75)
    X_portal_tex, X_portal_img, meta_portal_validos, fallos_portal = cargar_todas_capturas_portal(detector, extractor)

    # 4. Universidad: train CON revisiones aprobadas (18/18), val SIN revisión (automática)
    filas_univ_train, filas_univ_val = cargar_manifiesto_universidad_desarrollo()
    X_univ_train_tex, X_univ_train_img, traz_univ_train = cargar_universidad_train_con_revisiones(
        filas_univ_train, detector, extractor
    )
    univ_train_proc = [t for t in traz_univ_train if t["procesada"]]
    print(f" Universidad train procesadas con revisión: {len(univ_train_proc)}/18")
    for t in univ_train_proc:
        if not t["alineado"]:
            print(f"   [RESCATADA] {t['id']}: estado_yunet={t['estado_yunet']} -> {t['metodo_prep']} ({t['motivo_fallback']})")

    # Validación automática (exactamente como en vivo)
    X_univ_val_tex, X_univ_val_img, meta_univ_val = procesar_muestras_universidad(
        filas_univ_val, detector, extractor, es_entrenamiento=False
    )

    # 5. Aumentos EXCLUSIVAMENTE en entrenamiento con semilla fija
    rng = np.random.RandomState(SEMILLA)
    X_train_img_list = []
    y_train_list = []
    identidades_train: List[str] = []
    fuentes_train: List[str] = []

    for m in train_auth:
        persona = m["persona"]
        cid = CLASE_TO_ID[persona]
        img_orig = imagenes_autorizadas[m["id"]]
        X_train_img_list.append(img_orig)
        y_train_list.append(cid)
        identidades_train.append(persona)
        fuentes_train.append(f"autorizado_{persona}")
        for aum in aumentar_muestra_entrenamiento(img_orig, rng):
            X_train_img_list.append(aum)
            y_train_list.append(cid)
            identidades_train.append(persona)
            fuentes_train.append(f"autorizado_{persona}")

    cid_desc = CLASE_TO_ID["desconocido"]
    for i, meta in enumerate(meta_lfw_train):
        X_train_img_list.append(X_lfw_train_img[i])
        y_train_list.append(cid_desc)
        identidades_train.append(meta["id"])
        fuentes_train.append("desconocido_lfw")

    for i, meta in enumerate(meta_portal_validos):
        X_train_img_list.append(X_portal_img[i])
        y_train_list.append(cid_desc)
        identidades_train.append(f"portal_{meta['identidad']}")
        fuentes_train.append("desconocido_portal")

    for i, meta in enumerate(univ_train_proc):
        img_orig = X_univ_train_img[i]
        X_train_img_list.append(img_orig)
        y_train_list.append(cid_desc)
        identidades_train.append(meta["id"])
        fuentes_train.append("desconocido_universidad")
        for aum in aumentar_muestra_entrenamiento(img_orig, rng):
            X_train_img_list.append(aum)
            y_train_list.append(cid_desc)
            identidades_train.append(meta["id"])
            fuentes_train.append("desconocido_universidad")

    X_train_img = np.asarray(X_train_img_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)

    pesos_muestras, resumen_pesos = calcular_pesos_identidad_y_clase(y_train, identidades_train, fuentes_train)
    print(f" Total Train: {len(y_train)} muestras (antes 1221 -> +{len(y_train) - 1221} muestras por las 3 identidades rescatadas × 3)")

    # Validación automática
    X_val_auth_img = np.stack([imagenes_autorizadas[m["id"]] for m in val_auth_proc])
    y_val_auth = np.array([CLASE_TO_ID[m["persona"]] for m in val_auth_proc], dtype=np.int64)
    X_val_univ_img_arr = np.asarray(X_univ_val_img, dtype=np.float32)
    X_val_img = np.vstack([X_val_auth_img, X_lfw_val_img, X_val_univ_img_arr])
    y_val = np.concatenate([
        y_val_auth,
        np.full(len(X_lfw_val_img), cid_desc, dtype=np.int64),
        np.full(len(X_val_univ_img_arr), cid_desc, dtype=np.int64),
    ])

    # 6. Entrenar únicamente CNN con la misma arquitectura, hiperparámetros y semilla
    cnn = ClasificadorCNN(n_clases=N_CLASES, semilla=SEMILLA)
    cnn.entrenar_con_validacion(
        X_train_img, y_train, X_val_img, y_val,
        pesos_clases=None, pesos_muestras=pesos_muestras, metrica_checkpoint="macro_recall"
    )
    macro_rec = round(calcular_macro_recall(y_val, cnn.predecir(X_val_img)), 2)
    print(f" [CNN Nueva] mejor_epoca={cnn.mejor_epoca_}, macro_recall_val={macro_rec}%")

    dir_cnn = dir_salida / "cnn"
    dir_cnn.mkdir(parents=True, exist_ok=True)
    ruta_mod = dir_cnn / "modelo.pt"
    cnn.guardar(ruta_mod)

    # 7. Calibración automática en Validación y comparación con Referencia (0.61)
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

    res_nueva = evaluar_modelo_sobre_validacion("cnn_univ_corregida", cnn, datos_eval, es_cnn=True, umbral_fijo=None)

    ruta_ref_cfg = RAIZ / "resultados" / "exp_4clases_portal_completo_20261120_200452" / "configuracion_candidato_portal_completo_propuesto.json"
    pipe_ref = PipelineCandidatoCongelado(ruta_ref_cfg, raiz_proyecto=RAIZ)
    res_ref = evaluar_modelo_sobre_validacion("ref_portal_completo_061", pipe_ref.clasificador, datos_eval, es_cnn=True, umbral_fijo=pipe_ref.umbral_score)

    # 8. Diagnóstico separado ASISTIDO sobre las 7 imágenes universitarias de validación (con cajas aprobadas)
    _, X_univ_val_asist_img, traz_univ_val_asist = cargar_universidad_train_con_revisiones(
        filas_univ_val, detector, extractor
    )
    X_asist_arr = np.asarray(X_univ_val_asist_img, dtype=np.float32)

    def eval_asistido(modelo_c, th: float) -> Dict[str, Any]:
        probs = modelo_c.predecir_probabilidades(X_asist_arr)
        fa = 0
        desc_c = 0
        umb_c = 0
        for p_vec in probs:
            idx = int(np.argmax(p_vec))
            nom = ID_TO_CLASE[idx]
            sc = float(p_vec[idx])
            if nom in PERSONAS_AUTORIZADAS and sc >= th:
                fa += 1
            elif nom == "desconocido":
                desc_c += 1
            else:
                umb_c += 1
        return {"intentos": len(probs), "fa": fa, "far_pct": round(100.0 * fa / len(probs), 2), "desconocido": desc_c, "rechazo_umbral": umb_c}

    th_nuevo = res_nueva["umbral_recomendado"]
    diag_asistido = {
        "nota": "DIAGNÓSTICO ASISTIDO (con cajas aprobadas de revisión, NO disponible en inferencia automática en vivo).",
        "referencia_061": eval_asistido(pipe_ref.clasificador, 0.61),
        "nueva_cnn": eval_asistido(cnn, th_nuevo) if th_nuevo is not None else None,
    }

    # Guardar configuración propuesta si es viable
    if th_nuevo is not None:
        cfg_prop = {
            "candidato": {
                "identificador": "cnn_4clases_univ_corregida_propuesto",
                "nombre": "CNN 4 Clases (Portal Completo + Correcciones Univ Train)",
                "tipo_arquitectura": "CNN",
                "ruta_modelo": str(ruta_mod.relative_to(RAIZ)).replace("\\", "/"),
                "sha256_modelo": calcular_sha256(ruta_mod),
                "umbral_score": th_nuevo,
                "orden_clases": CLASES_NOMBRES,
                "mapeo_clases": {str(k): v for k, v in ID_TO_CLASE.items()},
                "personas_autorizadas": PERSONAS_AUTORIZADAS,
                "fecha_propuesta": datetime.now().isoformat(),
                "estado": "propuesto_sin_promocion_web",
            },
            "detector_facial": pipe_ref.config["detector_facial"],
            "preprocesamiento": pipe_ref.config["preprocesamiento"],
            "regla_acceso": pipe_ref.config["regla_acceso"],
        }
        (dir_salida / "configuracion_candidato_univ_corregida.json").write_text(
            json.dumps(cfg_prop, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Guardar decisiones individuales y curva de umbrales antes de limpiar el resumen
    decs_todas = []
    for r_mod in (res_ref, res_nueva):
        regla_obj = r_mod.get("regla_b") or r_mod.get("regla_a")
        if regla_obj and regla_obj.get("decisiones_individuales"):
            decs_todas.extend(regla_obj["decisiones_individuales"])
    if decs_todas:
        with open(dir_salida / "decisiones_individuales_validacion.csv", "w", newline="", encoding="utf-8") as f_dec:
            writer = csv.DictWriter(f_dec, fieldnames=list(decs_todas[0].keys()))
            writer.writeheader()
            writer.writerows(decs_todas)

    if res_nueva.get("curva_umbrales"):
        (dir_salida / "curva_umbrales.json").write_text(
            json.dumps(res_nueva["curva_umbrales"], indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def limpiar_res(r):
        rc = dict(r)
        for k in ("regla_a", "regla_b"):
            if rc.get(k):
                rc[k] = {kk: vv for kk, vv in rc[k].items() if kk != "decisiones_individuales"}
        rc.pop("curva_umbrales", None)
        return rc

    resumen = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "trazabilidad_universidad_train": traz_univ_train,
        "trazabilidad_universidad_val_asistida": traz_univ_val_asist,
        "resumen_pesos": resumen_pesos,
        "cnn_metadatos": {"mejor_epoca": cnn.mejor_epoca_, "macro_recall_val": macro_rec},
        "validacion_automatica_referencia_061": limpiar_res(res_ref),
        "validacion_automatica_nueva": limpiar_res(res_nueva),
        "diagnostico_asistido_univ_val_7": diag_asistido,
    }
    (dir_salida / "resumen_iteracion_univ_corregida.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print(" COMPARACIÓN EN VALIDACIÓN AUTOMÁTICA (64 AUTH + 100 LFW + 7 UNIV)")
    print("=" * 75)
    rb_ref = res_ref["regla_b"]
    print(f" Referencia (0.61): FRR={rb_ref['frr_puerta_pct']}% ({rb_ref['frr_conteo']}) | FAR_global={rb_ref['far_total_pct']}% ({rb_ref['far_conteo']}) | LFW={rb_ref['far_lfw']['far_pct']}% | Univ={rb_ref['far_universidad']['far_pct']}%")
    if res_nueva["regla_b_habilitada"]:
        rb_n = res_nueva["regla_b"]
        print(f" Nueva CNN  ({th_nuevo:.2f}): FRR={rb_n['frr_puerta_pct']}% ({rb_n['frr_conteo']}) | FAR_global={rb_n['far_total_pct']}% ({rb_n['far_conteo']}) | LFW={rb_n['far_lfw']['far_pct']}% | Univ={rb_n['far_universidad']['far_pct']}%")
    else:
        print(" Nueva CNN: sin_umbral_viable (Regla B deshabilitada)")
    print("-" * 75)
    print(" DIAGNÓSTICO ASISTIDO (7 UNIV CON CAJAS APROBADAS, NO EN VIVO):")
    print(f"   Referencia (0.61): {diag_asistido['referencia_061']}")
    print(f"   Nueva CNN        : {diag_asistido['nueva_cnn']}")
    print("=" * 75)
    return resumen, dir_salida


if __name__ == "__main__":
    ejecutar_iteracion_univ_corregida()
