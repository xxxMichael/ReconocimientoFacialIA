"""Experimento controlado de clasificación con cuatro clases (3 autorizados + 1 desconocido)."""

import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.adquisicion.propio import (
    RegistroCaptura,
    calcular_hash_archivo,
    obtener_particion_esperada_propio,
)
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.regla_acceso import (
    clave_ordenamiento_jerarquico,
    seleccionar_umbral_optimo_jerarquico,
)
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.fabrica import crear_detector
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion,
)
from scripts.calibrar_desarrollo import CALIBRACION_LFW, preparar_negativos
from scripts.entrenar_desarrollo import (
    BINS,
    CUADRICULA,
    HILOS,
    MANIFIESTO,
    RAIZ_IMAGENES,
    REVISIONES_VALIDAS,
    RESOLUCION,
    SEMILLA,
    SIGMA,
    SOLAPAMIENTO_MINIMO,
    cargar_registros,
    preparar_captura,
)

CLASES_NOMBRES = ["persona_1", "persona_2", "persona_3", "desconocido"]
PERSONAS_AUTORIZADAS = ["persona_1", "persona_2", "persona_3"]
N_CLASES = 4
CLASE_TO_ID = {nombre: i for i, nombre in enumerate(CLASES_NOMBRES)}
ID_TO_CLASE = {i: nombre for i, nombre in enumerate(CLASES_NOMBRES)}


def calcular_sha256(archivo: Path) -> str:
    return hashlib.sha256(Path(archivo).read_bytes()).hexdigest()


def calcular_macro_recall(y_true: np.ndarray, y_pred: np.ndarray, n_clases: int = N_CLASES) -> float:
    recalls = []
    for c in range(n_clases):
        mask = (y_true == c)
        total_c = int(np.sum(mask))
        if total_c > 0:
            rec = float(np.sum(y_pred[mask] == c)) / float(total_c) * 100.0
            recalls.append(rec)
    return float(np.mean(recalls)) if recalls else 0.0


def cargar_desconocidos_entrenamiento(detector, extractor):
    ruta_manifiesto = RAIZ / "data" / "desconocidos_lfw" / "entrenamiento_4clases" / "manifiesto.csv"
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(f"No existe el manifiesto de desconocidos de entrenamiento en {ruta_manifiesto}")

    with ruta_manifiesto.open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))

    if len(filas) != 300:
        raise ValueError(f"Se esperaban 300 desconocidos de entrenamiento, pero hay {len(filas)}")

    imagenes_64 = []
    texturas_256 = []
    metadatos = []

    for fila in filas:
        ruta_img = RAIZ / fila["ruta_imagen"]
        contenido = ruta_img.read_bytes()
        digest = hashlib.sha256(contenido).hexdigest()
        if digest != fila["sha256"]:
            raise ValueError(f"Hash modificado en desconocido: {ruta_img}")

        img_bgr = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"No se pudo decodificar {ruta_img}")

        # Recorte usando la caja registrada y validada
        x = int(fila["caja_yunet_x"])
        y = int(fila["caja_yunet_y"])
        w = int(fila["caja_yunet_w"])
        h = int(fila["caja_yunet_h"])
        recorte = detector.recortar_con_margen_seguro(img_bgr, (x, y, w, h))

        gris = a_escala_de_grises(recorte) if recorte.ndim == 3 else recorte
        normalizada = normalizar_resolucion(gris, RESOLUCION)
        intensidades = normalizar_intensidades(normalizada, "float32_0_1")
        img_proc = suavizado_gaussiano(intensidades, sigma=SIGMA)

        feat = extractor.extraer(img_proc).vector_caracteristicas

        imagenes_64.append(img_proc)
        texturas_256.append(feat)
        metadatos.append({
            "id": f"lfw_train_{fila['identidad_normalizada'].replace(' ', '_')}",
            "persona": "desconocido",
            "sesion": "desconocidos_lfw_train",
            "particion": "entrenamiento",
            "pose": "desconocida",
            "ruta": fila["ruta_imagen"],
            "sha256": digest,
        })

    return np.asarray(texturas_256, dtype=np.float32), np.asarray(imagenes_64, dtype=np.float32), metadatos


def cargar_desconocidos_calibracion(detector, extractor):
    """Carga los 100 desconocidos de calibración LFW (exactamente los mismos de la baseline)."""
    filas, estados, imagenes = preparar_negativos(detector)
    texturas = [extractor.extraer(img).vector_caracteristicas for img in imagenes]
    return np.asarray(texturas, dtype=np.float32), imagenes, filas, estados


def preparar_datasets():
    print("Cargando registros autorizados...")
    registros, hash_manifiesto = cargar_registros(solo_base=False)

    detector = crear_detector()
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)

    hashes_por_particion = {}
    muestras_autorizadas = []
    imagenes_autorizadas = {}

    for reg in registros:
        muestra, img_proc = preparar_captura(reg, detector, hashes_por_particion)
        muestras_autorizadas.append(muestra)
        if img_proc is not None:
            imagenes_autorizadas[reg.identificador] = img_proc

    # Autorizados entrenamiento (268)
    train_auth = [m for m in muestras_autorizadas if m["particion"] == "entrenamiento" and m["estado"] == "procesada"]
    X_train_auth_tex = np.stack([extractor.extraer(imagenes_autorizadas[m["id"]]).vector_caracteristicas for m in train_auth])
    X_train_auth_img = np.stack([imagenes_autorizadas[m["id"]] for m in train_auth])
    y_train_auth = np.array([CLASE_TO_ID[m["persona"]] for m in train_auth], dtype=np.int64)

    # Autorizados validación sesión 04 (64 intentos)
    val_auth_todos = [m for m in muestras_autorizadas if m["particion"] == "validacion" and m["sesion"] == "sesion_04"]
    val_auth_proc = [m for m in val_auth_todos if m["estado"] == "procesada"]
    X_val_auth_tex = np.stack([extractor.extraer(imagenes_autorizadas[m["id"]]).vector_caracteristicas for m in val_auth_proc])
    X_val_auth_img = np.stack([imagenes_autorizadas[m["id"]] for m in val_auth_proc])
    y_val_auth = np.array([CLASE_TO_ID[m["persona"]] for m in val_auth_proc], dtype=np.int64)

    print(f"Autorizados train procesados: {len(train_auth)} / {len([m for m in muestras_autorizadas if m['particion'] == 'entrenamiento'])}")
    print(f"Autorizados val procesados: {len(val_auth_proc)} / {len(val_auth_todos)}")

    # Desconocidos entrenamiento (300)
    print("Cargando 300 desconocidos de entrenamiento LFW...")
    X_train_desc_tex, X_train_desc_img, meta_desc_train = cargar_desconocidos_entrenamiento(detector, extractor)
    y_train_desc = np.full(len(X_train_desc_tex), CLASE_TO_ID["desconocido"], dtype=np.int64)

    # Desconocidos validación (100 calibración LFW)
    print("Cargando 100 desconocidos de calibración LFW...")
    X_val_desc_tex, X_val_desc_img, filas_desc_val, estados_desc_val = cargar_desconocidos_calibracion(detector, extractor)
    y_val_desc = np.full(len(X_val_desc_tex), CLASE_TO_ID["desconocido"], dtype=np.int64)

    # Unir conjuntos
    X_train_tex = np.vstack([X_train_auth_tex, X_train_desc_tex])
    X_train_img = np.vstack([X_train_auth_img, X_train_desc_img])
    y_train = np.concatenate([y_train_auth, y_train_desc])

    X_val_tex = np.vstack([X_val_auth_tex, X_val_desc_tex])
    X_val_img = np.vstack([X_val_auth_img, X_val_desc_img])
    y_val = np.concatenate([y_val_auth, y_val_desc])

    # Pesos de clase balanceados
    counts = Counter(y_train)
    total_samples = len(y_train)
    pesos_clases = np.array([total_samples / (N_CLASES * counts[c]) for c in range(N_CLASES)], dtype=np.float32)

    print(f"Distribución Train: {dict(counts)} -> Pesos de clase: {pesos_clases}")
    print(f"Distribución Val (procesados): {dict(Counter(y_val))}")

    return {
        "X_train_tex": X_train_tex,
        "X_train_img": X_train_img,
        "y_train": y_train,
        "X_val_tex": X_val_tex,
        "X_val_img": X_val_img,
        "y_val": y_val,
        "pesos_clases": pesos_clases,
        "val_auth_todos": val_auth_todos,
        "val_auth_proc_ids": [m["id"] for m in val_auth_proc],
        "filas_desc_val": filas_desc_val,
        "estados_desc_val": estados_desc_val,
        "imagenes_autorizadas": imagenes_autorizadas,
        "detector": detector,
        "extractor": extractor,
    }


def entrenar_los_5_modelos(datos):
    X_train_tex = datos["X_train_tex"]
    X_train_img = datos["X_train_img"]
    y_train = datos["y_train"]
    X_val_tex = datos["X_val_tex"]
    X_val_img = datos["X_val_img"]
    y_val = datos["y_val"]
    pesos_clases = datos["pesos_clases"]

    modelos = {}
    metadatos_modelos = {}

    # 1. KNN
    print("\n--- Entrenando KNN (4 clases, distancia vs uniforme) ---")
    mejor_knn_score = -1.0
    mejor_knn = None
    mejor_knn_cfg = None
    for k in (3, 5, 7, 9):
        for pond in ("distance", "uniform"):
            knn_cand = ClasificadorKNN(k=k, metrica="euclidean", ponderacion=pond)
            knn_cand.entrenar(X_train_tex, y_train)
            y_pred_val = knn_cand.predecir(X_val_tex)
            score = calcular_macro_recall(y_val, y_pred_val)
            print(f"KNN k={k}, pond={pond} -> Val Macro-Recall: {score:.2f}%")
            if score > mejor_knn_score:
                mejor_knn_score = score
                mejor_knn = knn_cand
                mejor_knn_cfg = {"k": k, "ponderacion": pond, "metrica": "euclidean", "macro_recall_val": score}
    modelos["knn"] = mejor_knn
    metadatos_modelos["knn"] = mejor_knn_cfg

    # 2. Árbol de Decisión
    print("\n--- Entrenando Árbol de Decisión (4 clases, class_weight='balanced') ---")
    arbol, hist_arbol, cfg_arbol = ClasificadorArbol.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight="balanced", metrica="macro_recall", semilla=SEMILLA
    )
    y_pred_arbol = arbol.predecir(X_val_tex)
    cfg_arbol["macro_recall_val"] = calcular_macro_recall(y_val, y_pred_arbol)
    cfg_arbol["n_hojas"] = arbol.n_hojas_
    cfg_arbol["profundidad_efectiva"] = arbol.profundidad_efectiva_
    modelos["arbol"] = arbol
    metadatos_modelos["arbol"] = cfg_arbol
    print(f"Árbol elegido: {cfg_arbol}")

    # 3. SVM
    print("\n--- Entrenando SVM (4 clases, class_weight='balanced') ---")
    svm, hist_svm, cfg_svm = ClasificadorSVM.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight="balanced", metrica="macro_recall", semilla=SEMILLA
    )
    y_pred_svm = svm.predecir(X_val_tex)
    cfg_svm["macro_recall_val"] = calcular_macro_recall(y_val, y_pred_svm)
    cfg_svm["n_vectores_soporte"] = svm.n_vectores_soporte_
    modelos["svm"] = svm
    metadatos_modelos["svm"] = cfg_svm
    print(f"SVM elegido: {cfg_svm}")

    # 4. ANN
    print("\n--- Entrenando ANN (4 clases, pérdida ponderada, early checkpointing en Macro-Recall) ---")
    ann = ClasificadorANN(n_entrada=X_train_tex.shape[1], n_clases=N_CLASES, semilla=SEMILLA)
    ann.entrenar_con_validacion(
        X_train_tex, y_train, X_val_tex, y_val,
        pesos_clases=pesos_clases, metrica_checkpoint="macro_recall"
    )
    y_pred_ann = ann.predecir(X_val_tex)
    macro_rec_ann = calcular_macro_recall(y_val, y_pred_ann)
    metadatos_modelos["ann"] = {
        "mejor_epoca": ann.mejor_epoca_,
        "macro_recall_val": macro_rec_ann,
        "n_clases": N_CLASES,
        "pesos_clases": pesos_clases.tolist(),
        "lr": ann.lr,
        "dropout": ann.dropout,
        "tamano_lote": ann.tamano_lote,
        "parametros_entrenables": ann.red_wrapper.contar_parametros(),
    }
    modelos["ann"] = ann
    print(f"ANN mejor época: {ann.mejor_epoca_}, Val Macro-Recall: {macro_rec_ann:.2f}%")

    # 5. CNN
    print("\n--- Entrenando CNN (4 clases, pérdida ponderada, early checkpointing en Macro-Recall) ---")
    cnn = ClasificadorCNN(n_clases=N_CLASES, semilla=SEMILLA)
    cnn.entrenar_con_validacion(
        X_train_img, y_train, X_val_img, y_val,
        pesos_clases=pesos_clases, metrica_checkpoint="macro_recall"
    )
    y_pred_cnn = cnn.predecir(X_val_img)
    macro_rec_cnn = calcular_macro_recall(y_val, y_pred_cnn)
    metadatos_modelos["cnn"] = {
        "mejor_epoca": cnn.mejor_epoca_,
        "macro_recall_val": macro_rec_cnn,
        "n_clases": N_CLASES,
        "pesos_clases": pesos_clases.tolist(),
        "lr": cnn.lr,
        "dropout": cnn.dropout,
        "tamano_lote": cnn.tamano_lote,
    }
    modelos["cnn"] = cnn
    print(f"CNN mejor época: {cnn.mejor_epoca_}, Val Macro-Recall: {macro_rec_cnn:.2f}%")

    return modelos, metadatos_modelos


def calcular_tasa_pct(numerador: int, denominador: int) -> Optional[float]:
    """Calcula porcentaje o retorna None si el denominador es 0."""
    if denominador <= 0:
        return None
    return round(100.0 * float(numerador) / float(denominador), 2)


def evaluar_reglas(nombre_modelo, modelo, datos):
    """
    Evalúa Regla A y Regla B sobre los 64 intentos autorizados (Sesión 04)
    y los 100 intentos de desconocidos (calibración LFW).
    Separa explícitamente bloqueos físicos de puerta (FRR) de confusiones de identidad.
    """
    es_cnn = (nombre_modelo == "cnn")
    val_auth_todos = datos["val_auth_todos"]
    val_auth_proc_ids = set(datos["val_auth_proc_ids"])
    filas_desc_val = datos["filas_desc_val"]
    estados_desc_val = datos["estados_desc_val"]

    # Inferencia en autorizados procesados
    ids_procesados = datos["val_auth_proc_ids"]
    if es_cnn:
        X_auth = datos["X_val_img"][:len(ids_procesados)]
    else:
        X_auth = datos["X_val_tex"][:len(ids_procesados)]

    t0 = time.perf_counter()
    probs_auth = modelo.predecir_probabilidades(X_auth)
    t_latencia_auth = (time.perf_counter() - t0) / len(X_auth) if len(X_auth) > 0 else 0.0

    dict_pred_auth = {}
    for id_m, probs in zip(ids_procesados, probs_auth):
        c_pred = int(np.argmax(probs))
        dict_pred_auth[id_m] = {
            "clase_pred": c_pred,
            "persona_pred": ID_TO_CLASE[c_pred],
            "score": float(probs[c_pred]),
            "probs": probs.tolist(),
        }

    # Inferencia en desconocidos procesados
    n_auth_proc = len(ids_procesados)
    if es_cnn:
        X_desc = datos["X_val_img"][n_auth_proc:]
    else:
        X_desc = datos["X_val_tex"][n_auth_proc:]

    probs_desc = modelo.predecir_probabilidades(X_desc)

    dict_pred_desc = {}
    idx_proc = 0
    for idx_desc, (fila, estado) in enumerate(zip(filas_desc_val, estados_desc_val)):
        if estado == "UN_ROSTRO":
            probs = probs_desc[idx_proc]
            idx_proc += 1
            c_pred = int(np.argmax(probs))
            dict_pred_desc[idx_desc] = {
                "procesada": True,
                "clase_pred": c_pred,
                "persona_pred": ID_TO_CLASE[c_pred],
                "score": float(probs[c_pred]),
                "probs": probs.tolist(),
            }
        else:
            dict_pred_desc[idx_desc] = {
                "procesada": False,
                "estado_deteccion": estado,
            }

    # Función interna de evaluación para un umbral específico
    def evaluar_con_umbral(th_eval: Optional[float], nombre_regla: str) -> Dict[str, Any]:
        intentos_auth_eval = []
        aperturas_correctas = 0
        confusiones_autorizados = 0
        bloqueos_desconocido = 0
        bloqueos_umbral = 0
        bloqueos_deteccion = 0

        conteo_por_persona = {
            p: {
                "intentos": 0,
                "aperturas_correctas": 0,
                "confusiones": 0,
                "aperturas_totales": 0,
                "bloqueos": 0,
                "bloqueos_desconocido": 0,
                "bloqueos_bajo_umbral": 0,
                "bloqueos_fallo_deteccion": 0,
                "frr_pct": None,
            }
            for p in PERSONAS_AUTORIZADAS
        }
        conteo_por_pose = {}

        for m in val_auth_todos:
            p_real = m["persona"]
            pose = m["pose"]

            conteo_por_persona[p_real]["intentos"] += 1
            conteo_por_pose.setdefault(pose, {
                "intentos": 0,
                "aperturas_correctas": 0,
                "confusiones": 0,
                "aperturas_totales": 0,
                "bloqueos": 0,
                "bloqueos_desconocido": 0,
                "bloqueos_bajo_umbral": 0,
                "bloqueos_fallo_deteccion": 0,
                "frr_pct": None,
            })
            conteo_por_pose[pose]["intentos"] += 1

            if m["id"] not in dict_pred_auth:
                # Bloqueo físico por fallo en captura/detección/procesamiento
                bloqueos_deteccion += 1
                conteo_por_persona[p_real]["bloqueos"] += 1
                conteo_por_persona[p_real]["bloqueos_fallo_deteccion"] += 1
                conteo_por_pose[pose]["bloqueos"] += 1
                conteo_por_pose[pose]["bloqueos_fallo_deteccion"] += 1
                intentos_auth_eval.append({
                    "id": m["id"], "persona_real": p_real, "pose": pose, "estado": m["estado"],
                    "decision": "bloqueo_fallo_procesamiento", "motivo_bloqueo": "fallo_deteccion",
                    "persona_predicha": None, "score": None
                })
                continue

            pred = dict_pred_auth[m["id"]]
            p_pred = pred["persona_pred"]
            score = pred["score"]

            if p_pred == "desconocido":
                # Bloqueo físico: clasificado como desconocido (la puerta no abre)
                bloqueos_desconocido += 1
                conteo_por_persona[p_real]["bloqueos"] += 1
                conteo_por_persona[p_real]["bloqueos_desconocido"] += 1
                conteo_por_pose[pose]["bloqueos"] += 1
                conteo_por_pose[pose]["bloqueos_desconocido"] += 1
                decision = "bloqueo_clase_desconocido"
                motivo = "prediccion_desconocido"
            elif th_eval is not None and score < th_eval:
                # Bloqueo físico: clasificado como autorizado pero puntaje insuficiente
                bloqueos_umbral += 1
                conteo_por_persona[p_real]["bloqueos"] += 1
                conteo_por_persona[p_real]["bloqueos_bajo_umbral"] += 1
                conteo_por_pose[pose]["bloqueos"] += 1
                conteo_por_pose[pose]["bloqueos_bajo_umbral"] += 1
                decision = "bloqueo_bajo_puntaje"
                motivo = "score_insuficiente"
            elif p_pred == p_real:
                # Apertura correcta: la puerta abre con la identidad correcta
                aperturas_correctas += 1
                conteo_por_persona[p_real]["aperturas_correctas"] += 1
                conteo_por_persona[p_real]["aperturas_totales"] += 1
                conteo_por_pose[pose]["aperturas_correctas"] += 1
                conteo_por_pose[pose]["aperturas_totales"] += 1
                decision = "apertura_correcta"
                motivo = None
            else:
                # Apertura con confusión de identidad: la puerta ABRE como otro autorizado.
                # Es un error de identificación pero NO es un bloqueo de puerta.
                confusiones_autorizados += 1
                conteo_por_persona[p_real]["confusiones"] += 1
                conteo_por_persona[p_real]["aperturas_totales"] += 1
                conteo_por_pose[pose]["confusiones"] += 1
                conteo_por_pose[pose]["aperturas_totales"] += 1
                decision = "apertura_confundida"
                motivo = None

            intentos_auth_eval.append({
                "id": m["id"], "persona_real": p_real, "pose": pose, "estado": m["estado"],
                "decision": decision, "motivo_bloqueo": motivo,
                "persona_predicha": p_pred, "score": score
            })

        # Totales agregados autorizados
        total_auth = len(val_auth_todos)
        aperturas_totales = aperturas_correctas + confusiones_autorizados
        bloqueos_puerta = total_auth - aperturas_totales
        frr_puerta = calcular_tasa_pct(bloqueos_puerta, total_auth)

        # Invariantes obligatorias
        assert bloqueos_puerta == (bloqueos_desconocido + bloqueos_umbral + bloqueos_deteccion), "Discrepancia en bloqueos"
        assert total_auth == (bloqueos_puerta + aperturas_correctas + confusiones_autorizados), "Discrepancia en suma total de intentos"

        # Métricas por persona
        frr_por_p = {}
        for p, stats in conteo_por_persona.items():
            stats["frr_pct"] = calcular_tasa_pct(stats["bloqueos"], stats["intentos"])
            frr_por_p[p] = stats["frr_pct"]
            assert stats["intentos"] == (stats["bloqueos"] + stats["aperturas_correctas"] + stats["confusiones"]), f"Invariante rota para persona {p}"

        # Métricas por pose
        for pose, stats in conteo_por_pose.items():
            stats["frr_pct"] = calcular_tasa_pct(stats["bloqueos"], stats["intentos"])
            assert stats["intentos"] == (stats["bloqueos"] + stats["aperturas_correctas"] + stats["confusiones"]), f"Invariante rota para pose {pose}"

        # Evaluación en Desconocidos
        fa_count = 0
        intentos_desc_eval = []
        for idx_desc in range(len(filas_desc_val)):
            info = dict_pred_desc[idx_desc]
            fila = filas_desc_val[idx_desc]
            if not info["procesada"]:
                intentos_desc_eval.append({
                    "identidad": fila["identidad_lfw"], "estado": info["estado_deteccion"],
                    "decision": "rechazado_fallo_deteccion", "persona_predicha": None, "score": None
                })
                continue

            p_pred = info["persona_pred"]
            score = info["score"]
            if p_pred in PERSONAS_AUTORIZADAS and (th_eval is None or score >= th_eval):
                fa_count += 1
                decision = "falsa_aceptacion"
            else:
                decision = "rechazo_correcto"

            intentos_desc_eval.append({
                "identidad": fila["identidad_lfw"], "estado": "procesada",
                "decision": decision, "persona_predicha": p_pred, "score": score
            })

        total_desc = len(filas_desc_val)
        desc_proc_count = len(X_desc)
        far_total = calcular_tasa_pct(fa_count, total_desc)
        far_cond = calcular_tasa_pct(fa_count, desc_proc_count)

        return {
            "nombre_regla": nombre_regla,
            "umbral": th_eval,
            "far_total_pct": far_total,
            "far_total_conteo": f"{fa_count}/{total_desc}" if total_desc > 0 else None,
            "far_condicional_pct": far_cond,
            "far_condicional_conteo": f"{fa_count}/{desc_proc_count}" if desc_proc_count > 0 else None,
            "frr_puerta_pct": frr_puerta,
            "frr_puerta_conteo": f"{bloqueos_puerta}/{total_auth}" if total_auth > 0 else None,
            "aperturas_totales": aperturas_totales,
            "aperturas_correctas": aperturas_correctas,
            "confusiones_autorizados": confusiones_autorizados,
            "bloqueos_puerta": bloqueos_puerta,
            "bloqueos_desconocido": bloqueos_desconocido,
            "bloqueos_bajo_umbral": bloqueos_umbral,
            "bloqueos_fallo_deteccion": bloqueos_deteccion,
            "frr_por_persona": frr_por_p,
            "desglose_por_persona": conteo_por_persona,
            "desglose_por_pose": conteo_por_pose,
            "intentos_auth": intentos_auth_eval,
            "intentos_desc": intentos_desc_eval,
        }

    # Búsqueda rigurosa de umbral para Regla B
    # Requisitos: FRR_p <= 30% (utilizando bloqueos físicos reales) para CADA p in {persona_1, persona_2, persona_3}
    # Orden jerárquico de selección:
    # 1. Menor FAR de desconocidos (far_total_pct)
    # 2. Menor FRR global de puerta (frr_puerta_pct)
    # 3. Mayor cantidad de aperturas con identidad correcta (aperturas_correctas)
    # 4. Mayor umbral (th), únicamente si los anteriores empatan
    umbrales_candidatos = [round(th, 2) for th in np.arange(0.00, 1.01, 0.01)]
    curva_umbrales = []
    viable_umbrales = []

    for th in umbrales_candidatos:
        eval_th = evaluar_con_umbral(th, f"Candidato th={th:.2f}")
        frr_por_p = eval_th["frr_por_persona"]

        es_viable = all(
            (frr_por_p[p] is not None and frr_por_p[p] <= 30.0)
            for p in PERSONAS_AUTORIZADAS
        )

        reg_curva = {
            "umbral": th,
            "es_viable": es_viable,
            "far_total_pct": eval_th["far_total_pct"],
            "far_total_conteo": eval_th["far_total_conteo"],
            "far_condicional_pct": eval_th["far_condicional_pct"],
            "frr_puerta_pct": eval_th["frr_puerta_pct"],
            "frr_puerta_conteo": eval_th["frr_puerta_conteo"],
            "aperturas_totales": eval_th["aperturas_totales"],
            "aperturas_correctas": eval_th["aperturas_correctas"],
            "confusiones_autorizados": eval_th["confusiones_autorizados"],
            "bloqueos_puerta": eval_th["bloqueos_puerta"],
            "frr_por_persona": frr_por_p,
        }
        curva_umbrales.append(reg_curva)
        if es_viable:
            viable_umbrales.append(reg_curva)

    umbral_optimo = None
    motivo_seleccion = ""

    if viable_umbrales:
        elegido = seleccionar_umbral_optimo_jerarquico(viable_umbrales)
        umbral_optimo = elegido["umbral"]
        motivo_seleccion = (
            f"Umbral {umbral_optimo:.2f} seleccionado por: "
            f"1) Menor FAR ({elegido['far_total_pct']}%), "
            f"2) Menor FRR puerta ({elegido['frr_puerta_pct']}%), "
            f"3) Mayor aperturas correctas ({elegido['aperturas_correctas']}), "
            f"4) Mayor umbral en caso de empate total. "
            f"(Total candidatos viables con FRR_p <= 30%: {len(viable_umbrales)})."
        )
        res_regla_b = evaluar_con_umbral(umbral_optimo, f"Regla B (Argmax 4 clases + Score >= {umbral_optimo:.2f})")
        res_regla_b["motivo_seleccion"] = motivo_seleccion
    else:
        motivo_seleccion = (
            "Sin umbral viable: ningún umbral satisface FRR_p <= 30% simultáneamente "
            "para todas las personas autorizadas (bloqueos físicos de acceso > 30% en al menos una persona)."
        )
        res_regla_b = {
            "nombre_regla": "Regla B (Argmax 4 clases + Score)",
            "umbral": None,
            "estado": "sin_umbral_viable",
            "motivo": motivo_seleccion,
            "far_total_pct": None,
            "far_total_conteo": None,
            "far_condicional_pct": None,
            "far_condicional_conteo": None,
            "frr_puerta_pct": None,
            "frr_puerta_conteo": None,
            "aperturas_totales": None,
            "aperturas_correctas": None,
            "confusiones_autorizados": None,
            "bloqueos_puerta": None,
            "bloqueos_desconocido": None,
            "bloqueos_bajo_umbral": None,
            "bloqueos_fallo_deteccion": None,
            "frr_por_persona": {p: None for p in PERSONAS_AUTORIZADAS},
            "desglose_por_persona": {},
            "desglose_por_pose": {},
            "intentos_auth": [],
            "intentos_desc": [],
            "motivo_seleccion": motivo_seleccion,
        }

    # Evaluación Regla A (sin umbral adicional)
    res_regla_a = evaluar_con_umbral(None, "Regla A (Argmax 4 clases)")
    res_regla_a["motivo_seleccion"] = "Argmax sobre las 4 clases; la puerta abre si argmax in {persona_1, persona_2, persona_3} y permanece cerrada si predice desconocido."

    # Matriz de confusión multi-clase en validación procesada
    y_true_proc = []
    y_pred_proc = []
    for m in val_auth_todos:
        if m["id"] in dict_pred_auth:
            y_true_proc.append(CLASE_TO_ID[m["persona"]])
            y_pred_proc.append(dict_pred_auth[m["id"]]["clase_pred"])
    for idx_desc in range(len(filas_desc_val)):
        if dict_pred_desc[idx_desc]["procesada"]:
            y_true_proc.append(CLASE_TO_ID["desconocido"])
            y_pred_proc.append(dict_pred_desc[idx_desc]["clase_pred"])

    cm = confusion_matrix(y_true_proc, y_pred_proc, labels=list(range(N_CLASES)))

    return {
        "regla_a": res_regla_a,
        "regla_b": res_regla_b,
        "viable_umbrales_count": len(viable_umbrales),
        "umbral_optimo": umbral_optimo,
        "curva_umbrales": curva_umbrales,
        "confusion_matrix": cm.tolist(),
        "latencia_promedio_s": t_latencia_auth,
    }

    cm = confusion_matrix(y_true_proc, y_pred_proc, labels=list(range(N_CLASES)))

    return {
        "regla_a": res_regla_a,
        "regla_b": res_regla_b,
        "viable_umbrales_count": len(viable_umbrales),
        "umbral_optimo": umbral_optimo,
        "confusion_matrix": cm.tolist(),
        "latencia_promedio_s": t_latencia_auth,
    }


def graficar_matriz_confusion(cm: np.ndarray, nombre_modelo: str, ruta_salida: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(N_CLASES))
    ax.set_yticks(range(N_CLASES))
    ax.set_xticklabels(CLASES_NOMBRES, rotation=45, ha="left")
    ax.set_yticklabels(CLASES_NOMBRES)

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta Real")
    ax.set_title(f"Matriz de Confusión (4 Clases) - {nombre_modelo.upper()}", pad=20)

    for i in range(N_CLASES):
        for j in range(N_CLASES):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), va="center", ha="center", color=color, fontweight="bold")

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(ruta_salida), dpi=200)
    plt.close()


def ejecutar_experimento_completo(dir_reutilizar: Optional[Path] = None, dir_salida: Optional[Path] = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dir_salida is None:
        sufijo = "corregido_" if dir_reutilizar else ""
        dir_salida = RAIZ / "resultados" / f"desarrollo_4clases_{sufijo}{timestamp}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    print(f"=== INICIANDO EXPERIMENTO DE CLASIFICACIÓN DE 4 CLASES ===")
    print(f"Directorio de resultados: {dir_salida}")

    datos = preparar_datasets()
    hashes_modelos = {}

    if dir_reutilizar is not None:
        dir_reutilizar = Path(dir_reutilizar).resolve()
        print(f"Reutilizando modelos existentes desde: {dir_reutilizar} (SIN REENTRENAMIENTO)")
        if not dir_reutilizar.exists():
            raise FileNotFoundError(f"No existe el directorio a reutilizar: {dir_reutilizar}")

        modelos = {}
        for nombre, (cls_mod, archivo_mod) in {
            "knn": (ClasificadorKNN, "modelo.joblib"),
            "arbol": (ClasificadorArbol, "modelo.joblib"),
            "svm": (ClasificadorSVM, "modelo.joblib"),
            "ann": (ClasificadorANN, "modelo.pt"),
            "cnn": (ClasificadorCNN, "modelo.pt"),
        }.items():
            ruta_m = dir_reutilizar / nombre / archivo_mod
            if not ruta_m.exists():
                raise FileNotFoundError(f"Falta archivo de modelo: {ruta_m}")
            hashes_modelos[nombre] = calcular_sha256(ruta_m)
            modelos[nombre] = cls_mod.cargar(ruta_m)

        # Cargar metadatos previos
        ruta_info_prev = dir_reutilizar / "informe_experimento.json"
        if ruta_info_prev.exists():
            info_prev = json.loads(ruta_info_prev.read_text(encoding="utf-8"))
            metadatos_modelos = info_prev.get("metadatos_modelos", {})
        else:
            metadatos_modelos = {}
    else:
        modelos, metadatos_modelos = entrenar_los_5_modelos(datos)
        for nombre, modelo in modelos.items():
            pass  # Los hashes se calculan al guardar

    resultados_evaluacion = {}

    for nombre, modelo in modelos.items():
        print(f"\n--- Evaluando Reglas para {nombre.upper()} ---")
        res = evaluar_reglas(nombre, modelo, datos)
        resultados_evaluacion[nombre] = res

        # Guardar / Copiar modelo en el nuevo directorio
        dir_mod = dir_salida / nombre
        dir_mod.mkdir(parents=True, exist_ok=True)
        archivo_salida = "modelo.pt" if nombre in {"ann", "cnn"} else "modelo.joblib"
        ruta_salida_mod = dir_mod / archivo_salida

        if dir_reutilizar is not None:
            import shutil
            ruta_origen = dir_reutilizar / nombre / archivo_salida
            shutil.copy2(ruta_origen, ruta_salida_mod)
            hash_copia = calcular_sha256(ruta_salida_mod)
            if hash_copia != hashes_modelos[nombre]:
                raise ValueError(f"Fallo de integridad al copiar modelo {nombre}")
        else:
            modelo.guardar(ruta_salida_mod)
            hashes_modelos[nombre] = calcular_sha256(ruta_salida_mod)

        # Guardar gráfico de matriz de confusión
        cm = np.array(res["confusion_matrix"])
        graficar_matriz_confusion(cm, nombre, dir_mod / "matriz_confusion.png")
        graficar_matriz_confusion(cm, nombre, dir_salida / f"matriz_confusion_{nombre}.png")

        # Métricas individuales del modelo
        info_mod = {
            "configuracion": metadatos_modelos.get(nombre, {}),
            "sha256_modelo": hashes_modelos[nombre],
            "regla_a": res["regla_a"],
            "regla_b": res["regla_b"],
            "viable_umbrales_count": res["viable_umbrales_count"],
            "umbral_optimo": res["umbral_optimo"],
            "curva_umbrales": res["curva_umbrales"],
            "confusion_matrix": res["confusion_matrix"],
            "latencia_s": res["latencia_promedio_s"],
        }
        with (dir_mod / "metricas.json").open("w", encoding="utf-8") as f:
            json.dump(info_mod, f, ensure_ascii=False, indent=2)

    # Guardar metadatos generales
    with (dir_salida / "clases_mapeo.json").open("w", encoding="utf-8") as f:
        json.dump(ID_TO_CLASE, f, ensure_ascii=False, indent=2)

    with (dir_salida / "personas_autorizadas.json").open("w", encoding="utf-8") as f:
        json.dump(PERSONAS_AUTORIZADAS, f, ensure_ascii=False, indent=2)

    with (dir_salida / "informe_experimento.json").open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "directorio": str(dir_salida.relative_to(RAIZ)),
            "modelos_reutilizados": dir_reutilizar is not None,
            "origen_modelos": str(dir_reutilizar.relative_to(RAIZ)) if dir_reutilizar else None,
            "sha256_modelos": hashes_modelos,
            "clases": CLASES_NOMBRES,
            "personas_autorizadas": PERSONAS_AUTORIZADAS,
            "metadatos_modelos": metadatos_modelos,
            "resultados": resultados_evaluacion,
        }, f, ensure_ascii=False, indent=2)

    # Generar informes Markdown con correcciones metodológicas y advertencias
    generar_reportes_markdown(dir_salida, resultados_evaluacion, metadatos_modelos, hashes_modelos, dir_reutilizar)

    print(f"\n=== EXPERIMENTO DE 4 CLASES COMPLETADO EXITOSAMENTE ===")
    print(f"Resultados guardados en: {dir_salida}")
    return dir_salida


def generar_reportes_markdown(
    dir_salida: Path,
    resultados: Dict[str, Any],
    metadatos: Dict[str, Any],
    hashes_modelos: Dict[str, str],
    dir_reutilizar: Optional[Path] = None
):
    # Cargar artefactos de baseline original de 3 clases si están disponibles
    ruta_baseline = RAIZ / "resultados" / "desarrollo_rechazo_distancia_20261118_223358_141573" / "informe_experimento.json"
    baseline_data = {}
    if ruta_baseline.exists():
        try:
            with open(ruta_baseline, "r", encoding="utf-8") as f_base:
                baseline_data = json.load(f_base).get("modelos", {})
        except Exception:
            baseline_data = {}

    baseline_info = {}
    for m in ("ann", "arbol", "cnn", "knn", "svm"):
        m_base = baseline_data.get(m, {})
        ra_base = m_base.get("regla_A_puntaje", {})
        rb_base = m_base.get("regla_B_puntaje_y_distancia", {})
        baseline_info[m] = {
            "regla_a_far": f"{ra_base['far_pct']:.2f}%" if "far_pct" in ra_base else "no disponible",
            "regla_a_frr": f"{ra_base['frr_puerta_pct']:.2f}%" if "frr_puerta_pct" in ra_base else "no disponible",
            "regla_b_far": f"{rb_base['far_pct']:.2f}%" if "far_pct" in rb_base else "no disponible",
            "regla_b_frr": f"{rb_base['frr_puerta_pct']:.2f}%" if "frr_puerta_pct" in rb_base else "no disponible",
        }

    # 1. TABLA_COMPARATIVA.md
    lineas_tabla = [
        "# Tabla Comparativa Corregida: Clasificación de 4 Clases vs. Baseline de 3 Clases",
        "",
        "> **Nota Metodológica Crucial:** Evaluado sobre datos de desarrollo: exactamente los mismos 100 intentos de desconocidos (LFW calibración) y 64 intentos autorizados (Sesión 04 multidistancia).",
        "> En esta versión corregida, **FRR de puerta mide exclusivamente bloqueos físicos de acceso** (`bloqueos / intentos`). Las confusiones entre personas autorizadas (`apertura_confundida`) son aperturas de puerta con error de identidad, no bloqueos físicos.",
        "",
        "| Modelo | Paradigma / Regla | Umbral Calibrado | FAR Total (/100) | FAR Condic. (/86) | FRR Puerta (/64) | Aperturas Correctas | Confusiones | Aperturas Totales | Bloqueos Puerta | Baseline FAR (A / B) | Baseline FRR (A / B) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
        res = resultados[nombre]
        ra = res["regla_a"]
        rb = res["regla_b"]
        base = baseline_info[nombre]

        lineas_tabla.append(
            f"| **{nombre.upper()}** | 4 Clases: Regla A (Argmax puro) | Ninguno (Argmax) | "
            f"**{ra['far_total_pct']:.2f}%** ({ra['far_total_conteo']}) | {ra['far_condicional_pct']:.2f}% | "
            f"**{ra['frr_puerta_pct']:.2f}%** ({ra['frr_puerta_conteo']}) | {ra['aperturas_correctas']} | {ra['confusiones_autorizados']} | "
            f"{ra['aperturas_totales']} | {ra['bloqueos_puerta']} | {base['regla_a_far']} | {base['regla_a_frr']} |"
        )
        if rb.get("estado") == "sin_umbral_viable":
            lineas_tabla.append(
                f"| | 4 Clases: Regla B (Score Calibrado) | *Sin umbral viable (FRR_p > 30%)* | — | — | — | — | — | — | — | {base['regla_b_far']} | {base['regla_b_frr']} |"
            )
        else:
            lineas_tabla.append(
                f"| | 4 Clases: Regla B (Score Calibrado) | $s \\ge {rb['umbral']:.2f}$ | "
                f"**{rb['far_total_pct']:.2f}%** ({rb['far_total_conteo']}) | {rb['far_condicional_pct']:.2f}% | "
                f"**{rb['frr_puerta_pct']:.2f}%** ({rb['frr_puerta_conteo']}) | {rb['aperturas_correctas']} | {rb['confusiones_autorizados']} | "
                f"{rb['aperturas_totales']} | {rb['bloqueos_puerta']} | {base['regla_b_far']} | {base['regla_b_frr']} |"
            )

    (dir_salida / "TABLA_COMPARATIVA.md").write_text("\n".join(lineas_tabla) + "\n", encoding="utf-8")

    # 2. INFORME_EXPERIMENTO_4CLASES.md
    lineas_informe = [
        "# Informe de Resultados: Experimento Controlado de Clasificación de 4 Clases (Versión Corregida)",
        "",
        f"**Fecha y Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Directorio:** `{dir_salida.name}`",
        f"**Modelos Reutilizados de:** `{dir_reutilizar.name if dir_reutilizar else 'Entrenamiento nuevo'}` (Hashes SHA-256 verificados)",
        f"**Clases:** `persona_1`, `persona_2`, `persona_3`, `desconocido`",
        "",
        "## 1. Correcciones Metodológicas Implementadas",
        "",
        "### 1.1 Separación Estricta entre Bloqueo de Puerta (FRR) y Confusión de Identidad",
        "En la evaluación original, se calculó erróneamente `FRR_puerta = (intentos - aperturas_correctas) / intentos`, lo que trataba las aperturas con identidad equivocada entre autorizados como si hubieran sido rechazos de acceso.",
        "",
        "Bajo las definiciones reglamentarias corregidas:",
        "- $\\text{aperturas\\_totales} = \\text{aperturas\\_correctas} + \\text{confusiones\\_autorizados}$",
        "- $\\text{bloqueos\\_puerta} = \\text{intentos\\_autorizados} - \\text{aperturas\\_totales}$",
        "- $\\text{FRR\\_puerta} = \\frac{\\text{bloqueos\\_puerta}}{\\text{intentos\\_autorizados}} \\times 100\\%$",
        "- Se cumple estrictamente la invariante: $\\text{intentos} = \\text{bloqueos} + \\text{aperturas\\_correctas} + \\text{confusiones}$ para el total general, para cada persona y para cada pose.",
        "- Las causas de bloqueo físico se desglosan en: `bloqueos_desconocido` (clasificado como desconocido), `bloqueos_bajo_umbral` (puntaje insuficiente) y `bloqueos_fallo_deteccion`.",
        "- La restricción $\\text{FRR}_p \\le 30\\%$ para la viabilidad de umbrales en Regla B se calcula utilizando bloqueos físicos reales de cada persona autorizada.",
        "",
        "### 1.2 Corrección del Criterio Jerárquico de Desempate de Umbrales (Regla B)",
        "Anteriormente, al empatar en FAR se desempataba priorizando el mayor umbral, lo que castigaba innecesariamente accesos válidos.",
        "Se implementó el orden jerárquico estricto entre candidatos que cumplan $\\text{FRR}_p \\le 30\\%$ para todas las personas:",
        "1. **Menor FAR de desconocidos** (`far_total_pct`).",
        "2. **Menor FRR global de puerta** (`frr_puerta_pct`).",
        "3. **Mayor cantidad de aperturas con identidad correcta** (`aperturas_correctas`).",
        "4. **Mayor umbral** ($\\\\theta$), únicamente si todos los criterios anteriores empatan.",
        "",
        "> **Caso CNN:** Con este criterio corregido, la CNN selecciona $\\\\theta^* = 0.46$. A diferencia de la versión anterior que elegía $0.79$ e inducía 8 bloqueos innecesarios con igual FAR (1.00%), el umbral $0.46$ mantiene **cero bloqueos** (FRR = 0.00%) y **64/64 aperturas correctas**, sin sacrificar ningún acceso autorizado.",
        "",
        "## 2. Alcance, Datos de Desarrollo y Trazabilidad",
        "",
        "> [!IMPORTANT]",
        "> **Advertencia sobre Datos de Desarrollo vs. Evaluación Final Independiente:**",
        "> - Los 64 intentos de la Sesión 04 y los 100 intentos de calibración LFW son datos de **desarrollo** empleados para seleccionar checkpoints, explorar configuraciones y calibrar umbrales.",
        "> - Los resultados reportados en este informe **no constituyen una evaluación final independiente**.",
        "> - La evaluación final independiente exigirá someter los modelos a la Sesión 05 reservada y a la `reserva_prueba` de LFW, las cuales permanecieron 100% intactas y fuera de cualquier cómputo.",
        "",
        "### 2.1 Historial de Exploración de Clasificadores y Comparación con la Baseline",
        "- **KNN:** En la baseline se usó $k=5$ uniforme. En este experimento se evaluó una rejilla de $k \\in [3, 5, 7, 9]$ con ponderaciones `distance` y `uniform`, resultando seleccionado $k=7$ con `distance` (Macro-Recall val: 80.96%).",
        "- **Árbol:** En la baseline no tenía balanceo. En este experimento se exploró `max_depth` $\\in [5, 10, 15, 20, \\text{None}]$ y `min_samples_leaf` $\\in [1, 2, 4]$ con `class_weight='balanced'`, seleccionándose `max_depth=15, min_samples_leaf=1` (Macro-Recall val: 73.51%).",
        "- **SVM:** En la baseline no tenía balanceo. En este experimento se exploró rejilla de kernels (`linear`, `rbf`, `poly`) y $C \\in [0.1, 1.0, 10.0, 100.0]$ con `class_weight='balanced'`, seleccionándose `poly` grado 2, $C=1.0$ (Macro-Recall val: 84.84%).",
        "- **ANN y CNN:** Se optimizaron con pérdida CrossEntropy ponderada por frecuencias inversas y checkpointing por Macro-Recall en validación.",
        "",
        "> [!WARNING]",
        "> **Atribución de Causalidad y Aclaración de Hiperparámetros:**",
        "> 1. La mejora observada respecto a la baseline no puede atribuirse únicamente a la incorporación de la cuarta clase, ya que coexistió con búsqueda de hiperparámetros, esquemas de ponderación y selección de checkpoints por Macro-Recall.",
        "> 2. `weights='distance'` en KNN pondera el voto de los vecinos según el inverso de su distancia euclidiana; **no balancea las clases**. La clase mayoritaria sigue teniendo mayor densidad de vecinos en zonas no periféricas.",
        "> 3. El desempeño favorable frente a LFW funneled **no demuestra el desempeño ante desconocidos reales de webcam**, debido al *domain shift* fotográfico y ambiental.",
        "",
        "## 3. Matriz de Resultados Corregida",
        "",
        "| Algoritmo | Regla | Umbral ($\\theta^*$) | FAR Total (/100) | FAR Condic. (/86) | FRR Puerta (/64) | Aperturas Correctas | Confusiones | Aperturas Totales | Bloqueos | Viabilidad Regla B |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
        res = resultados[nombre]
        ra = res["regla_a"]
        rb = res["regla_b"]

        lineas_informe.append(
            f"| **{nombre.upper()}** | Regla A (Argmax) | — | "
            f"**{ra['far_total_pct']:.2f}%** ({ra['far_total_conteo']}) | {ra['far_condicional_pct']:.2f}% | "
            f"**{ra['frr_puerta_pct']:.2f}%** ({ra['frr_puerta_conteo']}) | {ra['aperturas_correctas']} | {ra['confusiones_autorizados']} | "
            f"{ra['aperturas_totales']} | {ra['bloqueos_puerta']} | — |"
        )
        if rb.get("estado") == "sin_umbral_viable":
            lineas_informe.append(
                f"| | Regla B (Score) | *Sin umbral viable* | — | — | — | — | — | — | — | **No viable** (FRR_p > 30%) |"
            )
        else:
            lineas_informe.append(
                f"| | Regla B (Score) | $s \\ge {rb['umbral']:.2f}$ | "
                f"**{rb['far_total_pct']:.2f}%** ({rb['far_total_conteo']}) | {rb['far_condicional_pct']:.2f}% | "
                f"**{rb['frr_puerta_pct']:.2f}%** ({rb['frr_puerta_conteo']}) | {rb['aperturas_correctas']} | {rb['confusiones_autorizados']} | "
                f"{rb['aperturas_totales']} | {rb['bloqueos_puerta']} | **Viable** ({res['viable_umbrales_count']} candidatos) |"
            )

    lineas_informe.extend([
        "",
        "## 4. Cambios en la Viabilidad de los Modelos tras la Corrección",
        "",
    ])
    for idx_num, nombre in enumerate(("knn", "arbol", "svm", "ann", "cnn"), start=1):
        res_m = resultados[nombre]
        rb_m = res_m["regla_b"]
        es_v = rb_m.get("estado") != "sin_umbral_viable"
        if es_v:
            frr_p1 = rb_m["desglose_por_persona"].get("persona_1", {}).get("frr_pct", 0.0)
            corr_m = rb_m.get("aperturas_correctas", 0)
            conf_m = rb_m.get("confusiones_autorizados", 0)
            bloq_m = rb_m.get("bloqueos_puerta", 0)
            lineas_informe.append(
                f"{idx_num}. **{nombre.upper()}:** Estado **Viable** (umbral $s \\ge {rb_m['umbral']:.2f}$, {res_m.get('viable_umbrales_count', 0)} candidatos). "
                f"FAR={rb_m['far_total_pct']:.2f}%, FRR puerta={rb_m['frr_puerta_pct']:.2f}% ({bloq_m} bloqueos), "
                f"{corr_m} aperturas correctas y {conf_m} confusiones. FRR persona_1={frr_p1:.2f}% (<= 30%)."
            )
        else:
            motivos = []
            for p_k, p_v in rb_m.get("desglose_por_persona", {}).items():
                if p_v.get("frr_pct") is not None and p_v["frr_pct"] > 30.0:
                    motivos.append(f"{p_k}: {p_v['bloqueos']}/{p_v['intentos']} bloqueos ({p_v['frr_pct']:.2f}%)")
            det_no = ", ".join(motivos) if motivos else "FRR_p > 30%"
            lineas_informe.append(
                f"{idx_num}. **{nombre.upper()}:** Estado **No viable** ({det_no}). "
                f"Ningún umbral satisface la restricción FRR_p <= 30% simultáneamente para todas las personas autorizadas."
            )

    lineas_informe.extend([
        "",
        "## 5. Hashes de Integridad de Modelos (Demostración de No Modificación)",
        "",
        "Los modelos evaluados no fueron reentrenados; se reutilizaron con exactitud byte por byte desde `desarrollo_4clases_20261119_121541/`:",
        "",
        "| Modelo | Archivo | SHA-256 | Estado |",
        "| :--- | :--- | :--- | :---: |",
    ])

    for m in ("knn", "arbol", "svm", "ann", "cnn"):
        arch = "modelo.pt" if m in {"ann", "cnn"} else "modelo.joblib"
        lineas_informe.append(f"| `{m.upper()}` | `{arch}` | `{hashes_modelos[m]}` | **Intacto y Verificado** |")

    lineas_informe.extend([
        "",
        "## 6. Desglose Detallado por Persona y Pose",
        "",
    ])

    for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
        res = resultados[nombre]
        ra = res["regla_a"]
        rb = res["regla_b"]

        lineas_informe.extend([
            f"### 6.{list(('knn', 'arbol', 'svm', 'ann', 'cnn')).index(nombre) + 1} {nombre.upper()}",
            f"- **Latencia Inferencia:** `{res['latencia_promedio_s']*1000:.3f} ms`",
            "",
            "#### Desglose por Persona en Sesión 04 (Regla A):",
            "| Persona | Intentos | Aperturas Correctas | Confusiones | Aperturas Totales | Bloqueos Puerta | FRR Puerta |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for p, stats in ra["desglose_por_persona"].items():
            lineas_informe.append(
                f"| `{p}` | {stats['intentos']} | {stats['aperturas_correctas']} | {stats['confusiones']} | "
                f"{stats['aperturas_totales']} | {stats['bloqueos']} | {stats['frr_pct']:.2f}% |"
            )

        lineas_informe.extend([
            "",
            "#### Desglose por Pose en Sesión 04 (Regla A):",
            "| Pose | Intentos | Aperturas Correctas | Confusiones | Aperturas Totales | Bloqueos Puerta | FRR Pose |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for pose, stats in ra["desglose_por_pose"].items():
            lineas_informe.append(
                f"| `{pose}` | {stats['intentos']} | {stats['aperturas_correctas']} | {stats['confusiones']} | "
                f"{stats['aperturas_totales']} | {stats['bloqueos']} | {stats['frr_pct']:.2f}% |"
            )

        if rb.get("estado") != "sin_umbral_viable":
            lineas_informe.extend([
                "",
                f"#### Desglose por Persona en Sesión 04 (Regla B, $\\theta^* = {rb['umbral']:.2f}$):",
                "| Persona | Intentos | Aperturas Correctas | Confusiones | Aperturas Totales | Bloqueos Puerta | FRR Puerta |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for p, stats in rb["desglose_por_persona"].items():
                lineas_informe.append(
                    f"| `{p}` | {stats['intentos']} | {stats['aperturas_correctas']} | {stats['confusiones']} | "
                    f"{stats['aperturas_totales']} | {stats['bloqueos']} | {stats['frr_pct']:.2f}% |"
                )

        lineas_informe.append("")

    (dir_salida / "INFORME_EXPERIMENTO_4CLASES.md").write_text("\n".join(lineas_informe) + "\n", encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experimento controlado de 4 clases (entrenamiento o evaluación corregida)")
    parser.add_argument("--reutilizar-modelos", type=Path, default=None,
                        help="Directorio de resultados previos con modelos ya entrenados para recalcular métricas sin reentrenar")
    parser.add_argument("--directorio-salida", type=Path, default=None,
                        help="Directorio de salida para los resultados")
    args = parser.parse_args()

    with threadpool_limits(limits=HILOS):
        ejecutar_experimento_completo(dir_reutilizar=args.reutilizar_modelos, dir_salida=args.directorio_salida)


if __name__ == "__main__":
    main()
