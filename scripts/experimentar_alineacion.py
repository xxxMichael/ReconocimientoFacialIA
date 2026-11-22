"""
Experimento de reentrenamiento y calibración de los cinco modelos (Árbol, KNN, SVM, ANN, CNN)
con soporte de multi-orientación y alineación facial por similitud 2D de 5 puntos.

Características:
- Cuatro clases: persona_1, persona_2, persona_3, desconocido.
- Preprocesamiento canónico: alineación 2D por similitud hacia plantilla 64x64 + escala de grises
  + normalización min-max [0.0, 1.0] + suavizado gaussiano (sigma=0.5).
- Aumento de datos EXCLUSIVO en entrenamiento:
  * Rotaciones pre-alineación moderadas (+-10°, +-15°) para robustez ante ladeos.
  * Rotaciones post-alineación residuales (+-4°) para simular imprecisión de puntos.
  * Semilla reproducible (SEMILLA = 42).
- Validación y desarrollo SIN aumentos (Sesión 04 autorizados + 100 desconocidos LFW de calibración).
- Sesión 05 y reserva de prueba LFW estrictamente EXCLUIDAS y no consultadas.
- Selección jerárquica de umbrales idéntica al protocolo canónico del proyecto.
- Generación de informe comparativo, métricas por pose, latencias y configuración congelada.
"""

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
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
from reconocimiento.preprocesamiento.alineador import (
    PLANTILLA_FACIAL_64x64,
    alinear_rostro,
    preprocesar_rostro_alineado,
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
)
from scripts.calibrar_desarrollo import CALIBRACION_LFW
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


def calcular_tasa_pct(numerador: int, denominador: int) -> Optional[float]:
    if denominador <= 0:
        return None
    return round(100.0 * float(numerador) / float(denominador), 2)


def preparar_captura_alineada(
    registro: RegistroCaptura,
    detector: DetectorYuNet,
    hashes_por_particion: Dict[str, str]
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Procesa una captura de autorizado mediante detección y alineación por similitud de 5 puntos."""
    decision = registro.decision_humana
    entrenamiento = (registro.particion == "entrenamiento")
    caja = registro.caja_aprobada if decision in REVISIONES_VALIDAS else None
    referencia = caja if caja and min(caja[2:]) > 0 else None

    muestra = {
        "id": registro.identificador, "persona": registro.persona, "sesion": registro.sesion,
        "particion": registro.particion, "pose": registro.pose, "decision_humana": decision,
        "referencia": referencia, "origen_referencia": registro.origen_caja if referencia else None,
        "intento": False, "estado_deteccion": None, "caja_yunet": None,
        "puntos_faciales": None, "alineado": False, "angulo_deteccion": 0,
        "iou": None, "iomin": None, "sha256_imagen": None, "estado": None
    }

    if registro.tipo_sujeto != "autorizado" or not registro.consentimiento or decision == "excluida":
        muestra["estado"] = "excluida_por_protocolo"
        return muestra, None

    if entrenamiento and referencia is None:
        muestra["estado"] = "sin_caja_revisada"
        return muestra, None

    muestra["intento"] = True

    ruta = (RAIZ / registro.ruta).resolve()
    if not ruta.is_relative_to(RAIZ_IMAGENES) or ruta.relative_to(RAIZ_IMAGENES).parts[:2] != (
        registro.persona, registro.sesion
    ):
        raise ValueError(f"Ruta de imagen fuera de la sesión permitida para {registro.identificador}.")

    try:
        contenido = ruta.read_bytes()
    except OSError:
        muestra["estado"] = "error_lectura"
        return muestra, None

    sha256 = hashlib.sha256(contenido).hexdigest()
    muestra["sha256_imagen"] = sha256
    if registro.hash_contenido and sha256 != registro.hash_contenido:
        raise ValueError(f"El SHA-256 de {registro.identificador} difiere del manifiesto.")

    particion_previa = hashes_por_particion.setdefault(sha256, registro.particion)
    if particion_previa != registro.particion:
        raise ValueError(f"Imagen duplicada entre particiones de desarrollo: {registro.identificador}.")

    imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR) if contenido else None
    if imagen is None:
        muestra["estado"] = "error_decodificacion"
        return muestra, None

    # Detección con puntos faciales
    deteccion = detector.detectar(imagen)
    muestra["estado_deteccion"] = deteccion.estado.value
    muestra["caja_yunet"] = deteccion.recuadro
    muestra["puntos_faciales"] = deteccion.puntos_faciales.tolist() if deteccion.puntos_faciales is not None else None
    muestra["angulo_deteccion"] = deteccion.angulo_orientacion

    if deteccion.estado != EstadoDeteccion.UN_ROSTRO:
        # En entrenamiento, si el detector duda pero hay caja revisada humana, podemos alinear con fallback
        if entrenamiento and referencia:
            img_proc, meta = preprocesar_rostro_alineado(
                imagen, puntos_faciales=None, recuadro=referencia, resolucion=RESOLUCION, sigma=SIGMA
            )
            muestra["estado"] = "procesada"
            muestra["alineado"] = meta.alineado
            return muestra, img_proc
        muestra["estado"] = deteccion.estado.value.lower()
        return muestra, None

    if referencia and deteccion.recuadro:
        iou, iomin = detector.calcular_iou_y_iomin(deteccion.recuadro, referencia)
        muestra.update(iou=iou, iomin=iomin)
        if max(iou, iomin) < SOLAPAMIENTO_MINIMO:
            if not entrenamiento:
                muestra["estado"] = "localizacion_incorrecta"
                return muestra, None

    # Alineación canónica de 5 puntos
    caja_final = referencia if (referencia and entrenamiento) else deteccion.recuadro
    img_proc, meta = preprocesar_rostro_alineado(
        imagen=imagen,
        puntos_faciales=deteccion.puntos_faciales,
        recuadro=caja_final,
        resolucion=RESOLUCION,
        sigma=SIGMA
    )
    muestra["estado"] = "procesada"
    muestra["alineado"] = meta.alineado
    return muestra, img_proc


def cargar_desconocidos_entrenamiento_alineados(detector, extractor):
    """Carga y alinea los 300 desconocidos de entrenamiento LFW."""
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

        caja_man = (int(fila["caja_yunet_x"]), int(fila["caja_yunet_y"]), int(fila["caja_yunet_w"]), int(fila["caja_yunet_h"]))
        det = detector.detectar(img_bgr)
        pts = det.puntos_faciales if det.estado == EstadoDeteccion.UN_ROSTRO else None

        img_proc, meta = preprocesar_rostro_alineado(
            imagen=img_bgr, puntos_faciales=pts, recuadro=caja_man, resolucion=RESOLUCION, sigma=SIGMA
        )
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
            "alineado": meta.alineado,
        })

    return np.asarray(texturas_256, dtype=np.float32), np.asarray(imagenes_64, dtype=np.float32), metadatos


def cargar_desconocidos_calibracion_alineados(detector, extractor):
    """Carga y alinea los 100 desconocidos de calibración LFW (sin aumentos)."""
    ruta_manifiesto = CALIBRACION_LFW / "manifiesto.csv"
    with ruta_manifiesto.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    if len(filas) != 100:
        raise ValueError(f"Se esperaban 100 registros en {CALIBRACION_LFW}, encontrados: {len(filas)}")

    imagenes_64 = []
    texturas_256 = []
    filas_procesadas = []
    estados = []

    for fila in filas:
        ruta_img = RAIZ / fila["ruta_imagen"]
        if not ruta_img.exists():
            estados.append("archivo_no_encontrado")
            continue

        contenido = ruta_img.read_bytes()
        digest = hashlib.sha256(contenido).hexdigest()
        if digest != fila["sha256"]:
            raise ValueError(f"Hash modificado en desconocido de calibración: {ruta_img}")

        img = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            estados.append("error_decodificacion")
            continue

        det = detector.detectar(img)
        estados.append(det.estado.value)
        if det.estado != EstadoDeteccion.UN_ROSTRO:
            continue

        img_proc, meta = preprocesar_rostro_alineado(
            imagen=img, puntos_faciales=det.puntos_faciales, recuadro=det.recuadro, resolucion=RESOLUCION, sigma=SIGMA
        )
        feat = extractor.extraer(img_proc).vector_caracteristicas

        imagenes_64.append(img_proc)
        texturas_256.append(feat)
        filas_procesadas.append(fila)

    return (
        np.asarray(texturas_256, dtype=np.float32),
        np.asarray(imagenes_64, dtype=np.float32),
        filas,
        estados,
    )


def aumentar_muestra_entrenamiento(
    img_64: np.ndarray,
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """
    Genera aumentos controlados de una muestra de entrenamiento alineada:
    1. Pequeña rotación residual post-alineación (+-3° a +-6°) con BORDER_REFLECT_101.
    2. Variación ligera de escala/contraste (+-8%).
    """
    aumentadas = []

    # Aumento 1: Rotación residual post-alineación
    ang_res = float(rng.uniform(-5.0, 5.0))
    if abs(ang_res) < 1.0:
        ang_res = 3.0 if ang_res >= 0 else -3.0

    M_rot = cv2.getRotationMatrix2D((32.0, 32.0), ang_res, 1.0)
    rot_64 = cv2.warpAffine(img_64, M_rot, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    rot_64 = np.clip(rot_64, 0.0, 1.0).astype(np.float32)
    aumentadas.append(rot_64)

    # Aumento 2: Variación suave de contraste/brillo manteniendo rango [0, 1]
    factor_gamma = float(rng.uniform(0.90, 1.10))
    img_gamma = np.clip(img_64 ** factor_gamma, 0.0, 1.0).astype(np.float32)
    aumentadas.append(img_gamma)

    return aumentadas


def preparar_datasets_con_alineacion_y_aumento():
    print("=" * 70)
    print(" CARGANDO DATASETS CON DETECTOR MULTI-ORIENTACIÓN Y ALINEACIÓN")
    print("=" * 70)

    detector = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)

    # 1. Autorizados
    print("Cargando registros autorizados...")
    registros, hash_manifiesto = cargar_registros(solo_base=False)
    hashes_por_particion = {}
    muestras_autorizadas = []
    imagenes_autorizadas = {}

    for reg in registros:
        muestra, img_proc = preparar_captura_alineada(reg, detector, hashes_por_particion)
        muestras_autorizadas.append(muestra)
        if img_proc is not None:
            imagenes_autorizadas[reg.identificador] = img_proc

    train_auth = [m for m in muestras_autorizadas if m["particion"] == "entrenamiento" and m["estado"] == "procesada"]
    val_auth_todos = [m for m in muestras_autorizadas if m["particion"] == "validacion" and m["sesion"] == "sesion_04"]
    val_auth_proc = [m for m in val_auth_todos if m["estado"] == "procesada"]

    print(f"Autorizados train procesados originales: {len(train_auth)}")
    print(f"Autorizados val procesados (Sesión 04):   {len(val_auth_proc)} / {len(val_auth_todos)}")

    # 2. Desconocidos de entrenamiento (300 LFW)
    print("Cargando 300 desconocidos de entrenamiento LFW...")
    X_train_desc_tex_orig, X_train_desc_img_orig, meta_desc_train = cargar_desconocidos_entrenamiento_alineados(detector, extractor)

    # 3. Desconocidos de validación (100 LFW calibración)
    print("Cargando 100 desconocidos de calibración LFW (validación)...")
    X_val_desc_tex, X_val_desc_img, filas_desc_val, estados_desc_val = cargar_desconocidos_calibracion_alineados(detector, extractor)

    # 4. Aplicar aumento de datos EXCLUSIVAMENTE a entrenamiento
    print("\nAplicando aumentos de datos moderados (rotación residual y gamma) en entrenamiento...")
    rng = np.random.RandomState(SEMILLA)

    conteo_aumentos_por_clase = {c: {"original": 0, "aumentadas": 0, "total": 0} for c in CLASES_NOMBRES}

    # Autorizados con aumento
    X_train_auth_img_list = []
    X_train_auth_tex_list = []
    y_train_auth_list = []

    for m in train_auth:
        persona = m["persona"]
        cid = CLASE_TO_ID[persona]
        img_orig = imagenes_autorizadas[m["id"]]
        tex_orig = extractor.extraer(img_orig).vector_caracteristicas

        # Original
        X_train_auth_img_list.append(img_orig)
        X_train_auth_tex_list.append(tex_orig)
        y_train_auth_list.append(cid)
        conteo_aumentos_por_clase[persona]["original"] += 1

        # Aumentadas (2 por muestra autorizada)
        imgs_aum = aumentar_muestra_entrenamiento(img_orig, rng)
        for aum in imgs_aum:
            X_train_auth_img_list.append(aum)
            X_train_auth_tex_list.append(extractor.extraer(aum).vector_caracteristicas)
            y_train_auth_list.append(cid)
            conteo_aumentos_por_clase[persona]["aumentadas"] += 1

    for persona in PERSONAS_AUTORIZADAS:
        conteo_aumentos_por_clase[persona]["total"] = (
            conteo_aumentos_por_clase[persona]["original"] + conteo_aumentos_por_clase[persona]["aumentadas"]
        )

    # Desconocidos de entrenamiento con aumento ligero (1 aumento por cada una)
    X_train_desc_img_list = []
    X_train_desc_tex_list = []
    y_train_desc_list = []
    cid_desc = CLASE_TO_ID["desconocido"]

    for i in range(len(X_train_desc_img_orig)):
        img_orig = X_train_desc_img_orig[i]
        tex_orig = X_train_desc_tex_orig[i]

        X_train_desc_img_list.append(img_orig)
        X_train_desc_tex_list.append(tex_orig)
        y_train_desc_list.append(cid_desc)
        conteo_aumentos_por_clase["desconocido"]["original"] += 1

        # 1 aumento por desconocido para mantener proporción equilibrada
        ang_res = float(rng.uniform(-4.0, 4.0))
        M_rot = cv2.getRotationMatrix2D((32.0, 32.0), ang_res, 1.0)
        rot_64 = cv2.warpAffine(img_orig, M_rot, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        rot_64 = np.clip(rot_64, 0.0, 1.0).astype(np.float32)

        X_train_desc_img_list.append(rot_64)
        X_train_desc_tex_list.append(extractor.extraer(rot_64).vector_caracteristicas)
        y_train_desc_list.append(cid_desc)
        conteo_aumentos_por_clase["desconocido"]["aumentadas"] += 1

    conteo_aumentos_por_clase["desconocido"]["total"] = (
        conteo_aumentos_por_clase["desconocido"]["original"] + conteo_aumentos_por_clase["desconocido"]["aumentadas"]
    )

    print("Resumen de muestras por clase en entrenamiento:")
    for cname, cinfo in conteo_aumentos_por_clase.items():
        print(f"  {cname:15s}: {cinfo['original']} orig + {cinfo['aumentadas']} aum = {cinfo['total']} total")

    # Matrices finales de entrenamiento
    X_train_tex = np.vstack([np.asarray(X_train_auth_tex_list), np.asarray(X_train_desc_tex_list)])
    X_train_img = np.vstack([np.asarray(X_train_auth_img_list), np.asarray(X_train_desc_img_list)])
    y_train = np.concatenate([np.asarray(y_train_auth_list), np.asarray(y_train_desc_list)])

    # Validación (SIN AUMENTOS)
    X_val_auth_tex = np.stack([extractor.extraer(imagenes_autorizadas[m["id"]]).vector_caracteristicas for m in val_auth_proc])
    X_val_auth_img = np.stack([imagenes_autorizadas[m["id"]] for m in val_auth_proc])
    y_val_auth = np.array([CLASE_TO_ID[m["persona"]] for m in val_auth_proc], dtype=np.int64)

    y_val_desc = np.full(len(X_val_desc_tex), CLASE_TO_ID["desconocido"], dtype=np.int64)

    X_val_tex = np.vstack([X_val_auth_tex, X_val_desc_tex])
    X_val_img = np.vstack([X_val_auth_img, X_val_desc_img])
    y_val = np.concatenate([y_val_auth, y_val_desc])

    # Pesos de clase balanceados
    counts = Counter(y_train)
    total_samples = len(y_train)
    pesos_clases = np.array([total_samples / (N_CLASES * counts[c]) for c in range(N_CLASES)], dtype=np.float32)

    print(f"\nTotal Train (con aumentos): {len(y_train)} muestras. Pesos clase: {pesos_clases}")
    print(f"Total Val (sin aumentos):   {len(y_val)} muestras procesadas ({len(val_auth_proc)} auth + {len(X_val_desc_tex)} desc).")

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
        "X_val_desc_img": X_val_desc_img,
        "X_val_desc_tex": X_val_desc_tex,
        "imagenes_autorizadas": imagenes_autorizadas,
        "conteo_aumentos": conteo_aumentos_por_clase,
    }


def entrenar_los_cinco_modelos(datos: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(" REENTRENANDO LOS CINCO CLASIFICADORES (ÁRBOL, KNN, SVM, ANN, CNN)")
    print("=" * 70)

    X_train_tex = datos["X_train_tex"]
    y_train = datos["y_train"]
    X_val_tex = datos["X_val_tex"]
    y_val = datos["y_val"]
    X_train_img = datos["X_train_img"]
    X_val_img = datos["X_val_img"]
    pesos_clases = datos["pesos_clases"]

    modelos = {}
    metadatos_modelos = {}

    # 1. KNN
    print("\n--- 1. KNN (Búsqueda sobre k in (3, 5, 7, 9) y ponderación in ('distance', 'uniform')) ---")
    mejor_knn_score = -1.0
    mejor_knn = None
    mejor_knn_cfg = None
    for k in (3, 5, 7, 9):
        for pond in ("distance", "uniform"):
            knn_cand = ClasificadorKNN(k=k, metrica="euclidean", ponderacion=pond)
            knn_cand.entrenar(X_train_tex, y_train)
            y_pred_val = knn_cand.predecir(X_val_tex)
            score = calcular_macro_recall(y_val, y_pred_val)
            if score > mejor_knn_score:
                mejor_knn_score = score
                mejor_knn = knn_cand
                mejor_knn_cfg = {"k": k, "ponderacion": pond, "metrica": "euclidean", "macro_recall_val": round(score, 2)}
    modelos["knn"] = mejor_knn
    metadatos_modelos["knn"] = mejor_knn_cfg
    print(f"KNN seleccionado: {mejor_knn_cfg}")

    # 2. Árbol de Decisión
    print("\n--- 2. Árbol de Decisión (4 clases, class_weight='balanced') ---")
    arbol, hist_arbol, cfg_arbol = ClasificadorArbol.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight="balanced", metrica="macro_recall", semilla=SEMILLA
    )
    y_pred_arbol = arbol.predecir(X_val_tex)
    cfg_arbol["macro_recall_val"] = round(calcular_macro_recall(y_val, y_pred_arbol), 2)
    cfg_arbol["n_hojas"] = arbol.n_hojas_
    cfg_arbol["profundidad_efectiva"] = arbol.profundidad_efectiva_
    modelos["arbol"] = arbol
    metadatos_modelos["arbol"] = cfg_arbol
    print(f"Árbol seleccionado: {cfg_arbol}")

    # 3. SVM
    print("\n--- 3. SVM (4 clases, class_weight='balanced') ---")
    svm, hist_svm, cfg_svm = ClasificadorSVM.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight="balanced", metrica="macro_recall", semilla=SEMILLA
    )
    y_pred_svm = svm.predecir(X_val_tex)
    cfg_svm["macro_recall_val"] = round(calcular_macro_recall(y_val, y_pred_svm), 2)
    cfg_svm["n_vectores_soporte"] = svm.n_vectores_soporte_
    modelos["svm"] = svm
    metadatos_modelos["svm"] = cfg_svm
    print(f"SVM seleccionado: {cfg_svm}")

    # 4. ANN
    print("\n--- 4. ANN (4 clases, pérdida ponderada, early checkpointing) ---")
    ann = ClasificadorANN(n_entrada=X_train_tex.shape[1], n_clases=N_CLASES, semilla=SEMILLA)
    ann.entrenar_con_validacion(
        X_train_tex, y_train, X_val_tex, y_val,
        pesos_clases=pesos_clases, metrica_checkpoint="macro_recall"
    )
    y_pred_ann = ann.predecir(X_val_tex)
    macro_rec_ann = calcular_macro_recall(y_val, y_pred_ann)
    metadatos_modelos["ann"] = {
        "mejor_epoca": ann.mejor_epoca_,
        "macro_recall_val": round(macro_rec_ann, 2),
        "n_clases": N_CLASES,
        "lr": ann.lr,
        "dropout": ann.dropout,
        "tamano_lote": ann.tamano_lote,
        "parametros_entrenables": ann.red_wrapper.contar_parametros(),
    }
    modelos["ann"] = ann
    print(f"ANN mejor época: {ann.mejor_epoca_}, Val Macro-Recall: {macro_rec_ann:.2f}%")

    # 5. CNN
    print("\n--- 5. CNN (4 clases, tensores (64,64) alineados, pérdida ponderada) ---")
    cnn = ClasificadorCNN(n_clases=N_CLASES, semilla=SEMILLA)
    cnn.entrenar_con_validacion(
        X_train_img, y_train, X_val_img, y_val,
        pesos_clases=pesos_clases, metrica_checkpoint="macro_recall"
    )
    y_pred_cnn = cnn.predecir(X_val_img)
    macro_rec_cnn = calcular_macro_recall(y_val, y_pred_cnn)
    metadatos_modelos["cnn"] = {
        "mejor_epoca": cnn.mejor_epoca_,
        "macro_recall_val": round(macro_rec_cnn, 2),
        "n_clases": N_CLASES,
        "lr": cnn.lr,
        "dropout": cnn.dropout,
        "tamano_lote": cnn.tamano_lote,
    }
    modelos["cnn"] = cnn
    print(f"CNN mejor época: {cnn.mejor_epoca_}, Val Macro-Recall: {macro_rec_cnn:.2f}%")

    return modelos, metadatos_modelos


def evaluar_modelo_reglas(nombre_modelo: str, modelo: Any, datos: Dict[str, Any]) -> Dict[str, Any]:
    """Evalúa el modelo sobre las 64 muestras autorizadas y 100 desconocidos de validación."""
    es_cnn = (nombre_modelo == "cnn")
    val_auth_todos = datos["val_auth_todos"]
    filas_desc_val = datos["filas_desc_val"]
    estados_desc_val = datos["estados_desc_val"]

    # Inferencia en autorizados procesados
    ids_procesados = datos["val_auth_proc_ids"]
    if es_cnn:
        imagenes_proc = [datos["imagenes_autorizadas"][mid] for mid in ids_procesados]
        X_auth = np.stack(imagenes_proc)
        probs_auth = modelo.predecir_probabilidades(X_auth)
    else:
        extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)
        texturas_proc = [extractor.extraer(datos["imagenes_autorizadas"][mid]).vector_caracteristicas for mid in ids_procesados]
        X_auth = np.stack(texturas_proc)
        probs_auth = modelo.predecir_probabilidades(X_auth)

    dict_pred_auth = {}
    for idx, mid in enumerate(ids_procesados):
        p_vec = probs_auth[idx]
        pred_c = int(np.argmax(p_vec))
        dict_pred_auth[mid] = {
            "pred_idx": pred_c,
            "persona_pred": ID_TO_CLASE[pred_c],
            "score": float(p_vec[pred_c]),
            "probs": p_vec.tolist(),
        }

    # Inferencia en desconocidos de calibración
    dict_pred_desc = {}
    idx_proc = 0
    if es_cnn and "X_val_desc_img" in datos:
        X_desc = datos["X_val_desc_img"]
    elif not es_cnn and "X_val_desc_tex" in datos:
        X_desc = datos["X_val_desc_tex"]
    elif "X_val_desc_img" in datos and "X_val_desc_tex" in datos:
        X_desc = datos["X_val_desc_img"] if es_cnn else datos["X_val_desc_tex"]
    else:
        if es_cnn:
            X_desc = datos["X_val_img"][len(ids_procesados):]
        else:
            X_desc = datos["X_val_tex"][len(ids_procesados):]

    if len(X_desc) > 0:
        probs_desc = modelo.predecir_probabilidades(X_desc)

    for idx_desc in range(len(filas_desc_val)):
        est = estados_desc_val[idx_desc]
        if est != "UN_ROSTRO":
            dict_pred_desc[idx_desc] = {"procesada": False, "estado_deteccion": est, "persona_pred": None, "score": None}
        else:
            p_vec = probs_desc[idx_proc]
            pred_c = int(np.argmax(p_vec))
            dict_pred_desc[idx_desc] = {
                "procesada": True,
                "estado_deteccion": est,
                "pred_idx": pred_c,
                "persona_pred": ID_TO_CLASE[pred_c],
                "score": float(p_vec[pred_c]),
            }
            idx_proc += 1

    def evaluar_con_umbral(th_eval: Optional[float], nombre_regla: str) -> Dict[str, Any]:
        aperturas_correctas = 0
        confusiones_autorizados = 0
        bloqueos_desconocido = 0
        bloqueos_umbral = 0
        bloqueos_deteccion = 0

        conteo_por_persona = {
            p: {
                "intentos": 0, "aperturas_correctas": 0, "confusiones": 0,
                "aperturas_totales": 0, "bloqueos": 0, "bloqueos_desconocido": 0,
                "bloqueos_bajo_umbral": 0, "bloqueos_fallo_deteccion": 0, "frr_pct": None
            }
            for p in PERSONAS_AUTORIZADAS
        }
        conteo_por_pose = {}

        for m in val_auth_todos:
            p_real = m["persona"]
            pose = m["pose"]

            conteo_por_persona[p_real]["intentos"] += 1
            conteo_por_pose.setdefault(pose, {
                "intentos": 0, "aperturas_correctas": 0, "confusiones": 0,
                "aperturas_totales": 0, "bloqueos": 0, "frr_pct": None
            })
            conteo_por_pose[pose]["intentos"] += 1

            if m["id"] not in dict_pred_auth:
                bloqueos_deteccion += 1
                conteo_por_persona[p_real]["bloqueos"] += 1
                conteo_por_persona[p_real]["bloqueos_fallo_deteccion"] += 1
                conteo_por_pose[pose]["bloqueos"] += 1
                continue

            pred = dict_pred_auth[m["id"]]
            p_pred = pred["persona_pred"]
            score = pred["score"]

            if p_pred == "desconocido":
                bloqueos_desconocido += 1
                conteo_por_persona[p_real]["bloqueos"] += 1
                conteo_por_persona[p_real]["bloqueos_desconocido"] += 1
                conteo_por_pose[pose]["bloqueos"] += 1
            elif th_eval is not None and score < th_eval:
                bloqueos_umbral += 1
                conteo_por_persona[p_real]["bloqueos"] += 1
                conteo_por_persona[p_real]["bloqueos_bajo_umbral"] += 1
                conteo_por_pose[pose]["bloqueos"] += 1
            elif p_pred == p_real:
                aperturas_correctas += 1
                conteo_por_persona[p_real]["aperturas_correctas"] += 1
                conteo_por_persona[p_real]["aperturas_totales"] += 1
                conteo_por_pose[pose]["aperturas_correctas"] += 1
                conteo_por_pose[pose]["aperturas_totales"] += 1
            else:
                confusiones_autorizados += 1
                conteo_por_persona[p_real]["confusiones"] += 1
                conteo_por_persona[p_real]["aperturas_totales"] += 1
                conteo_por_pose[pose]["confusiones"] += 1
                conteo_por_pose[pose]["aperturas_totales"] += 1

        total_auth = len(val_auth_todos)
        aperturas_totales = aperturas_correctas + confusiones_autorizados
        bloqueos_puerta = total_auth - aperturas_totales
        frr_puerta = calcular_tasa_pct(bloqueos_puerta, total_auth)

        for p, stats in conteo_por_persona.items():
            stats["frr_pct"] = calcular_tasa_pct(stats["bloqueos"], stats["intentos"])
        for pose, stats in conteo_por_pose.items():
            stats["frr_pct"] = calcular_tasa_pct(stats["bloqueos"], stats["intentos"])

        # Evaluación en Desconocidos
        fa_count = 0
        for idx_desc in range(len(filas_desc_val)):
            info = dict_pred_desc[idx_desc]
            if not info["procesada"]:
                continue
            p_pred = info["persona_pred"]
            score = info["score"]
            if p_pred in PERSONAS_AUTORIZADAS and (th_eval is None or score >= th_eval):
                fa_count += 1

        total_desc = len(filas_desc_val)
        far_total = calcular_tasa_pct(fa_count, total_desc)

        return {
            "nombre_regla": nombre_regla,
            "umbral": th_eval,
            "far_total_pct": far_total,
            "far_conteo": f"{fa_count}/{total_desc}",
            "frr_puerta_pct": frr_puerta,
            "frr_conteo": f"{bloqueos_puerta}/{total_auth}",
            "aperturas_totales": aperturas_totales,
            "aperturas_correctas": aperturas_correctas,
            "confusiones_autorizados": confusiones_autorizados,
            "bloqueos_puerta": bloqueos_puerta,
            "bloqueos_desconocido": bloqueos_desconocido,
            "bloqueos_bajo_umbral": bloqueos_umbral,
            "bloqueos_fallo_deteccion": bloqueos_deteccion,
            "frr_por_persona": {p: conteo_por_persona[p]["frr_pct"] for p in PERSONAS_AUTORIZADAS},
            "desglose_por_persona": conteo_por_persona,
            "desglose_por_pose": conteo_por_pose,
        }

    # Búsqueda de umbral
    umbrales_candidatos = [round(th, 2) for th in np.arange(0.00, 1.01, 0.01)]
    curva_umbrales = []
    viable_umbrales = []

    for th in umbrales_candidatos:
        eval_th = evaluar_con_umbral(th, f"Candidato th={th:.2f}")
        frr_por_p = eval_th["frr_por_persona"]
        es_viable = all((frr_por_p[p] is not None and frr_por_p[p] <= 30.0) for p in PERSONAS_AUTORIZADAS)
        curva_umbrales.append(eval_th)
        if es_viable:
            viable_umbrales.append(eval_th)

    res_regla_a = evaluar_con_umbral(None, "Regla A (Sin Umbral)")

    if viable_umbrales:
        mejor_curva = seleccionar_umbral_optimo_jerarquico(viable_umbrales)
        mejor_umbral = mejor_curva["umbral"]
        tiene_viable = True
        estado = "viable"
        res_regla_b = evaluar_con_umbral(mejor_umbral, f"Regla B (Umbral Óptimo={mejor_umbral:.2f})")
        res_regla_b["habilitada"] = True
    else:
        mejor_umbral = None
        tiene_viable = False
        estado = "sin_umbral_viable"
        res_regla_b = None  # Regla B no habilitada para este modelo

    return {
        "estado": estado,
        "umbral_optimo": mejor_umbral,
        "umbral_recomendado": mejor_umbral,
        "tiene_umbral_viable": tiene_viable,
        "regla_a": res_regla_a,
        "regla_b": res_regla_b,
        "regla_b_habilitada": tiene_viable,
        "curva_umbrales": curva_umbrales,
    }


def medir_latencias_componentes(cnn_model, datos):
    print("\n" + "=" * 70)
    print(" MEDICIÓN RIGUROSA DE LATENCIAS POR FASE (CPU)")
    print("=" * 70)

    detector_estandar = DetectorYuNet(modo_multiorientacion="estandar")
    detector_cascada = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    n_rep = 25

    # 1. Detección estándar (0°)
    t_det_0 = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        _ = detector_estandar.detectar(dummy_frame)
        t_det_0.append((time.perf_counter() - t0) * 1000.0)

    # 2. Detección cascada con rostro upright (1 pasada)
    t_det_cascada = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        _ = detector_cascada.detectar(dummy_frame)
        t_det_cascada.append((time.perf_counter() - t0) * 1000.0)

    # 3. Alineación 2D por similitud pura (a partir de puntos)
    pts_dummy = np.array([[320.0, 200.0], [410.0, 202.0], [365.0, 250.0], [330.0, 290.0], [400.0, 290.0]], dtype=np.float32)
    t_alin = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        _ = alinear_rostro(dummy_frame, puntos_faciales=pts_dummy, resolucion=(64, 64))
        t_alin.append((time.perf_counter() - t0) * 1000.0)

    # 4. Inferencia CNN
    dummy_input_cnn = np.random.rand(1, 64, 64).astype(np.float32)
    t_cnn = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        _ = cnn_model.predecir_probabilidades(dummy_input_cnn)
        t_cnn.append((time.perf_counter() - t0) * 1000.0)

    lat_det_ms = float(np.mean(t_det_cascada))
    lat_alin_ms = float(np.mean(t_alin))
    lat_inf_ms = float(np.mean(t_cnn))
    lat_total_ms = lat_det_ms + lat_alin_ms + lat_inf_ms
    fps_efectivo = 1000.0 / lat_total_ms if lat_total_ms > 0 else 0.0

    print(f"Latencia Detección YuNet:    {lat_det_ms:.2f} ms")
    print(f"Latencia Alineación 5-pts:   {lat_alin_ms:.2f} ms")
    print(f"Latencia Inferencia CNN:     {lat_inf_ms:.2f} ms")
    print(f"Latencia Total Recorrido:    {lat_total_ms:.2f} ms (~{fps_efectivo:.1f} FPS)")

    return {
        "latencia_deteccion_ms": round(lat_det_ms, 2),
        "latencia_alineacion_ms": round(lat_alin_ms, 2),
        "latencia_inferencia_ms": round(lat_inf_ms, 2),
        "latencia_total_recorrido_ms": round(lat_total_ms, 2),
        "fps_estimado": round(fps_efectivo, 1),
    }


def ejecutar_experimento_completo() -> Path:
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"desarrollo_4clases_alineado_{marca_tiempo}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" INICIO DEL EXPERIMENTO DE ALINEACIÓN FACIAL Y MULTI-ORIENTACIÓN")
    print(f" Directorio de salida: {dir_salida}")
    print("=" * 70)

    # 1. Preparar datos
    datos = preparar_datasets_con_alineacion_y_aumento()

    # 2. Reentrenar modelos
    modelos, metadatos_modelos = entrenar_los_cinco_modelos(datos)

    # 3. Guardar modelos entrenados en el directorio aislado
    ext_modelos = {
        "knn": "modelo.joblib",
        "arbol": "modelo.joblib",
        "svm": "modelo.joblib",
        "ann": "modelo.pt",
        "cnn": "modelo.pt",
    }
    for nombre, modelo in modelos.items():
        dir_mod = dir_salida / nombre
        dir_mod.mkdir(parents=True, exist_ok=True)
        archivo_modelo = dir_mod / ext_modelos.get(nombre, "modelo.bin")
        modelo.guardar(archivo_modelo)

    # Guardar mapeo de clases y autorizados
    clases_mapeo = {
        "id_to_persona": ID_TO_CLASE,
        "persona_to_id": CLASE_TO_ID,
        "personas_autorizadas": PERSONAS_AUTORIZADAS,
    }
    with open(dir_salida / "clases_mapeo.json", "w", encoding="utf-8") as f:
        json.dump(clases_mapeo, f, indent=2, ensure_ascii=False)

    with open(dir_salida / "personas_autorizadas.json", "w", encoding="utf-8") as f:
        json.dump(PERSONAS_AUTORIZADAS, f, indent=2, ensure_ascii=False)

    # 4. Evaluar reglas y calibrar umbrales
    print("\n" + "=" * 70)
    print(" EVALUACIÓN DE REGLAS DE CONTROL DE ACCESO Y CALIBRACIÓN DE UMBRALES")
    print("=" * 70)

    evaluaciones = {}
    for nombre, modelo in modelos.items():
        print(f"\n--- Evaluando {nombre.upper()} ---")
        ev = evaluar_modelo_reglas(nombre, modelo, datos)
        evaluaciones[nombre] = ev
        if ev["tiene_umbral_viable"] and ev["regla_b"] is not None:
            rb = ev["regla_b"]
            print(f"  Estado:              {ev['estado']}")
            print(f"  Umbral seleccionado: {ev['umbral_optimo']}")
            print(f"  FAR Desconocidos:    {rb['far_total_pct']}% ({rb['far_conteo']})")
            print(f"  FRR Puerta Global:   {rb['frr_puerta_pct']}% ({rb['frr_conteo']})")
            print(f"  Aperturas Correctas: {rb['aperturas_correctas']} / 64")
            print(f"  Confusiones Auth:    {rb['confusiones_autorizados']} / 64")
            print(f"  FRR por Persona:     {rb['frr_por_persona']}")
        else:
            ra = ev["regla_a"]
            print(f"  Estado:              {ev['estado']} (Umbral recomendado: null, Regla B no habilitada)")
            print(f"  Regla A - FAR Desc:  {ra['far_total_pct']}% ({ra['far_conteo']})")
            print(f"  Regla A - FRR Global:{ra['frr_puerta_pct']}% ({ra['frr_conteo']})")
            print(f"  Regla A - FRR por P: {ra['frr_por_persona']}")

    # 5. Medir latencias con CNN
    latencias = medir_latencias_componentes(modelos["cnn"], datos)

    # 6. Generar tabla comparativa Markdown
    texto_tabla = generar_tabla_comparativa(evaluaciones, metadatos_modelos)
    with open(dir_salida / "TABLA_COMPARATIVA.md", "w", encoding="utf-8") as f:
        f.write(texto_tabla)

    # 7. Generar informe detallado JSON
    informe_completo = {
        "fecha": datetime.now().isoformat(),
        "directorio": str(dir_salida),
        "conteo_aumentos": datos["conteo_aumentos"],
        "metadatos_modelos": metadatos_modelos,
        "evaluaciones": {
            k: {
                "estado": v["estado"],
                "umbral_optimo": v["umbral_optimo"],
                "umbral_recomendado": v["umbral_recomendado"],
                "tiene_umbral_viable": v["tiene_umbral_viable"],
                "regla_b_habilitada": v["regla_b_habilitada"],
                "regla_a": v["regla_a"],
                "regla_b": v["regla_b"],
                "curva_umbrales": v["curva_umbrales"],
            }
            for k, v in evaluaciones.items()
        },
        "latencias": latencias,
    }
    with open(dir_salida / "informe_experimento.json", "w", encoding="utf-8") as f:
        json.dump(informe_completo, f, indent=2, ensure_ascii=False)

    # 8. Congelar la configuración del nuevo candidato experimental alineado SOLO SI es viable
    exportar_candidato_congelado_si_viable(evaluaciones, dir_salida)

    print(f"\nDirectorio de resultados generado: {dir_salida}")
    print(f"Tabla comparativa: {dir_salida / 'TABLA_COMPARATIVA.md'}")

    return dir_salida


def generar_tabla_comparativa(
    evaluaciones: Dict[str, Any],
    metadatos_modelos: Dict[str, Any],
    fecha_str: Optional[str] = None
) -> str:
    fecha_display = fecha_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lineas_tabla = [
        "# Tabla Comparativa de los Cinco Modelos (Con Alineación y Multi-Orientación)\n",
        f"Fecha: {fecha_display}\n",
        "| Modelo | Val Macro-Recall | Umbral Calibrado | FAR Desconocidos (%) | FRR Puerta (%) | Aperturas Correctas | Confusiones | FRR p1 / p2 / p3 (%) | Cumple Restricción (FRR_p ≤ 30%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for nombre in ("arbol", "knn", "svm", "ann", "cnn"):
        if nombre not in evaluaciones:
            continue
        ev = evaluaciones[nombre]
        cfg = metadatos_modelos.get(nombre, {})
        ra = ev["regla_a"]
        rb = ev.get("regla_b")
        es_viable = ev.get("tiene_umbral_viable", False)

        if es_viable and rb is not None:
            th_str = f"{ev['umbral_optimo']:.2f}"
            far_str = f"{rb['far_total_pct']}% ({rb['far_conteo']})"
            frr_str = f"{rb['frr_puerta_pct']}% ({rb['frr_conteo']})"
            aperturas_str = f"{rb['aperturas_correctas']}/64"
            confusiones_str = f"{rb['confusiones_autorizados']}/64"
            frr_p = rb["frr_por_persona"]
            p_str = f"{frr_p.get('persona_1', 'N/A')}% / {frr_p.get('persona_2', 'N/A')}% / {frr_p.get('persona_3', 'N/A')}%"
            cumple_str = "**SÍ**"
        else:
            th_str = "null"
            far_str = "N/A (Regla B deshabilitada)"
            frr_str = "N/A (Regla B deshabilitada)"
            aperturas_str = "N/A"
            confusiones_str = "N/A"
            p_str = "N/A"
            cumple_str = f"**NO** ({ev.get('estado', 'sin_umbral_viable')})"

        lineas_tabla.append(
            f"| **{nombre.upper()}** | {cfg.get('macro_recall_val', 'N/A')}% | {th_str} | "
            f"{far_str} | {frr_str} | "
            f"{aperturas_str} | {confusiones_str} | {p_str} | {cumple_str} |"
        )

    # Detalle explicativo de Regla A para modelos sin umbral viable
    lineas_tabla.extend([
        "\n### Detalle de Modelos sin Umbral Viable (Métricas de Regla A - Sin Umbral de Score)\n",
        "Para los modelos donde ningún umbral cumplió la restricción física $FRR_p \\le 30\\%$ simultáneamente para todas las personas autorizadas, el estado es `sin_umbral_viable`, el **umbral recomendado es null** y la **Regla B permanece deshabilitada**. A continuación se reportan sus métricas base bajo la Regla A (argmax puro sin umbral) y su curva exploratoria:\n",
    ])
    for nombre in ("arbol", "knn", "svm", "ann", "cnn"):
        if nombre not in evaluaciones:
            continue
        ev = evaluaciones[nombre]
        if not ev.get("tiene_umbral_viable", False):
            ra = ev["regla_a"]
            lineas_tabla.append(
                f"- **{nombre.upper()}:** Estado `{ev.get('estado')}`, Umbral Recomendado: `null` (Regla B deshabilitada).\n"
                f"  * Regla A (Sin Umbral): FAR = {ra['far_total_pct']}% ({ra['far_conteo']}), "
                f"FRR Puerta Global = {ra['frr_puerta_pct']}% ({ra['frr_conteo']}), "
                f"Aperturas Correctas = {ra['aperturas_correctas']}/64, "
                f"Confusiones Autorizados = {ra['confusiones_autorizados']}/64.\n"
                f"  * FRR por persona en Regla A: persona_1={ra['frr_por_persona'].get('persona_1')}%, "
                f"persona_2={ra['frr_por_persona'].get('persona_2')}%, "
                f"persona_3={ra['frr_por_persona'].get('persona_3')}%. "
                f"(Requiere $\\le 30\\%$ para todas las personas para habilitar Regla B).\n"
                f"  * Curva exploratoria disponible: {len(ev.get('curva_umbrales', []))} umbrales evaluados en rango [0.00, 1.00]."
            )

    return "\n".join(lineas_tabla) + "\n"


def exportar_candidato_congelado_si_viable(evaluaciones: Dict[str, Any], dir_salida: Path) -> Optional[Path]:
    mejor_modelo_nombre = "cnn"
    mejor_ev = evaluaciones.get(mejor_modelo_nombre, {})

    if not mejor_ev.get("tiene_umbral_viable") or mejor_ev.get("umbral_optimo") is None:
        print(f"\n[SEGURIDAD] El modelo {mejor_modelo_nombre.upper()} no cuenta con un umbral viable (estado: {mejor_ev.get('estado')}).")
        print("POR REGLA DE SEGURIDAD: NO se recomienda ni exporta automáticamente un candidato no viable.")
        return None

    dir_salida_res = Path(dir_salida).resolve()
    ruta_pt = dir_salida_res / mejor_modelo_nombre / "modelo.pt"
    if not ruta_pt.exists():
        raise FileNotFoundError(f"No existe el archivo de modelo en {ruta_pt}")
    sha_cnn = calcular_sha256(ruta_pt)

    ruta_yunet = RAIZ / "modelos" / "detector" / "face_detection_yunet_2023mar.onnx"
    sha_yunet = calcular_sha256(ruta_yunet)

    cfg_candidato_alineado = {
        "candidato": {
            "identificador": "cnn_4clases_alineada",
            "nombre": "CNN 4 Clases (Experimental Con Alineación 5-Puntos)",
            "tipo_arquitectura": "CNN",
            "ruta_modelo": str(ruta_pt.resolve().relative_to(RAIZ.resolve())).replace("\\", "/"),
            "sha256_modelo": sha_cnn,
            "umbral_score": mejor_ev["umbral_optimo"],
            "orden_clases": CLASES_NOMBRES,
            "mapeo_clases": ID_TO_CLASE,
            "personas_autorizadas": PERSONAS_AUTORIZADAS,
            "fecha_congelamiento": datetime.now().isoformat(),
            "estado": "experimental_alineado"
        },
        "detector_facial": {
            "tipo": "YuNet",
            "ruta_modelo": str(ruta_yunet.relative_to(RAIZ)).replace("\\", "/"),
            "sha256_modelo": sha_yunet,
            "parametros": {
                "score_threshold": 0.6,
                "nms_threshold": 0.3,
                "margen_expansion": 0.1,
                "modo_multiorientacion": "cascada",
                "orientaciones_soportadas": [0, 90, 180, 270]
            }
        },
        "preprocesamiento": {
            "metodo": "alineacion_5puntos",
            "plantilla_canonica": "PLANTILLA_FACIAL_64x64",
            "resolucion": [64, 64],
            "interpolacion": "INTER_AREA_o_LINEAR",
            "borde": "BORDER_REFLECT_101",
            "transformacion": "similitud_2d_sin_reflejo",
            "normalizacion_intensidad": "float32_0_1",
            "suavizado_gaussiano": {"sigma": 0.5, "kernel": "automatico"}
        },
        "regla_acceso": {
            "tipo": "regla_canonica_completa",
            "restriccion_frr_persona_max": 0.30,
            "decision_puerta_desconocido": "CERRADA_ESTRICTA",
            "criterio_apertura": "score >= umbral_score AND clase_predicha IN personas_autorizadas AND deteccion == 'UN_ROSTRO'"
        }
    }

    ruta_cfg_congelado = RAIZ / "configuracion_candidato_alineado.json"
    with open(ruta_cfg_congelado, "w", encoding="utf-8") as f:
        json.dump(cfg_candidato_alineado, f, indent=2, ensure_ascii=False)

    print(f"\nConfiguración del candidato alineado congelada en: {ruta_cfg_congelado}")
    return ruta_cfg_congelado


def preparar_datos_validacion_desarrollo() -> Dict[str, Any]:
    print("\nCargando datos de validación de desarrollo (Sesión 04 + Calibración LFW)...")
    detector = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)

    # 1. Autorizados Sesión 04 (Validación de desarrollo exclusivamente)
    registros, _ = cargar_registros(solo_base=False)
    hashes_por_particion = {}
    val_auth_todos = []
    imagenes_autorizadas = {}

    for reg in registros:
        if reg.particion == "validacion" and reg.sesion == "sesion_04":
            muestra, img_proc = preparar_captura_alineada(reg, detector, hashes_por_particion)
            val_auth_todos.append(muestra)
            if img_proc is not None:
                imagenes_autorizadas[reg.identificador] = img_proc

    val_auth_proc = [m for m in val_auth_todos if m["estado"] == "procesada"]
    print(f"Autorizados val procesados (Sesión 04): {len(val_auth_proc)} / {len(val_auth_todos)}")

    # 2. Desconocidos de validación (100 LFW calibración exclusivamente)
    X_val_desc_tex, X_val_desc_img, filas_desc_val, estados_desc_val = cargar_desconocidos_calibracion_alineados(detector, extractor)
    print(f"Desconocidos val procesados (Calibración): {len(X_val_desc_img)} / {len(filas_desc_val)}")

    return {
        "val_auth_todos": val_auth_todos,
        "val_auth_proc_ids": [m["id"] for m in val_auth_proc],
        "imagenes_autorizadas": imagenes_autorizadas,
        "filas_desc_val": filas_desc_val,
        "estados_desc_val": estados_desc_val,
        "X_val_desc_img": X_val_desc_img,
        "X_val_desc_tex": X_val_desc_tex,
    }


def regenerar_comparacion_desde_guardados(dir_experimento: Optional[Path] = None) -> Path:
    if dir_experimento is None or str(dir_experimento).strip() == "":
        dirs = sorted([d for d in (RAIZ / "resultados").glob("desarrollo_4clases_alineado_*") if d.is_dir()])
        if not dirs:
            raise FileNotFoundError("No se encontró ningún directorio de desarrollo_4clases_alineado_* en resultados/")
        dir_salida = dirs[-1].resolve()
    else:
        dir_salida = Path(dir_experimento).resolve()

    print("=" * 70)
    print(" REGENERANDO COMPARACIÓN DESDE MODELOS GUARDADOS (SIN REENTRENAR)")
    print(f" Directorio de modelos: {dir_salida}")
    print("=" * 70)

    # 1. Cargar modelos guardados
    ext_modelos = {
        "knn": "modelo.joblib",
        "arbol": "modelo.joblib",
        "svm": "modelo.joblib",
        "ann": "modelo.pt",
        "cnn": "modelo.pt",
    }
    modelos = {
        "knn": ClasificadorKNN.cargar(dir_salida / "knn" / ext_modelos["knn"]),
        "arbol": ClasificadorArbol.cargar(dir_salida / "arbol" / ext_modelos["arbol"]),
        "svm": ClasificadorSVM.cargar(dir_salida / "svm" / ext_modelos["svm"]),
        "ann": ClasificadorANN.cargar(dir_salida / "ann" / ext_modelos["ann"]),
        "cnn": ClasificadorCNN.cargar(dir_salida / "cnn" / ext_modelos["cnn"]),
    }

    # 2. Cargar metadatos previos para preservar conteos y métricas históricas
    ruta_inf_previa = dir_salida / "informe_experimento.json"
    if ruta_inf_previa.exists():
        with open(ruta_inf_previa, "r", encoding="utf-8") as f:
            inf_prev = json.load(f)
        metadatos_modelos = inf_prev.get("metadatos_modelos", {})
        conteo_aumentos = inf_prev.get("conteo_aumentos", {})
        latencias = inf_prev.get("latencias", {})
        # Preservar informe original antes de actualizar
        ruta_bkp_inf = dir_salida / "informe_experimento_original.json"
        if not ruta_bkp_inf.exists():
            with open(ruta_bkp_inf, "w", encoding="utf-8") as f_bkp:
                json.dump(inf_prev, f_bkp, indent=2, ensure_ascii=False)
    else:
        metadatos_modelos = {}
        conteo_aumentos = {}
        latencias = {}

    ruta_tabla_previa = dir_salida / "TABLA_COMPARATIVA.md"
    if ruta_tabla_previa.exists():
        ruta_bkp_tabla = dir_salida / "TABLA_COMPARATIVA_original.md"
        if not ruta_bkp_tabla.exists():
            ruta_bkp_tabla.write_text(ruta_tabla_previa.read_text(encoding="utf-8"), encoding="utf-8")

    # 3. Cargar datos de validación de desarrollo exclusivamente (Sesión 04 + Calibración LFW)
    datos_val = preparar_datos_validacion_desarrollo()

    # 4. Evaluar modelos con la lógica corregida de umbrales
    evaluaciones = {}
    for nombre, modelo in modelos.items():
        print(f"\n--- Evaluando {nombre.upper()} ---")
        ev = evaluar_modelo_reglas(nombre, modelo, datos_val)
        evaluaciones[nombre] = ev
        if ev["tiene_umbral_viable"] and ev["regla_b"] is not None:
            rb = ev["regla_b"]
            print(f"  Estado:              {ev['estado']}")
            print(f"  Umbral seleccionado: {ev['umbral_optimo']}")
            print(f"  FAR Desconocidos:    {rb['far_total_pct']}% ({rb['far_conteo']})")
            print(f"  FRR Puerta Global:   {rb['frr_puerta_pct']}% ({rb['frr_conteo']})")
            print(f"  Aperturas Correctas: {rb['aperturas_correctas']} / 64")
            print(f"  Confusiones Auth:    {rb['confusiones_autorizados']} / 64")
            print(f"  FRR por Persona:     {rb['frr_por_persona']}")
        else:
            ra = ev["regla_a"]
            print(f"  Estado:              {ev['estado']}")
            print(f"  Umbral recomendado:  {ev['umbral_optimo']} (Regla B deshabilitada)")
            print(f"  Regla A - FAR Desc:  {ra['far_total_pct']}% ({ra['far_conteo']})")
            print(f"  Regla A - FRR Global:{ra['frr_puerta_pct']}% ({ra['frr_conteo']})")
            print(f"  Regla A - FRR por P: {ra['frr_por_persona']}")

    # 5. Si no había latencias previas, medirlas
    if not latencias:
        latencias = medir_latencias_componentes(modelos["cnn"], datos_val)

    # 6. Generar TABLA_COMPARATIVA.md corregida
    texto_tabla = generar_tabla_comparativa(evaluaciones, metadatos_modelos)
    with open(dir_salida / "TABLA_COMPARATIVA.md", "w", encoding="utf-8") as f:
        f.write(texto_tabla)

    # 7. Actualizar informe_experimento.json
    informe_completo = {
        "fecha": datetime.now().isoformat(),
        "directorio": str(dir_salida),
        "conteo_aumentos": conteo_aumentos,
        "metadatos_modelos": metadatos_modelos,
        "evaluaciones": {
            k: {
                "estado": v["estado"],
                "umbral_optimo": v["umbral_optimo"],
                "umbral_recomendado": v["umbral_recomendado"],
                "tiene_umbral_viable": v["tiene_umbral_viable"],
                "regla_b_habilitada": v["regla_b_habilitada"],
                "regla_a": v["regla_a"],
                "regla_b": v["regla_b"],
                "curva_umbrales": v["curva_umbrales"],
            }
            for k, v in evaluaciones.items()
        },
        "latencias": latencias,
    }
    with open(dir_salida / "informe_experimento.json", "w", encoding="utf-8") as f:
        json.dump(informe_completo, f, indent=2, ensure_ascii=False)

    # 8. Exportar candidato congelado SOLO SI es viable
    exportar_candidato_congelado_si_viable(evaluaciones, dir_salida)

    print(f"\nComparación regenerada exitosamente en: {dir_salida}")
    print(f"Tabla comparativa actualizada: {dir_salida / 'TABLA_COMPARATIVA.md'}")
    return dir_salida


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experimento de alineación facial y multi-orientación (4 clases)")
    parser.add_argument("--regenerar-comparacion", nargs="?", const="", default=None,
                        help="Regenera tabla comparativa e informe desde modelos guardados sin reentrenar")
    args = parser.parse_args()

    with threadpool_limits(limits=HILOS):
        if args.regenerar_comparacion is not None:
            dir_exp = Path(args.regenerar_comparacion) if args.regenerar_comparacion else None
            regenerar_comparacion_desde_guardados(dir_exp)
        else:
            ejecutar_experimento_completo()
