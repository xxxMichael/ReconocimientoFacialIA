"""
Nuevo experimento de reconocimiento facial de 4 clases (persona_1, persona_2, persona_3, desconocido)
incorporando desconocidos reales adicionales (5 voluntarios del portal + 18 identidades universitarias en
entrenamiento; 7 identidades universitarias + 100 LFW + Sesión 04 en validación).

Preserva el candidato alineado anterior y los modelos activos de la web sin sobreescribirlos.
Guarda todos los artefactos en resultados/exp_4clases_universidad_<timestamp>/.
"""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    calcular_hash_archivo,
)
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.clasificacion.regla_acceso import (
    evaluar_regla_acceso_completa,
    seleccionar_umbral_optimo_jerarquico,
)
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.preprocesamiento.alineador import preprocesar_rostro_alineado
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from scripts.calibrar_desarrollo import CALIBRACION_LFW
from scripts.entrenar_desarrollo import (
    BINS,
    CUADRICULA,
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
    calcular_tasa_pct,
    cargar_desconocidos_calibracion_alineados,
    cargar_desconocidos_entrenamiento_alineados,
    preparar_captura_alineada,
)

MANIFIESTO_UNIVERSIDAD = RAIZ / "data" / "universidad_desc" / "manifiesto_universidad.csv"
POSES_PORTAL_SELECCIONADAS = {
    ("frontal", "media"),
    ("izquierda", "media"),
    ("derecha", "media"),
}


def cargar_manifiesto_universidad_desarrollo(
    ruta_manifiesto: Path = MANIFIESTO_UNIVERSIDAD,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Carga exclusivamente las particiones 'entrenamiento' y 'validacion' del manifiesto universitario.
    Excluye estrictamente 'prueba_reservada'.
    """
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(f"No existe el manifiesto universitario: {ruta_manifiesto}")

    with ruta_manifiesto.open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))

    train_filas = []
    val_filas = []
    hashes_por_particion: Dict[str, str] = {}

    for fila in filas:
        part = fila["particion"].strip().lower()
        h = fila["hash_sha256"].strip().lower()
        if part in hashes_por_particion.values():
            pass
        if h in hashes_por_particion and hashes_por_particion[h] != part:
            raise ValueError(f"Solapamiento de hash entre particiones: {fila['identificador']}")
        hashes_por_particion[h] = part

        if part == "entrenamiento":
            train_filas.append(fila)
        elif part == "validacion":
            val_filas.append(fila)
        elif part == "prueba_reservada":
            # Estrictamente excluida de desarrollo
            continue
        else:
            raise ValueError(f"Partición desconocida en manifiesto universitario: {part}")

    return train_filas, val_filas


def procesar_muestras_universidad(
    filas: List[Dict[str, str]],
    detector: DetectorYuNet,
    extractor: ExtractorHistogramasRegionales,
    es_entrenamiento: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    """Procesa imágenes universitarias con detección YuNet y alineación 5-puntos."""
    imagenes_64 = []
    texturas_256 = []
    registros_proc = []

    for fila in filas:
        ruta_img = RAIZ / fila["ruta"]
        contenido = ruta_img.read_bytes()
        digest = hashlib.sha256(contenido).hexdigest()
        if digest != fila["hash_sha256"]:
            raise ValueError(f"SHA-256 modificado en {fila['identificador']}")

        img_bgr = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            if es_entrenamiento:
                raise ValueError(f"No se pudo decodificar {ruta_img}")
            registros_proc.append({
                "id": fila["identificador"],
                "persona": fila["identificador"],
                "fuente": "universidad",
                "particion": fila["particion"],
                "procesada": False,
                "estado_deteccion": "error_decodificacion",
                "img_64": None,
                "tex_256": None,
                "ruta": fila["ruta"],
                "sha256": digest,
            })
            continue

        det = detector.detectar(img_bgr)
        if det.estado != EstadoDeteccion.UN_ROSTRO:
            if es_entrenamiento:
                # En entrenamiento, si no detecta rostro con 0.6, intentamos registrar fallo transparente
                registros_proc.append({
                    "id": fila["identificador"],
                    "persona": fila["identificador"],
                    "fuente": "universidad",
                    "particion": fila["particion"],
                    "procesada": False,
                    "estado_deteccion": det.estado.value,
                    "img_64": None,
                    "tex_256": None,
                    "ruta": fila["ruta"],
                    "sha256": digest,
                })
                continue
            else:
                registros_proc.append({
                    "id": fila["identificador"],
                    "persona": fila["identificador"],
                    "fuente": "universidad",
                    "particion": fila["particion"],
                    "procesada": False,
                    "estado_deteccion": det.estado.value,
                    "img_64": None,
                    "tex_256": None,
                    "ruta": fila["ruta"],
                    "sha256": digest,
                })
                continue

        img_proc, meta = preprocesar_rostro_alineado(
            imagen=img_bgr,
            puntos_faciales=det.puntos_faciales,
            recuadro=det.recuadro,
            resolucion=RESOLUCION,
            sigma=SIGMA,
        )
        feat = extractor.extraer(img_proc).vector_caracteristicas

        imagenes_64.append(img_proc)
        texturas_256.append(feat)
        registros_proc.append({
            "id": fila["identificador"],
            "persona": fila["identificador"],
            "fuente": "universidad",
            "particion": fila["particion"],
            "procesada": True,
            "estado_deteccion": det.estado.value,
            "alineado": meta.alineado,
            "img_64": img_proc,
            "tex_256": feat,
            "ruta": fila["ruta"],
            "sha256": digest,
        })

    return texturas_256, imagenes_64, registros_proc


def cargar_desconocidos_portal_entrenamiento(
    detector: DetectorYuNet,
    extractor: ExtractorHistogramasRegionales,
    cap_por_identidad: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Carga las capturas de las 5 personas del portal (david, externo_piloto, paul, sonia, stevenl)
    desde data/propio/manifiesto_local.csv para entrenamiento de esta versión.
    Controla la contribución por identidad seleccionando máximo `cap_por_identidad` fotos
    estratificadas por pose/distancia (frontal/media, izquierda/media, derecha/media)
    para evitar que 5 personas con 15 fotos dominen sobre las 18 personas universitarias.
    """
    ruta_manifiesto = RAIZ / "data" / "propio" / "manifiesto_local.csv"
    manifiesto = ManifiestoDataset(ruta_manifiesto)
    regs_portal = [
        r for r in manifiesto.registros
        if r.procedencia == "portal_web_externo"
    ]

    por_sujeto: Dict[str, List[RegistroCaptura]] = {}
    for r in regs_portal:
        por_sujeto.setdefault(r.persona, []).append(r)

    imagenes_64 = []
    texturas_256 = []
    metadatos = []

    for sujeto in sorted(por_sujeto.keys()):
        candidatos = sorted(
            por_sujeto[sujeto],
            key=lambda x: (0 if (x.pose, x.distancia) in POSES_PORTAL_SELECCIONADAS else 1, x.pose, x.distancia or "")
        )
        seleccionados = candidatos[:cap_por_identidad]

        for reg in seleccionados:
            ruta_img = RAIZ / reg.ruta
            contenido = ruta_img.read_bytes()
            digest = hashlib.sha256(contenido).hexdigest()
            img_bgr = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            det = detector.detectar(img_bgr)
            if det.estado != EstadoDeteccion.UN_ROSTRO:
                continue

            img_proc, meta = preprocesar_rostro_alineado(
                imagen=img_bgr,
                puntos_faciales=det.puntos_faciales,
                recuadro=det.recuadro,
                resolucion=RESOLUCION,
                sigma=SIGMA,
            )
            feat = extractor.extraer(img_proc).vector_caracteristicas
            imagenes_64.append(img_proc)
            texturas_256.append(feat)
            metadatos.append({
                "id": reg.identificador,
                "identidad": sujeto,
                "pose": reg.pose,
                "distancia": reg.distancia,
                "fuente": "portal_desarrollo",
                "ruta": reg.ruta,
                "sha256": digest,
                "alineado": meta.alineado,
            })

    return texturas_256, imagenes_64, metadatos


def preparar_datasets_experimento_universidad() -> Dict[str, Any]:
    print("=" * 75)
    print(" PREPARANDO DATASETS: EXPERIMENTO 4 CLASES + UNIVERSIDAD")
    print("=" * 75)

    detector = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)

    # 1. Autorizados (Entrenamiento + Sesión 04 Validación)
    print("[1/5] Cargando autorizados (Entrenamiento y Sesión 04)...")
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

    print(f"  Autorizados train: {len(train_auth)} | val (Sesión 04): {len(val_auth_proc)}/{len(val_auth_todos)}")

    # 2. Desconocidos LFW (300 entrenamiento + 100 calibración)
    print("[2/5] Cargando desconocidos LFW (300 train + 100 val)...")
    X_lfw_train_tex, X_lfw_train_img, meta_lfw_train = cargar_desconocidos_entrenamiento_alineados(detector, extractor)
    X_lfw_val_tex, X_lfw_val_img, filas_lfw_val, estados_lfw_val = cargar_desconocidos_calibracion_alineados(detector, extractor)

    # 3. Desconocidos Portal (5 personas -> cap 3 fotos/identidad = 15 muestras)
    print("[3/5] Cargando 5 personas del portal (cap 3 fotos/identidad para control de contribución)...")
    X_portal_tex, X_portal_img, meta_portal_train = cargar_desconocidos_portal_entrenamiento(detector, extractor, cap_por_identidad=3)
    print(f"  Portal procesadas: {len(meta_portal_train)} muestras de {len({m['identidad'] for m in meta_portal_train})} identidades")

    # 4. Desconocidos Universidad (18 train + 7 val; 9 prueba_reservada excluidas)
    print("[4/5] Cargando desconocidos universitarios (18 train + 7 val; reserva excluida)...")
    filas_univ_train, filas_univ_val = cargar_manifiesto_universidad_desarrollo()
    X_univ_train_tex, X_univ_train_img, meta_univ_train = procesar_muestras_universidad(
        filas_univ_train, detector, extractor, es_entrenamiento=True
    )
    X_univ_val_tex, X_univ_val_img, meta_univ_val = procesar_muestras_universidad(
        filas_univ_val, detector, extractor, es_entrenamiento=False
    )
    univ_train_proc = [m for m in meta_univ_train if m["procesada"]]
    univ_val_proc = [m for m in meta_univ_val if m["procesada"]]
    print(f"  Universidad train procesadas: {len(univ_train_proc)}/{len(filas_univ_train)}")
    print(f"  Universidad val procesadas:   {len(univ_val_proc)}/{len(filas_univ_val)}")

    # 5. Aumentos EXCLUSIVAMENTE en entrenamiento y registro detallado
    print("[5/5] Aplicando aumentos controlados exclusivamente en entrenamiento...")
    rng = np.random.RandomState(SEMILLA)

    registro_composicion = {
        "autorizado_persona_1": {"original": 0, "aumentadas": 0, "total": 0, "identidades": 1},
        "autorizado_persona_2": {"original": 0, "aumentadas": 0, "total": 0, "identidades": 1},
        "autorizado_persona_3": {"original": 0, "aumentadas": 0, "total": 0, "identidades": 1},
        "desconocido_lfw": {"original": 0, "aumentadas": 0, "total": 0, "identidades": 300},
        "desconocido_portal": {"original": 0, "aumentadas": 0, "total": 0, "identidades": len({m["identidad"] for m in meta_portal_train})},
        "desconocido_universidad": {"original": 0, "aumentadas": 0, "total": 0, "identidades": len(univ_train_proc)},
    }

    X_train_img_list = []
    X_train_tex_list = []
    y_train_list = []

    # 5a. Autorizados (1 orig + 2 aug = 3x por imagen)
    for m in train_auth:
        persona = m["persona"]
        cid = CLASE_TO_ID[persona]
        img_orig = imagenes_autorizadas[m["id"]]
        tex_orig = extractor.extraer(img_orig).vector_caracteristicas

        X_train_img_list.append(img_orig)
        X_train_tex_list.append(tex_orig)
        y_train_list.append(cid)
        registro_composicion[f"autorizado_{persona}"]["original"] += 1

        for aum in aumentar_muestra_entrenamiento(img_orig, rng):
            X_train_img_list.append(aum)
            X_train_tex_list.append(extractor.extraer(aum).vector_caracteristicas)
            y_train_list.append(cid)
            registro_composicion[f"autorizado_{persona}"]["aumentadas"] += 1

    cid_desc = CLASE_TO_ID["desconocido"]

    # 5b. LFW (300 orig, 0 aug -> 1 por identidad LFW)
    for i in range(len(X_lfw_train_img)):
        X_train_img_list.append(X_lfw_train_img[i])
        X_train_tex_list.append(X_lfw_train_tex[i])
        y_train_list.append(cid_desc)
        registro_composicion["desconocido_lfw"]["original"] += 1

    # 5c. Portal (hasta 3 orig por identidad, 0 aug -> 3 por identidad Portal)
    for i in range(len(X_portal_img)):
        X_train_img_list.append(X_portal_img[i])
        X_train_tex_list.append(X_portal_tex[i])
        y_train_list.append(cid_desc)
        registro_composicion["desconocido_portal"]["original"] += 1

    # 5d. Universidad train (1 orig + 2 aug = 3 por identidad Universidad)
    for i in range(len(X_univ_train_img)):
        img_orig = X_univ_train_img[i]
        tex_orig = X_univ_train_tex[i]

        X_train_img_list.append(img_orig)
        X_train_tex_list.append(tex_orig)
        y_train_list.append(cid_desc)
        registro_composicion["desconocido_universidad"]["original"] += 1

        for aum in aumentar_muestra_entrenamiento(img_orig, rng):
            X_train_img_list.append(aum)
            X_train_tex_list.append(extractor.extraer(aum).vector_caracteristicas)
            y_train_list.append(cid_desc)
            registro_composicion["desconocido_universidad"]["aumentadas"] += 1

    for k, v in registro_composicion.items():
        v["total"] = v["original"] + v["aumentadas"]

    X_train_img = np.asarray(X_train_img_list, dtype=np.float32)
    X_train_tex = np.asarray(X_train_tex_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)

    # Validación SIN aumentos
    X_val_auth_tex = np.stack([extractor.extraer(imagenes_autorizadas[m["id"]]).vector_caracteristicas for m in val_auth_proc])
    X_val_auth_img = np.stack([imagenes_autorizadas[m["id"]] for m in val_auth_proc])
    y_val_auth = np.array([CLASE_TO_ID[m["persona"]] for m in val_auth_proc], dtype=np.int64)

    X_val_univ_tex_arr = np.asarray(X_univ_val_tex, dtype=np.float32) if X_univ_val_tex else np.empty((0, X_train_tex.shape[1]), dtype=np.float32)
    X_val_univ_img_arr = np.asarray(X_univ_val_img, dtype=np.float32) if X_univ_val_img else np.empty((0, 64, 64), dtype=np.float32)

    X_val_tex = np.vstack([X_val_auth_tex, X_lfw_val_tex, X_val_univ_tex_arr])
    X_val_img = np.vstack([X_val_auth_img, X_lfw_val_img, X_val_univ_img_arr])
    y_val_desc_lfw = np.full(len(X_lfw_val_tex), cid_desc, dtype=np.int64)
    y_val_desc_univ = np.full(len(X_val_univ_tex_arr), cid_desc, dtype=np.int64)
    y_val = np.concatenate([y_val_auth, y_val_desc_lfw, y_val_desc_univ])

    counts = Counter(y_train)
    total_samples = len(y_train)
    pesos_clases = np.array([total_samples / (N_CLASES * counts[c]) for c in range(N_CLASES)], dtype=np.float32)

    print("\nComposición final de Entrenamiento:")
    for fuente, info in registro_composicion.items():
        print(f"  {fuente:24s}: {info['original']:3d} orig + {info['aumentadas']:3d} aum = {info['total']:3d} ({info['identidades']} ids)")
    print(f"Total Train: {len(y_train)} | Pesos clase: {pesos_clases}")
    print(f"Total Val procesados: {len(y_val)} ({len(val_auth_proc)} auth + {len(X_lfw_val_tex)} LFW + {len(X_val_univ_tex_arr)} Univ)")

    return {
        "X_train_tex": X_train_tex,
        "X_train_img": X_train_img,
        "y_train": y_train,
        "X_val_tex": X_val_tex,
        "X_val_img": X_val_img,
        "y_val": y_val,
        "pesos_clases": pesos_clases,
        "registro_composicion": registro_composicion,
        "val_auth_todos": val_auth_todos,
        "val_auth_proc_ids": [m["id"] for m in val_auth_proc],
        "imagenes_autorizadas": imagenes_autorizadas,
        "filas_lfw_val": filas_lfw_val,
        "estados_lfw_val": estados_lfw_val,
        "X_lfw_val_img": X_lfw_val_img,
        "X_lfw_val_tex": X_lfw_val_tex,
        "meta_univ_val": meta_univ_val,
        "X_univ_val_img": X_val_univ_img_arr,
        "X_univ_val_tex": X_val_univ_tex_arr,
        "meta_portal_train": meta_portal_train,
        "meta_univ_train": meta_univ_train,
    }


def evaluar_modelo_sobre_validacion(
    nombre_modelo: str,
    modelo: Any,
    datos: Dict[str, Any],
    es_cnn: bool,
    umbral_fijo: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evalúa un modelo sobre las muestras exactas de validación:
    - 64 autorizados (Sesión 04)
    - 100 desconocidos LFW calibración
    - 7 desconocidos universitarios validación
    Si `umbral_fijo` es None, ejecuta búsqueda jerárquica de umbral con restricción FRR_puerta <= 30% por autorizado.
    """
    val_auth_todos = datos["val_auth_todos"]
    ids_auth_proc = datos["val_auth_proc_ids"]
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS)

    # 1. Inferencia Autorizados
    if es_cnn:
        X_auth = np.stack([datos["imagenes_autorizadas"][mid] for mid in ids_auth_proc])
    else:
        X_auth = np.stack([extractor.extraer(datos["imagenes_autorizadas"][mid]).vector_caracteristicas for mid in ids_auth_proc])
    probs_auth = modelo.predecir_probabilidades(X_auth)

    dict_pred_auth = {}
    for idx, mid in enumerate(ids_auth_proc):
        p_vec = probs_auth[idx]
        pred_c = int(np.argmax(p_vec))
        dict_pred_auth[mid] = {
            "pred_idx": pred_c,
            "persona_pred": ID_TO_CLASE[pred_c],
            "score": float(p_vec[pred_c]),
            "probs": [float(x) for x in p_vec],
        }

    # 2. Inferencia Desconocidos LFW
    X_lfw = datos["X_lfw_val_img"] if es_cnn else datos["X_lfw_val_tex"]
    probs_lfw = modelo.predecir_probabilidades(X_lfw) if len(X_lfw) > 0 else np.empty((0, N_CLASES))
    dict_pred_lfw = {}
    idx_lfw = 0
    for i, est in enumerate(datos["estados_lfw_val"]):
        if est != "UN_ROSTRO":
            dict_pred_lfw[i] = {"procesada": False, "estado_deteccion": est, "persona_pred": None, "score": None, "probs": None}
        else:
            p_vec = probs_lfw[idx_lfw]
            pred_c = int(np.argmax(p_vec))
            dict_pred_lfw[i] = {
                "procesada": True,
                "estado_deteccion": est,
                "persona_pred": ID_TO_CLASE[pred_c],
                "score": float(p_vec[pred_c]),
                "probs": [float(x) for x in p_vec],
            }
            idx_lfw += 1

    # 3. Inferencia Desconocidos Universidad
    X_univ = datos["X_univ_val_img"] if es_cnn else datos["X_univ_val_tex"]
    probs_univ = modelo.predecir_probabilidades(X_univ) if len(X_univ) > 0 else np.empty((0, N_CLASES))
    dict_pred_univ = {}
    idx_univ = 0
    for i, m in enumerate(datos["meta_univ_val"]):
        if not m["procesada"]:
            dict_pred_univ[i] = {"procesada": False, "estado_deteccion": m["estado_deteccion"], "persona_pred": None, "score": None, "probs": None}
        else:
            p_vec = probs_univ[idx_univ]
            pred_c = int(np.argmax(p_vec))
            dict_pred_univ[i] = {
                "procesada": True,
                "estado_deteccion": m["estado_deteccion"],
                "persona_pred": ID_TO_CLASE[pred_c],
                "score": float(p_vec[pred_c]),
                "probs": [float(x) for x in p_vec],
            }
            idx_univ += 1

    def evaluar_con_umbral(th_eval: Optional[float], nombre_regla: str, generar_decisiones: bool = False) -> Dict[str, Any]:
        aperturas_correctas = 0
        confusiones_autorizados = 0
        bloqueos_desconocido = 0
        bloqueos_umbral = 0
        bloqueos_deteccion = 0
        decisiones_ind = []

        conteo_por_persona = {
            p: {"intentos": 0, "aperturas_correctas": 0, "confusiones": 0, "bloqueos": 0, "frr_pct": None}
            for p in PERSONAS_AUTORIZADAS
        }

        for m in val_auth_todos:
            p_real = m["persona"]
            conteo_por_persona[p_real]["intentos"] += 1
            if m["id"] not in dict_pred_auth:
                bloqueos_deteccion += 1
                conteo_por_persona[p_real]["bloqueos"] += 1
                if generar_decisiones:
                    decisiones_ind.append({
                        "modelo": nombre_modelo, "fuente": "autorizado_sesion_04", "id": m["id"],
                        "clase_real": p_real, "estado_deteccion": m["estado_deteccion"],
                        "clase_pred": None, "score": None, "umbral": th_eval,
                        "decision": "CERRADA", "categoria": "fallo_deteccion"
                    })
                continue

            pred = dict_pred_auth[m["id"]]
            p_pred = pred["persona_pred"]
            score = pred["score"]

            dec = evaluar_regla_acceso_completa(
                estado_deteccion="UN_ROSTRO",
                procesamiento_exitoso=True,
                clase_predicha=p_pred,
                score=score,
                umbral=th_eval if th_eval is not None else 0.0,
                personas_autorizadas=PERSONAS_AUTORIZADAS,
            )

            if dec.estado_puerta == "ABIERTA":
                if p_pred == p_real:
                    aperturas_correctas += 1
                    conteo_por_persona[p_real]["aperturas_correctas"] += 1
                    cat = "apertura_correcta"
                else:
                    confusiones_autorizados += 1
                    conteo_por_persona[p_real]["confusiones"] += 1
                    cat = "confusion_autorizado"
            else:
                conteo_por_persona[p_real]["bloqueos"] += 1
                if p_pred == "desconocido":
                    bloqueos_desconocido += 1
                    cat = "bloqueo_desconocido"
                else:
                    bloqueos_umbral += 1
                    cat = "bloqueo_umbral"

            if generar_decisiones:
                decisiones_ind.append({
                    "modelo": nombre_modelo, "fuente": "autorizado_sesion_04", "id": m["id"],
                    "clase_real": p_real, "estado_deteccion": "UN_ROSTRO",
                    "clase_pred": p_pred, "score": round(score, 4), "umbral": th_eval,
                    "decision": dec.estado_puerta, "categoria": cat
                })

        total_auth = len(val_auth_todos)
        aperturas_totales = aperturas_correctas + confusiones_autorizados
        bloqueos_puerta = total_auth - aperturas_totales
        frr_puerta = calcular_tasa_pct(bloqueos_puerta, total_auth)
        for p, stats in conteo_por_persona.items():
            stats["frr_pct"] = calcular_tasa_pct(stats["bloqueos"], stats["intentos"])

        # Desconocidos por fuente
        def eval_desc_sub(dict_pred, lista_meta, nombre_fuente):
            fa = 0
            rech_desc = 0
            rech_umb = 0
            fallo_det = 0
            for idx_item in range(len(lista_meta)):
                info = dict_pred[idx_item]
                meta_item = lista_meta[idx_item]
                item_id = (
                    meta_item.get("identificador")
                    or meta_item.get("id")
                    or meta_item.get("identidad_normalizada")
                    or f"item_{idx_item}"
                )
                if not info["procesada"]:
                    fallo_det += 1
                    if generar_decisiones:
                        decisiones_ind.append({
                            "modelo": nombre_modelo, "fuente": nombre_fuente, "id": item_id,
                            "clase_real": "desconocido", "estado_deteccion": info["estado_deteccion"],
                            "clase_pred": None, "score": None, "umbral": th_eval,
                            "decision": "CERRADA", "categoria": "fallo_deteccion"
                        })
                    continue

                dec = evaluar_regla_acceso_completa(
                    estado_deteccion="UN_ROSTRO",
                    procesamiento_exitoso=True,
                    clase_predicha=info["persona_pred"],
                    score=info["score"],
                    umbral=th_eval if th_eval is not None else 0.0,
                    personas_autorizadas=PERSONAS_AUTORIZADAS,
                )
                if dec.estado_puerta == "ABIERTA":
                    fa += 1
                    cat = "falsa_aceptacion"
                elif info["persona_pred"] == "desconocido":
                    rech_desc += 1
                    cat = "rechazo_desconocido"
                else:
                    rech_umb += 1
                    cat = "rechazo_umbral"

                if generar_decisiones:
                    decisiones_ind.append({
                        "modelo": nombre_modelo, "fuente": nombre_fuente, "id": item_id,
                        "clase_real": "desconocido", "estado_deteccion": "UN_ROSTRO",
                        "clase_pred": info["persona_pred"], "score": round(info["score"], 4), "umbral": th_eval,
                        "decision": dec.estado_puerta, "categoria": cat
                    })

            tot = len(lista_meta)
            return {
                "intentos": tot,
                "falsas_aceptaciones": fa,
                "far_pct": calcular_tasa_pct(fa, tot),
                "rechazos_desconocido": rech_desc,
                "rechazos_umbral": rech_umb,
                "fallos_deteccion": fallo_det,
            }

        res_lfw = eval_desc_sub(dict_pred_lfw, datos["filas_lfw_val"], "desconocido_lfw_calibracion")
        res_univ = eval_desc_sub(dict_pred_univ, datos["meta_univ_val"], "desconocido_universidad_validacion")

        fa_global = res_lfw["falsas_aceptaciones"] + res_univ["falsas_aceptaciones"]
        tot_desc_global = res_lfw["intentos"] + res_univ["intentos"]

        return {
            "nombre_regla": nombre_regla,
            "umbral": th_eval,
            "far_total_pct": calcular_tasa_pct(fa_global, tot_desc_global),
            "far_conteo": f"{fa_global}/{tot_desc_global}",
            "far_lfw": res_lfw,
            "far_universidad": res_univ,
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
            "decisiones_individuales": decisiones_ind if generar_decisiones else None,
        }

    if umbral_fijo is not None:
        res_fijo = evaluar_con_umbral(umbral_fijo, f"Umbral Fijo ({umbral_fijo:.2f})", generar_decisiones=True)
        return {
            "estado": "umbral_fijo",
            "umbral_recomendado": umbral_fijo,
            "regla_b": res_fijo,
            "regla_b_habilitada": True,
        }

    # Búsqueda de umbral
    umbrales_candidatos = [round(th, 2) for th in np.arange(0.00, 1.01, 0.01)]
    curva_umbrales = []
    viable_umbrales = []

    for th in umbrales_candidatos:
        eval_th = evaluar_con_umbral(th, f"th={th:.2f}", generar_decisiones=False)
        frr_p = eval_th["frr_por_persona"]
        es_viable = all((frr_p[p] is not None and frr_p[p] <= 30.0) for p in PERSONAS_AUTORIZADAS)
        curva_umbrales.append({k: v for k, v in eval_th.items() if k != "decisiones_individuales"})
        if es_viable:
            viable_umbrales.append(eval_th)

    res_regla_a = evaluar_con_umbral(None, "Regla A (Sin Umbral)", generar_decisiones=False)

    if viable_umbrales:
        mejor = seleccionar_umbral_optimo_jerarquico(viable_umbrales)
        mejor_th = mejor["umbral"]
        res_regla_b = evaluar_con_umbral(mejor_th, f"Regla B (Umbral={mejor_th:.2f})", generar_decisiones=True)
        return {
            "estado": "viable",
            "umbral_recomendado": mejor_th,
            "regla_b_habilitada": True,
            "regla_a": {k: v for k, v in res_regla_a.items() if k != "decisiones_individuales"},
            "regla_b": res_regla_b,
            "curva_umbrales": curva_umbrales,
        }
    else:
        res_a_con_dec = evaluar_con_umbral(None, "Regla A (Sin Umbral)", generar_decisiones=True)
        return {
            "estado": "sin_umbral_viable",
            "umbral_recomendado": None,
            "regla_b_habilitada": False,
            "regla_a": res_a_con_dec,
            "regla_b": None,
            "curva_umbrales": curva_umbrales,
        }


def ejecutar_experimento_universidad():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"exp_4clases_universidad_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    print(f"Directorio de resultados del experimento: {dir_salida}")

    # 1. Preparar datasets
    datos = preparar_datasets_experimento_universidad()

    # 2. Entrenar los 5 modelos
    print("\n" + "=" * 75)
    print(" ENTRENANDO LOS CINCO MODELOS (ÁRBOL, KNN, SVM, ANN, CNN)")
    print("=" * 75)

    X_train_tex = datos["X_train_tex"]
    y_train = datos["y_train"]
    X_val_tex = datos["X_val_tex"]
    y_val = datos["y_val"]
    X_train_img = datos["X_train_img"]
    X_val_img = datos["X_val_img"]
    pesos_clases = datos["pesos_clases"]

    modelos = {}
    metadatos_modelos = {}

    # KNN
    mejor_knn_score = -1.0
    mejor_knn = None
    mejor_knn_cfg = None
    for k in (3, 5, 7, 9):
        for pond in ("distance", "uniform"):
            knn_c = ClasificadorKNN(k=k, metrica="euclidean", ponderacion=pond)
            knn_c.entrenar(X_train_tex, y_train)
            sc = calcular_macro_recall(y_val, knn_c.predecir(X_val_tex))
            if sc > mejor_knn_score:
                mejor_knn_score = sc
                mejor_knn = knn_c
                mejor_knn_cfg = {"k": k, "ponderacion": pond, "macro_recall_val": round(sc, 2)}
    modelos["knn"] = mejor_knn
    metadatos_modelos["knn"] = mejor_knn_cfg
    print(f"  [KNN]   {mejor_knn_cfg}")

    # Árbol
    arbol, _, cfg_arbol = ClasificadorArbol.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight="balanced", metrica="macro_recall", semilla=SEMILLA
    )
    cfg_arbol["macro_recall_val"] = round(calcular_macro_recall(y_val, arbol.predecir(X_val_tex)), 2)
    modelos["arbol"] = arbol
    metadatos_modelos["arbol"] = cfg_arbol
    print(f"  [Árbol] {cfg_arbol}")

    # SVM
    svm, _, cfg_svm = ClasificadorSVM.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight="balanced", metrica="macro_recall", semilla=SEMILLA
    )
    cfg_svm["macro_recall_val"] = round(calcular_macro_recall(y_val, svm.predecir(X_val_tex)), 2)
    modelos["svm"] = svm
    metadatos_modelos["svm"] = cfg_svm
    print(f"  [SVM]   {cfg_svm}")

    # ANN
    ann = ClasificadorANN(n_entrada=X_train_tex.shape[1], n_clases=N_CLASES, semilla=SEMILLA)
    ann.entrenar_con_validacion(
        X_train_tex, y_train, X_val_tex, y_val,
        pesos_clases=pesos_clases, metrica_checkpoint="macro_recall"
    )
    metadatos_modelos["ann"] = {
        "mejor_epoca": ann.mejor_epoca_,
        "macro_recall_val": round(calcular_macro_recall(y_val, ann.predecir(X_val_tex)), 2),
    }
    modelos["ann"] = ann
    print(f"  [ANN]   {metadatos_modelos['ann']}")

    # CNN
    cnn = ClasificadorCNN(n_clases=N_CLASES, semilla=SEMILLA)
    cnn.entrenar_con_validacion(
        X_train_img, y_train, X_val_img, y_val,
        pesos_clases=pesos_clases, metrica_checkpoint="macro_recall"
    )
    metadatos_modelos["cnn"] = {
        "mejor_epoca": cnn.mejor_epoca_,
        "macro_recall_val": round(calcular_macro_recall(y_val, cnn.predecir(X_val_img)), 2),
    }
    modelos["cnn"] = cnn
    print(f"  [CNN]   {metadatos_modelos['cnn']}")

    # Guardar los 5 modelos en subcarpetas separadas
    ext_modelos = {
        "knn": "modelo.joblib",
        "arbol": "modelo.joblib",
        "svm": "modelo.joblib",
        "ann": "modelo.pt",
        "cnn": "modelo.pt",
    }
    rutas_modelos_guardados = {}
    for nombre_m, mod in modelos.items():
        sub = dir_salida / nombre_m
        sub.mkdir(parents=True, exist_ok=True)
        archivo_modelo = sub / ext_modelos.get(nombre_m, "modelo.bin")
        mod.guardar(archivo_modelo)
        rutas_modelos_guardados[nombre_m] = archivo_modelo

    # Guardar mapeo de clases, autorizados e historiales
    clases_mapeo = {
        "id_to_persona": ID_TO_CLASE,
        "persona_to_id": CLASE_TO_ID,
        "personas_autorizadas": PERSONAS_AUTORIZADAS,
    }
    (dir_salida / "clases_mapeo.json").write_text(json.dumps(clases_mapeo, indent=2, ensure_ascii=False), encoding="utf-8")
    (dir_salida / "personas_autorizadas.json").write_text(json.dumps(PERSONAS_AUTORIZADAS, indent=2, ensure_ascii=False), encoding="utf-8")

    historiales = {
        "ann": getattr(ann, "historial_", None),
        "cnn": getattr(cnn, "historial_", None),
        "metadatos": metadatos_modelos,
    }
    (dir_salida / "historiales_entrenamiento.json").write_text(json.dumps(historiales, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Copiar manifiesto universitario utilizado
    if MANIFIESTO_UNIVERSIDAD.exists():
        (dir_salida / "manifiesto_universidad_usado.csv").write_text(
            MANIFIESTO_UNIVERSIDAD.read_text(encoding="utf-8"), encoding="utf-8"
        )

    # 3. Calibrar los 5 nuevos modelos en Validación
    print("\n" + "=" * 75)
    print(" CALIBRACIÓN DE LOS CINCO NUEVOS MODELOS EN VALIDACIÓN")
    print("=" * 75)
    resultados_calibracion = {}
    todas_decisiones_csv = []

    for nombre_m in ("arbol", "knn", "svm", "ann", "cnn"):
        res = evaluar_modelo_sobre_validacion(
            nombre_modelo=f"nuevo_{nombre_m}",
            modelo=modelos[nombre_m],
            datos=datos,
            es_cnn=(nombre_m == "cnn"),
            umbral_fijo=None,
        )
        resultados_calibracion[nombre_m] = res
        dec_list = (res["regla_b"] or res["regla_a"]).get("decisiones_individuales") or []
        todas_decisiones_csv.extend(dec_list)

        if res["regla_b_habilitada"]:
            rb = res["regla_b"]
            print(
                f"  {nombre_m.upper():5s}: VIABLE th={res['umbral_recomendado']:.2f} | "
                f"FRR_puerta={rb['frr_puerta_pct']}% ({rb['frr_conteo']}) | "
                f"FAR_global={rb['far_total_pct']}% ({rb['far_conteo']}) | "
                f"FAR_LFW={rb['far_lfw']['far_pct']}% ({rb['far_lfw']['falsas_aceptaciones']}/100) | "
                f"FAR_Univ={rb['far_universidad']['far_pct']}% ({rb['far_universidad']['falsas_aceptaciones']}/7)"
            )
        else:
            ra = res["regla_a"]
            print(
                f"  {nombre_m.upper():5s}: sin_umbral_viable (Regla B deshabilitada) | "
                f"Regla A FAR_global={ra['far_total_pct']}% | FRR={ra['frr_puerta_pct']}%"
            )

    # 4. Evaluar Candidato Alineado Anterior (th=0.84) sobre las MISMAS muestras de validación
    print("\n" + "-" * 75)
    print(" REFERENCIA: CANDIDATO ALINEADO ANTERIOR (th=0.84)")
    print("-" * 75)
    pipe_ant = PipelineCandidatoCongelado(RAIZ / "configuracion_candidato_alineado.json", raiz_proyecto=RAIZ)
    res_anterior = evaluar_modelo_sobre_validacion(
        nombre_modelo="anterior_cnn_alineado_084",
        modelo=pipe_ant.clasificador,
        datos=datos,
        es_cnn=True,
        umbral_fijo=pipe_ant.umbral_score,
    )
    rb_ant = res_anterior["regla_b"]
    todas_decisiones_csv.extend(rb_ant.get("decisiones_individuales") or [])
    print(
        f"  ANT_CNN: th=0.84 | "
        f"FRR_puerta={rb_ant['frr_puerta_pct']}% ({rb_ant['frr_conteo']}) | "
        f"FAR_global={rb_ant['far_total_pct']}% ({rb_ant['far_conteo']}) | "
        f"FAR_LFW={rb_ant['far_lfw']['far_pct']}% ({rb_ant['far_lfw']['falsas_aceptaciones']}/100) | "
        f"FAR_Univ={rb_ant['far_universidad']['far_pct']}% ({rb_ant['far_universidad']['falsas_aceptaciones']}/7)"
    )

    # Guardar decisiones individuales CSV
    ruta_dec_csv = dir_salida / "decisiones_individuales_validacion.csv"
    if todas_decisiones_csv:
        with open(ruta_dec_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(todas_decisiones_csv[0].keys()))
            writer.writeheader()
            writer.writerows(todas_decisiones_csv)

    # Guardar curvas sin listas gigantes de decisiones
    curvas_para_json = {}
    for k, v in resultados_calibracion.items():
        v_copy = dict(v)
        if v_copy.get("regla_b"):
            v_copy["regla_b"] = {kk: vv for kk, vv in v_copy["regla_b"].items() if kk != "decisiones_individuales"}
        if v_copy.get("regla_a"):
            v_copy["regla_a"] = {kk: vv for kk, vv in v_copy["regla_a"].items() if kk != "decisiones_individuales"}
        curvas_para_json[k] = v_copy

    res_ant_limpio = {
        "estado": res_anterior["estado"],
        "umbral_recomendado": res_anterior["umbral_recomendado"],
        "regla_b": {kk: vv for kk, vv in rb_ant.items() if kk != "decisiones_individuales"},
    }

    # Elegir mejor candidato viable entre los 5 nuevos modelos según Regla B
    viables_nuevos = [
        (nm, resultados_calibracion[nm]["regla_b"])
        for nm in ("arbol", "knn", "svm", "ann", "cnn")
        if resultados_calibracion[nm]["regla_b_habilitada"]
    ]
    mejor_nuevo_nombre = None
    if viables_nuevos:
        viables_nuevos.sort(
            key=lambda x: (
                x[1]["far_total_pct"],
                x[1]["frr_puerta_pct"],
                -x[1]["aperturas_correctas"],
                -x[1]["umbral"],
            )
        )
        mejor_nuevo_nombre = viables_nuevos[0][0]

    resumen_final = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "semilla_entrenamiento": SEMILLA,
        "composicion_entrenamiento": datos["registro_composicion"],
        "metadatos_modelos": metadatos_modelos,
        "calibracion_nuevos": curvas_para_json,
        "referencia_anterior_alineado": res_ant_limpio,
        "mejor_candidato_nuevo": mejor_nuevo_nombre,
        "nota_independencia_portal": "Las 5 personas del portal (david, externo_piloto, paul, sonia, stevenl) forman parte del entrenamiento de esta versión; ya no son prueba independiente.",
        "nota_muestra_universidad_validacion": "Las 7 imágenes universitarias de validación constituyen una muestra pequeña; su porcentaje de rechazo debe interpretarse junto con LFW (100) y las 9 reservadas.",
    }

    with open(dir_salida / "resumen_experimento_universidad.json", "w", encoding="utf-8") as f:
        json.dump(resumen_final, f, indent=2, ensure_ascii=False)

    # Guardar tabla comparativa CSV
    filas_tabla = []
    for nm in ("arbol", "knn", "svm", "ann", "cnn"):
        cal = resultados_calibracion[nm]
        regla = cal["regla_b"] if cal["regla_b_habilitada"] else cal["regla_a"]
        filas_tabla.append({
            "modelo": f"nuevo_{nm}",
            "estado_calibracion": cal["estado"],
            "umbral_recomendado": cal["umbral_recomendado"] if cal["umbral_recomendado"] is not None else "null",
            "regla_b_habilitada": cal["regla_b_habilitada"],
            "aperturas_correctas": regla["aperturas_correctas"],
            "confusiones_autorizados": regla["confusiones_autorizados"],
            "frr_puerta_pct": regla["frr_puerta_pct"],
            "frr_conteo": regla["frr_conteo"],
            "far_global_pct": regla["far_total_pct"],
            "far_global_conteo": regla["far_conteo"],
            "far_lfw_pct": regla["far_lfw"]["far_pct"],
            "far_lfw_conteo": f"{regla['far_lfw']['falsas_aceptaciones']}/{regla['far_lfw']['intentos']}",
            "far_universidad_pct": regla["far_universidad"]["far_pct"],
            "far_universidad_conteo": f"{regla['far_universidad']['falsas_aceptaciones']}/{regla['far_universidad']['intentos']}",
        })
    filas_tabla.append({
        "modelo": "anterior_cnn_alineado_084",
        "estado_calibracion": "referencia_congelada",
        "umbral_recomendado": 0.84,
        "regla_b_habilitada": True,
        "aperturas_correctas": rb_ant["aperturas_correctas"],
        "confusiones_autorizados": rb_ant["confusiones_autorizados"],
        "frr_puerta_pct": rb_ant["frr_puerta_pct"],
        "frr_conteo": rb_ant["frr_conteo"],
        "far_global_pct": rb_ant["far_total_pct"],
        "far_global_conteo": rb_ant["far_conteo"],
        "far_lfw_pct": rb_ant["far_lfw"]["far_pct"],
        "far_lfw_conteo": f"{rb_ant['far_lfw']['falsas_aceptaciones']}/{rb_ant['far_lfw']['intentos']}",
        "far_universidad_pct": rb_ant["far_universidad"]["far_pct"],
        "far_universidad_conteo": f"{rb_ant['far_universidad']['falsas_aceptaciones']}/{rb_ant['far_universidad']['intentos']}",
    })
    with open(dir_salida / "tabla_comparativa.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_tabla[0].keys()))
        writer.writeheader()
        writer.writerows(filas_tabla)

    # Guardar configuración propuesta si existe candidato viable (sin reemplazar la web ni el candidato alineado anterior)
    if mejor_nuevo_nombre == "cnn":
        ruta_mod_mejor = rutas_modelos_guardados["cnn"]
        cfg_prop = {
            "candidato": {
                "identificador": "cnn_4clases_universidad_propuesto",
                "nombre": "CNN 4 Clases (Alineación 5-Puntos + Desconocidos Portal/Universidad)",
                "tipo_arquitectura": "CNN",
                "ruta_modelo": str(ruta_mod_mejor.relative_to(RAIZ)).replace("\\", "/"),
                "sha256_modelo": calcular_sha256(ruta_mod_mejor),
                "umbral_score": resultados_calibracion["cnn"]["umbral_recomendado"],
                "orden_clases": CLASES_NOMBRES,
                "mapeo_clases": {str(k): v for k, v in ID_TO_CLASE.items()},
                "personas_autorizadas": PERSONAS_AUTORIZADAS,
                "fecha_propuesta": datetime.now().isoformat(),
                "estado": "propuesto_desarrollo_sin_prueba_final",
            },
            "detector_facial": pipe_ant.config["detector_facial"],
            "preprocesamiento": pipe_ant.config["preprocesamiento"],
            "regla_acceso": pipe_ant.config["regla_acceso"],
        }
        (dir_salida / "configuracion_candidato_universidad_propuesto.json").write_text(
            json.dumps(cfg_prop, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Hashes SHA-256 de los artefactos
    hashes_art = {
        k: calcular_sha256(v) for k, v in rutas_modelos_guardados.items()
    }
    hashes_art["decisiones_individuales_validacion_csv"] = calcular_sha256(ruta_dec_csv)
    hashes_art["resumen_experimento_universidad_json"] = calcular_sha256(dir_salida / "resumen_experimento_universidad.json")
    (dir_salida / "hashes_sha256.json").write_text(json.dumps(hashes_art, indent=2), encoding="utf-8")

    print(f"\nExperimento guardado en: {dir_salida}")
    print(f"Mejor candidato nuevo en validación: {mejor_nuevo_nombre}")
    return resumen_final, dir_salida


if __name__ == "__main__":
    ejecutar_experimento_universidad()
