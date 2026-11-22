"""
Nueva iteración del experimento de 4 clases utilizando TODAS las condiciones disponibles
de las 75 capturas del portal (david, externo_piloto, paul, sonia, stevenl), junto con
autorizados de entrenamiento, 300 LFW y 18 identidades universitarias de entrenamiento.

Implementa control separado de:
1. Contribución de cada identidad dentro de la clase 'desconocido' (evitando que
   las 5 personas del portal con hasta 15 fotos dominen sobre identidades de 1 foto).
2. Balance global entre las cuatro clases (25% efectivo por clase).

Estrategia compatible por algoritmo:
- Árbol y SVM: sample_weight con class_weight=None (evitando doble compensación).
- ANN y CNN: pérdida individual ponderada (pesos_muestras) con pesos_clases=None.
- KNN: entrenado sobre el conjunto completo con cobertura íntegra de poses/distancias
  (documentando explícitamente que weights='distance' no es balanceo de clases).

Preserva modelos anteriores, modelo activo web y las 9 identidades universitarias reservadas.
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
from reconocimiento.clasificacion.svm import ClasificadorSVM
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
    POSES_PORTAL_SELECCIONADAS,
    cargar_manifiesto_universidad_desarrollo,
    evaluar_modelo_sobre_validacion,
    procesar_muestras_universidad,
)


def cargar_todas_capturas_portal(
    detector: DetectorYuNet,
    extractor: ExtractorHistogramasRegionales,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Procesa TODAS las 75 capturas disponibles del portal (todas las poses y distancias).
    Registra explícitamente cualquier fallo de detección sin inventar recortes.
    """
    ruta_manifiesto = RAIZ / "data" / "propio" / "manifiesto_local.csv"
    manifiesto = ManifiestoDataset(ruta_manifiesto)
    regs_portal = [
        r for r in manifiesto.registros
        if r.procedencia == "portal_web_externo" and r.particion == "prueba"
    ]
    regs_portal.sort(key=lambda r: (r.persona, r.pose, r.distancia or ""))

    imagenes_64 = []
    texturas_256 = []
    metadatos_validos = []
    registro_fallos = []

    for reg in regs_portal:
        ruta_img = RAIZ / reg.ruta
        if not ruta_img.exists():
            registro_fallos.append({
                "id": reg.identificador,
                "identidad": reg.persona,
                "pose": reg.pose,
                "distancia": reg.distancia,
                "motivo": "archivo_no_encontrado",
            })
            continue

        contenido = ruta_img.read_bytes()
        digest = hashlib.sha256(contenido).hexdigest()
        img_bgr = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            registro_fallos.append({
                "id": reg.identificador,
                "identidad": reg.persona,
                "pose": reg.pose,
                "distancia": reg.distancia,
                "motivo": "error_decodificacion",
            })
            continue

        det = detector.detectar(img_bgr)
        if det.estado != EstadoDeteccion.UN_ROSTRO:
            registro_fallos.append({
                "id": reg.identificador,
                "identidad": reg.persona,
                "pose": reg.pose,
                "distancia": reg.distancia,
                "motivo": f"fallo_deteccion_{det.estado.value}",
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
        metadatos_validos.append({
            "id": reg.identificador,
            "identidad": reg.persona,
            "pose": reg.pose,
            "distancia": reg.distancia,
            "fuente": "portal_completo",
            "ruta": reg.ruta,
            "sha256": digest,
            "alineado": meta.alineado,
        })

    return texturas_256, imagenes_64, metadatos_validos, registro_fallos


def calcular_pesos_identidad_y_clase(
    y_train: np.ndarray,
    identidades_train: List[str],
    fuentes_train: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calcula pesos por muestra controlando por separado:
    1. Contribución por identidad dentro de la clase 'desconocido':
       Cada identidad desconocida tiene peso total raw = 1.0 (w_id = 1 / n_muestras_identidad).
       Así, los aumentos o múltiples fotos de un sujeto del portal no multiplican su peso
       frente a identidades con una sola foto (LFW o universidad).
    2. Balance global entre las 4 clases:
       Cada clase recibe un peso efectivo total idéntico = N_train / 4 (25% del gradiente/partición),
       evitando aplicar 1/n_fotos a las 3 identidades autorizadas frente a >300 desconocidos.
    """
    n_total = len(y_train)
    cid_desc = CLASE_TO_ID["desconocido"]

    # Contar muestras por identidad desconocida
    conteo_id_desc = Counter(
        identidades_train[i] for i in range(n_total) if y_train[i] == cid_desc
    )

    pesos_raw = np.zeros(n_total, dtype=np.float64)
    for i in range(n_total):
        if y_train[i] == cid_desc:
            n_u = conteo_id_desc[identidades_train[i]]
            pesos_raw[i] = 1.0 / float(n_u)
        else:
            # Dentro de cada clase autorizada (1 sola identidad), todas las imágenes
            # tienen igual peso unitario (1 original + 2 aumentos por captura)
            pesos_raw[i] = 1.0

    # Escalar cada clase para que su suma de pesos efectivos sea exactamente n_total / 4
    pesos_finales = np.zeros(n_total, dtype=np.float32)
    objetivo_clase = float(n_total) / float(N_CLASES)

    resumen_clases = {}
    for c in range(N_CLASES):
        mask = (y_train == c)
        suma_raw_c = float(np.sum(pesos_raw[mask]))
        factor_c = objetivo_clase / suma_raw_c if suma_raw_c > 0 else 1.0
        pesos_finales[mask] = (pesos_raw[mask] * factor_c).astype(np.float32)
        resumen_clases[ID_TO_CLASE[c]] = {
            "muestras": int(np.sum(mask)),
            "peso_efectivo_total": round(float(np.sum(pesos_finales[mask])), 4),
            "proporcion_pct": round(100.0 * float(np.sum(pesos_finales[mask])) / float(n_total), 2),
        }

    # Resumen por fuente y por identidad portal
    resumen_fuentes = {}
    for f_name in sorted(set(fuentes_train)):
        idx_f = [i for i, f in enumerate(fuentes_train) if f == f_name]
        w_f = float(np.sum(pesos_finales[idx_f]))
        ids_f = len({identidades_train[i] for i in idx_f})
        resumen_fuentes[f_name] = {
            "muestras": len(idx_f),
            "identidades": ids_f,
            "peso_efectivo_total": round(w_f, 4),
            "peso_medio_por_muestra": round(w_f / len(idx_f), 4) if idx_f else 0.0,
            "peso_por_identidad": round(w_f / ids_f, 4) if ids_f else 0.0,
        }

    return pesos_finales, {
        "n_total": n_total,
        "clases": resumen_clases,
        "fuentes": resumen_fuentes,
    }


def preparar_datasets_portal_completo(dir_salida: Path) -> Dict[str, Any]:
    print("=" * 75)
    print(" PREPARANDO DATASETS: COBERTURA COMPLETA DEL PORTAL + BALANCE 2 NIVELES")
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

    # 2. Desconocidos LFW (300 train + 100 val)
    print("[2/5] Cargando desconocidos LFW (300 train + 100 val)...")
    X_lfw_train_tex, X_lfw_train_img, meta_lfw_train = cargar_desconocidos_entrenamiento_alineados(detector, extractor)
    X_lfw_val_tex, X_lfw_val_img, filas_lfw_val, estados_lfw_val = cargar_desconocidos_calibracion_alineados(detector, extractor)

    # 3. TODAS las 75 capturas del portal
    print("[3/5] Procesando TODAS las 75 capturas del portal (todas las poses y distancias)...")
    X_portal_tex, X_portal_img, meta_portal_validos, fallos_portal = cargar_todas_capturas_portal(detector, extractor)
    print(f"  Portal válidas: {len(meta_portal_validos)}/75 | Fallos detección: {len(fallos_portal)}")
    if fallos_portal:
        print(f"  Detalle fallos portal: {[f['id'] for f in fallos_portal]}")

    # 4. Desconocidos Universidad (18 train + 7 val; 9 prueba_reservada estrictamente excluidas)
    print("[4/5] Cargando desconocidos universitarios (18 train + 7 val; reserva excluida)...")
    filas_univ_train, filas_univ_val = cargar_manifiesto_universidad_desarrollo()
    X_univ_train_tex, X_univ_train_img, meta_univ_train = procesar_muestras_universidad(
        filas_univ_train, detector, extractor, es_entrenamiento=True
    )
    X_univ_val_tex, X_univ_val_img, meta_univ_val = procesar_muestras_universidad(
        filas_univ_val, detector, extractor, es_entrenamiento=False
    )
    fallos_univ_train = [m for m in meta_univ_train if not m["procesada"]]
    univ_train_proc = [m for m in meta_univ_train if m["procesada"]]
    print(f"  Univ train válidas: {len(univ_train_proc)}/{len(filas_univ_train)} | Fallos: {len(fallos_univ_train)}")

    # 5. Construir conjunto de entrenamiento con aumentos EXCLUSIVOS en entrenamiento
    print("[5/5] Aplicando aumentos controlados exclusivamente en entrenamiento...")
    rng = np.random.RandomState(SEMILLA)

    X_train_img_list = []
    X_train_tex_list = []
    y_train_list = []
    identidades_train: List[str] = []
    fuentes_train: List[str] = []
    manifiesto_exp_filas = []

    conteo_fuentes = {
        "autorizado_persona_1": {"original": 0, "aumentadas": 0, "total": 0},
        "autorizado_persona_2": {"original": 0, "aumentadas": 0, "total": 0},
        "autorizado_persona_3": {"original": 0, "aumentadas": 0, "total": 0},
        "desconocido_lfw": {"original": 0, "aumentadas": 0, "total": 0},
        "desconocido_portal": {"original": 0, "aumentadas": 0, "total": 0},
        "desconocido_universidad": {"original": 0, "aumentadas": 0, "total": 0},
    }

    # 5a. Autorizados (1 orig + 2 aug)
    for m in train_auth:
        persona = m["persona"]
        cid = CLASE_TO_ID[persona]
        img_orig = imagenes_autorizadas[m["id"]]
        tex_orig = extractor.extraer(img_orig).vector_caracteristicas

        X_train_img_list.append(img_orig)
        X_train_tex_list.append(tex_orig)
        y_train_list.append(cid)
        identidades_train.append(persona)
        fuentes_train.append(f"autorizado_{persona}")
        conteo_fuentes[f"autorizado_{persona}"]["original"] += 1
        manifiesto_exp_filas.append({"id": m["id"], "clase": persona, "identidad": persona, "fuente": f"autorizado_{persona}", "variante": "original", "particion": "entrenamiento"})

        for idx_a, aum in enumerate(aumentar_muestra_entrenamiento(img_orig, rng), 1):
            X_train_img_list.append(aum)
            X_train_tex_list.append(extractor.extraer(aum).vector_caracteristicas)
            y_train_list.append(cid)
            identidades_train.append(persona)
            fuentes_train.append(f"autorizado_{persona}")
            conteo_fuentes[f"autorizado_{persona}"]["aumentadas"] += 1
            manifiesto_exp_filas.append({"id": f"{m['id']}_aug{idx_a}", "clase": persona, "identidad": persona, "fuente": f"autorizado_{persona}", "variante": f"aumento_{idx_a}", "particion": "entrenamiento"})

    cid_desc = CLASE_TO_ID["desconocido"]

    # 5b. LFW (300 orig, 0 aug)
    for i, meta in enumerate(meta_lfw_train):
        X_train_img_list.append(X_lfw_train_img[i])
        X_train_tex_list.append(X_lfw_train_tex[i])
        y_train_list.append(cid_desc)
        identidades_train.append(meta["id"])
        fuentes_train.append("desconocido_lfw")
        conteo_fuentes["desconocido_lfw"]["original"] += 1
        manifiesto_exp_filas.append({"id": meta["id"], "clase": "desconocido", "identidad": meta["id"], "fuente": "desconocido_lfw", "variante": "original", "particion": "entrenamiento"})

    # 5c. Portal completo (72 válidas, 0 aug -> cubre todas las poses y distancias)
    for i, meta in enumerate(meta_portal_validos):
        X_train_img_list.append(X_portal_img[i])
        X_train_tex_list.append(X_portal_tex[i])
        y_train_list.append(cid_desc)
        id_sujeto = f"portal_{meta['identidad']}"
        identidades_train.append(id_sujeto)
        fuentes_train.append("desconocido_portal")
        conteo_fuentes["desconocido_portal"]["original"] += 1
        manifiesto_exp_filas.append({"id": meta["id"], "clase": "desconocido", "identidad": id_sujeto, "fuente": "desconocido_portal", "variante": "original", "particion": "entrenamiento"})

    # 5d. Universidad train (15 válidas + 2 aug cada una = 45)
    for i, meta in enumerate(univ_train_proc):
        img_orig = X_univ_train_img[i]
        tex_orig = X_univ_train_tex[i]

        X_train_img_list.append(img_orig)
        X_train_tex_list.append(tex_orig)
        y_train_list.append(cid_desc)
        identidades_train.append(meta["id"])
        fuentes_train.append("desconocido_universidad")
        conteo_fuentes["desconocido_universidad"]["original"] += 1
        manifiesto_exp_filas.append({"id": meta["id"], "clase": "desconocido", "identidad": meta["id"], "fuente": "desconocido_universidad", "variante": "original", "particion": "entrenamiento"})

        for idx_a, aum in enumerate(aumentar_muestra_entrenamiento(img_orig, rng), 1):
            X_train_img_list.append(aum)
            X_train_tex_list.append(extractor.extraer(aum).vector_caracteristicas)
            y_train_list.append(cid_desc)
            identidades_train.append(meta["id"])
            fuentes_train.append("desconocido_universidad")
            conteo_fuentes["desconocido_universidad"]["aumentadas"] += 1
            manifiesto_exp_filas.append({"id": f"{meta['id']}_aug{idx_a}", "clase": "desconocido", "identidad": meta["id"], "fuente": "desconocido_universidad", "variante": f"aumento_{idx_a}", "particion": "entrenamiento"})

    for k in conteo_fuentes:
        conteo_fuentes[k]["total"] = conteo_fuentes[k]["original"] + conteo_fuentes[k]["aumentadas"]

    X_train_img = np.asarray(X_train_img_list, dtype=np.float32)
    X_train_tex = np.asarray(X_train_tex_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)

    # Guardar manifiesto específico de entrenamiento del experimento
    with open(dir_salida / "manifiesto_entrenamiento_experimento.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "clase", "identidad", "fuente", "variante", "particion"])
        writer.writeheader()
        writer.writerows(manifiesto_exp_filas)

    # Calcular pesos de 2 niveles (identidad + clase)
    pesos_muestras, resumen_pesos = calcular_pesos_identidad_y_clase(
        y_train=y_train,
        identidades_train=identidades_train,
        fuentes_train=fuentes_train,
    )

    print("\nPesos efectivos calculados (2 niveles):")
    for c_nom, c_inf in resumen_pesos["clases"].items():
        print(f"  Clase {c_nom:12s}: {c_inf['muestras']:4d} muestras -> peso efectivo={c_inf['peso_efectivo_total']:6.2f} ({c_inf['proporcion_pct']}%)")
    print("Pesos efectivos por fuente:")
    for f_nom, f_inf in resumen_pesos["fuentes"].items():
        print(f"  Fuente {f_nom:24s}: {f_inf['muestras']:3d} muestras ({f_inf['identidades']:3d} ids) -> total={f_inf['peso_efectivo_total']:6.2f} | por_id={f_inf['peso_por_identidad']:.4f}")

    # Validación SIN aumentos
    X_val_auth_tex = np.stack([extractor.extraer(imagenes_autorizadas[m["id"]]).vector_caracteristicas for m in val_auth_proc])
    X_val_auth_img = np.stack([imagenes_autorizadas[m["id"]] for m in val_auth_proc])
    y_val_auth = np.array([CLASE_TO_ID[m["persona"]] for m in val_auth_proc], dtype=np.int64)

    X_val_univ_tex_arr = np.asarray(X_univ_val_tex, dtype=np.float32) if X_univ_val_tex else np.empty((0, X_train_tex.shape[1]), dtype=np.float32)
    X_val_univ_img_arr = np.asarray(X_univ_val_img, dtype=np.float32) if X_univ_val_img else np.empty((0, 64, 64), dtype=np.float32)

    X_val_tex = np.vstack([X_val_auth_tex, X_lfw_val_tex, X_val_univ_tex_arr])
    X_val_img = np.vstack([X_val_auth_img, X_lfw_val_img, X_val_univ_img_arr])
    y_val = np.concatenate([
        y_val_auth,
        np.full(len(X_lfw_val_tex), cid_desc, dtype=np.int64),
        np.full(len(X_val_univ_tex_arr), cid_desc, dtype=np.int64),
    ])

    return {
        "X_train_tex": X_train_tex,
        "X_train_img": X_train_img,
        "y_train": y_train,
        "pesos_muestras": pesos_muestras,
        "resumen_pesos": resumen_pesos,
        "conteo_fuentes": conteo_fuentes,
        "fallos_portal": fallos_portal,
        "fallos_univ_train": fallos_univ_train,
        "meta_portal_validos": meta_portal_validos,
        "X_val_tex": X_val_tex,
        "X_val_img": X_val_img,
        "y_val": y_val,
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
    }


def evaluar_diagnostico_portal(
    modelos_pipes: List[Tuple[str, PipelineCandidatoCongelado]],
    dir_salida: Path,
) -> Dict[str, Any]:
    """
    Evalúa las 75 capturas del portal como diagnóstico de entrenamiento/desarrollo
    para la nueva iteración frente a las dos CNN anteriores.
    """
    ruta_manifiesto = RAIZ / "data" / "propio" / "manifiesto_local.csv"
    manifiesto = ManifiestoDataset(ruta_manifiesto)
    regs_portal = [
        r for r in manifiesto.registros
        if r.procedencia == "portal_web_externo" and r.particion == "prueba"
    ]
    regs_portal.sort(key=lambda r: (r.persona, r.pose, r.distancia or ""))

    resultados_por_modelo = {etq: [] for etq, _ in modelos_pipes}

    for reg in regs_portal:
        frame_bgr = cv2.imread(str(RAIZ / reg.ruta))
        usada_antes_15 = (reg.pose, reg.distancia) in POSES_PORTAL_SELECCIONADAS

        for etq, pipe in modelos_pipes:
            t0 = time.perf_counter()
            det, dec, probs = pipe.evaluar_fotograma_completo(frame_bgr)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            if det.estado != EstadoDeteccion.UN_ROSTRO:
                cat = "fallo_deteccion"
                pred_nom = None
                score = 0.0
            else:
                pred_idx = int(np.argmax(probs))
                pred_nom = pipe.mapeo_id_clase.get(pred_idx, "desconocido")
                score = float(probs[pred_idx])
                if dec.estado_puerta == "ABIERTA":
                    cat = "falsa_aceptacion"
                elif pred_nom == "desconocido":
                    cat = "clasificacion_desconocido"
                else:
                    cat = "rechazo_umbral"

            resultados_por_modelo[etq].append({
                "id": reg.identificador,
                "participante": reg.persona,
                "pose": reg.pose,
                "distancia": reg.distancia,
                "usada_antes_15": usada_antes_15,
                "decision": dec.estado_puerta,
                "categoria": cat,
                "clase_argmax": pred_nom,
                "score": round(score, 4),
                "latencia_ms": round(lat_ms, 2),
            })

    def resumir(items):
        tot = len(items)
        fa = sum(1 for x in items if x["categoria"] == "falsa_aceptacion")
        desc = sum(1 for x in items if x["categoria"] == "clasificacion_desconocido")
        umb = sum(1 for x in items if x["categoria"] == "rechazo_umbral")
        det_f = sum(1 for x in items if x["categoria"] == "fallo_deteccion")
        return {
            "intentos": tot,
            "fa": fa,
            "far_pct": round(100.0 * fa / tot, 2) if tot > 0 else 0.0,
            "desconocido": desc,
            "umbral": umb,
            "fallo_deteccion": det_f,
        }

    reporte_portal = {}
    for etq, items in resultados_por_modelo.items():
        rep = {
            "global_75": resumir(items),
            "subconjunto_historico_15": resumir([x for x in items if x["usada_antes_15"]]),
            "subconjunto_historico_60": resumir([x for x in items if not x["usada_antes_15"]]),
            "por_participante": {p: resumir([x for x in items if x["participante"] == p]) for p in sorted({x["participante"] for x in items})},
            "por_pose": {po: resumir([x for x in items if x["pose"] == po]) for po in ("frontal", "izquierda", "derecha", "arriba", "abajo")},
            "por_distancia": {di: resumir([x for x in items if x["distancia"] == di]) for di in ("cercana", "media", "lejana")},
        }
        reporte_portal[etq] = rep

    return reporte_portal


def ejecutar_experimento_portal_completo():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"exp_4clases_portal_completo_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)
    print(f"Directorio de resultados: {dir_salida}")

    datos = preparar_datasets_portal_completo(dir_salida)

    X_train_tex = datos["X_train_tex"]
    y_train = datos["y_train"]
    X_val_tex = datos["X_val_tex"]
    y_val = datos["y_val"]
    X_train_img = datos["X_train_img"]
    X_val_img = datos["X_val_img"]
    pesos_muestras = datos["pesos_muestras"]

    print("\n" + "=" * 75)
    print(" ENTRENANDO LOS CINCO MODELOS CON BALANCE POR IDENTIDAD Y CLASE")
    print("=" * 75)

    modelos = {}
    metadatos_modelos = {}

    # 1. KNN (sin sample_weight; cobertura completa de poses y distancias)
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

    # 2. Árbol (sample_weight + class_weight=None para evitar doble compensación)
    arbol, _, cfg_arbol = ClasificadorArbol.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight=None, sample_weight=pesos_muestras, metrica="macro_recall", semilla=SEMILLA
    )
    cfg_arbol["macro_recall_val"] = round(calcular_macro_recall(y_val, arbol.predecir(X_val_tex)), 2)
    modelos["arbol"] = arbol
    metadatos_modelos["arbol"] = cfg_arbol
    print(f"  [Árbol] {cfg_arbol}")

    # 3. SVM (sample_weight + class_weight=None para evitar doble compensación)
    svm, _, cfg_svm = ClasificadorSVM.seleccionar_y_entrenar(
        X_train_tex, y_train, X_val_tex, y_val,
        class_weight=None, sample_weight=pesos_muestras, metrica="macro_recall", semilla=SEMILLA
    )
    cfg_svm["macro_recall_val"] = round(calcular_macro_recall(y_val, svm.predecir(X_val_tex)), 2)
    modelos["svm"] = svm
    metadatos_modelos["svm"] = cfg_svm
    print(f"  [SVM]   {cfg_svm}")

    # 4. ANN (pérdida individual ponderada pesos_muestras + pesos_clases=None)
    ann = ClasificadorANN(n_entrada=X_train_tex.shape[1], n_clases=N_CLASES, semilla=SEMILLA)
    ann.entrenar_con_validacion(
        X_train_tex, y_train, X_val_tex, y_val,
        pesos_clases=None, pesos_muestras=pesos_muestras, metrica_checkpoint="macro_recall"
    )
    metadatos_modelos["ann"] = {
        "mejor_epoca": ann.mejor_epoca_,
        "macro_recall_val": round(calcular_macro_recall(y_val, ann.predecir(X_val_tex)), 2),
    }
    modelos["ann"] = ann
    print(f"  [ANN]   {metadatos_modelos['ann']}")

    # 5. CNN (pérdida individual ponderada pesos_muestras + pesos_clases=None)
    cnn = ClasificadorCNN(n_clases=N_CLASES, semilla=SEMILLA)
    cnn.entrenar_con_validacion(
        X_train_img, y_train, X_val_img, y_val,
        pesos_clases=None, pesos_muestras=pesos_muestras, metrica_checkpoint="macro_recall"
    )
    metadatos_modelos["cnn"] = {
        "mejor_epoca": cnn.mejor_epoca_,
        "macro_recall_val": round(calcular_macro_recall(y_val, cnn.predecir(X_val_img)), 2),
    }
    modelos["cnn"] = cnn
    print(f"  [CNN]   {metadatos_modelos['cnn']}")

    # Guardar los 5 modelos
    ext_modelos = {"knn": "modelo.joblib", "arbol": "modelo.joblib", "svm": "modelo.joblib", "ann": "modelo.pt", "cnn": "modelo.pt"}
    rutas_modelos = {}
    for nm, mod in modelos.items():
        sub = dir_salida / nm
        sub.mkdir(parents=True, exist_ok=True)
        arch = sub / ext_modelos[nm]
        mod.guardar(arch)
        rutas_modelos[nm] = arch

    # Calibración en Validación
    print("\n" + "=" * 75)
    print(" CALIBRACIÓN EXCLUSIVA EN VALIDACIÓN (64 AUTH + 100 LFW + 7 UNIV)")
    print("=" * 75)

    calibraciones = {}
    for nm in ("arbol", "knn", "svm", "ann", "cnn"):
        res = evaluar_modelo_sobre_validacion(
            nombre_modelo=f"portal_completo_{nm}",
            modelo=modelos[nm],
            datos=datos,
            es_cnn=(nm == "cnn"),
            umbral_fijo=None,
        )
        calibraciones[nm] = res
        if res["regla_b_habilitada"]:
            rb = res["regla_b"]
            print(
                f"  {nm.upper():5s}: VIABLE th={res['umbral_recomendado']:.2f} | "
                f"FRR={rb['frr_puerta_pct']}% ({rb['frr_conteo']}) | "
                f"FAR_global={rb['far_total_pct']}% ({rb['far_conteo']}) | "
                f"LFW={rb['far_lfw']['far_pct']}% | Univ={rb['far_universidad']['far_pct']}%"
            )
        else:
            ra = res["regla_a"]
            print(f"  {nm.upper():5s}: sin_umbral_viable | Regla A FAR={ra['far_total_pct']}% | FRR={ra['frr_puerta_pct']}%")

    # Referencia de las 2 CNN anteriores sobre validación
    pipe_ant = PipelineCandidatoCongelado(RAIZ / "configuracion_candidato_alineado.json", raiz_proyecto=RAIZ)
    pipe_v1 = PipelineCandidatoCongelado(
        RAIZ / "resultados" / "exp_4clases_universidad_20261120_192345" / "configuracion_candidato_universidad_propuesto.json",
        raiz_proyecto=RAIZ,
    )
    res_ant = evaluar_modelo_sobre_validacion("anterior_cnn_084", pipe_ant.clasificador, datos, True, pipe_ant.umbral_score)
    res_v1 = evaluar_modelo_sobre_validacion("univ_v1_cnn_050", pipe_v1.clasificador, datos, True, pipe_v1.umbral_score)

    # Guardar configuración propuesta de CNN (sin tocar web)
    ruta_cfg_prop = dir_salida / "configuracion_candidato_portal_completo_propuesto.json"
    umbral_cnn = calibraciones["cnn"]["umbral_recomendado"]
    if umbral_cnn is not None:
        cfg_prop = {
            "candidato": {
                "identificador": "cnn_4clases_portal_completo_propuesto",
                "nombre": "CNN 4 Clases (Cobertura Completa Portal + Balance 2 Niveles)",
                "tipo_arquitectura": "CNN",
                "ruta_modelo": str(rutas_modelos["cnn"].relative_to(RAIZ)).replace("\\", "/"),
                "sha256_modelo": calcular_sha256(rutas_modelos["cnn"]),
                "umbral_score": umbral_cnn,
                "orden_clases": CLASES_NOMBRES,
                "mapeo_clases": {str(k): v for k, v in ID_TO_CLASE.items()},
                "personas_autorizadas": PERSONAS_AUTORIZADAS,
                "fecha_propuesta": datetime.now().isoformat(),
                "estado": "propuesto_sin_prueba_final",
            },
            "detector_facial": pipe_ant.config["detector_facial"],
            "preprocesamiento": pipe_ant.config["preprocesamiento"],
            "regla_acceso": pipe_ant.config["regla_acceso"],
        }
        ruta_cfg_prop.write_text(json.dumps(cfg_prop, indent=2, ensure_ascii=False), encoding="utf-8")

        # Diagnóstico comparativo sobre las 75 capturas del portal (diagnóstico de entrenamiento!)
        print("\n" + "=" * 75)
        print(" DIAGNÓSTICO SOBRE LAS 75 CAPTURAS DEL PORTAL (DATOS DE ENTRENAMIENTO)")
        print("=" * 75)
        pipe_nuevo = PipelineCandidatoCongelado(ruta_cfg_prop, raiz_proyecto=RAIZ)
        diag_portal = evaluar_diagnostico_portal([
            ("anterior_cnn_084", pipe_ant),
            ("univ_v1_cnn_050", pipe_v1),
            ("nueva_portal_completo_cnn", pipe_nuevo),
        ], dir_salida)

        for etq in ("anterior_cnn_084", "univ_v1_cnn_050", "nueva_portal_completo_cnn"):
            g = diag_portal[etq]["global_75"]
            arriba = diag_portal[etq]["por_pose"]["arriba"]
            abajo = diag_portal[etq]["por_pose"]["abajo"]
            cercana = diag_portal[etq]["por_distancia"]["cercana"]
            lejana = diag_portal[etq]["por_distancia"]["lejana"]
            print(
                f"  {etq:26s}: Global={g['fa']}/75 ({g['far_pct']}%) | "
                f"arriba={arriba['fa']}/15 | abajo={abajo['fa']}/15 | "
                f"cercana={cercana['fa']}/25 | lejana={lejana['fa']}/25"
            )
    else:
        diag_portal = {"nota": "CNN sin umbral viable en validación"}

    # Limpiar curvas grandes al guardar resumen JSON
    cal_limpias = {}
    for k, v in calibraciones.items():
        vc = dict(v)
        if vc.get("regla_b"):
            vc["regla_b"] = {kk: vv for kk, vv in vc["regla_b"].items() if kk != "decisiones_individuales"}
        if vc.get("regla_a"):
            vc["regla_a"] = {kk: vv for kk, vv in vc["regla_a"].items() if kk != "decisiones_individuales"}
        cal_limpias[k] = vc

    resumen = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "semilla": SEMILLA,
        "conteo_fuentes": datos["conteo_fuentes"],
        "resumen_pesos": datos["resumen_pesos"],
        "fallos_portal_excluidos": datos["fallos_portal"],
        "fallos_univ_train_excluidos": datos["fallos_univ_train"],
        "metadatos_modelos": metadatos_modelos,
        "calibracion_nuevos": cal_limpias,
        "referencia_anterior_084": {kk: vv for kk, vv in res_ant["regla_b"].items() if kk != "decisiones_individuales"},
        "referencia_univ_v1_050": {kk: vv for kk, vv in res_v1["regla_b"].items() if kk != "decisiones_individuales"},
        "diagnostico_portal_entrenamiento": diag_portal,
    }
    (dir_salida / "resumen_experimento_portal_completo.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados guardados en: {dir_salida}")
    return resumen, dir_salida


if __name__ == "__main__":
    ejecutar_experimento_portal_completo()
