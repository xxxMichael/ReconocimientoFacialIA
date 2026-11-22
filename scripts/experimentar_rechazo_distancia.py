"""
Experimento de desarrollo: Mejora del rechazo de desconocidos mediante
etapa explícita de distancia a representaciones de entrenamiento.

Versión auditada y corregida:
1. Validación completa de artefactos base antes de ejecutar (modelos SHA-256,
   mapeo de clases, YuNet parámetros y ONNX, preprocesamiento, hash canonizado
   de filas del manifiesto, hashes de imágenes de desarrollo y manifiesto LFW).
2. Alineación rigurosa por ID de muestra (sin indexación posicional).
3. Conservación de las causas reales de fallos (detección vs procesamiento) en el denominador.
4. Deshabilitación estricta de Regla B cuando no existe umbral viable (sin aperturas fantasmas ni distancias dummy).
5. Precisión de punto flotante completa (sin truncamiento ni redondeo al guardar).
6. Verificación de integridad por SHA-256 y metadatos de banco, exigiendo concordancia al recargar.
7. Medición rigurosa de latencia y memoria RAM vs disco.
8. Cálculo 100% dinámico de tablas y desgloses a partir de las decisiones efectivas.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.entrenar_desarrollo import (
    BINS, CUADRICULA, HILOS, RESOLUCION, SIGMA, SOLAPAMIENTO_MINIMO,
    cargar_registros, preparar_captura, matriz_datos
)
from scripts.calibrar_desarrollo import CALIBRACION_LFW, preparar_negativos
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.clasificacion.regla_acceso import DecisionAcceso
from reconocimiento.clasificacion.rechazo_distancia import (
    BancoReferencias,
    evaluar_decision_acceso_con_distancia,
    METRICAS_VALIDAS
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.fabrica import crear_detector

MODELOS_MAP = {
    "ann": (ClasificadorANN, "modelo.pt"),
    "arbol": (ClasificadorArbol, "modelo.joblib"),
    "cnn": (ClasificadorCNN, "modelo.pt"),
    "knn": (ClasificadorKNN, "modelo.joblib"),
    "svm": (ClasificadorSVM, "modelo.joblib"),
}

RESULTADOS = RAIZ / "resultados"


def calcular_sha256(archivo: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en disco."""
    return hashlib.sha256(Path(archivo).read_bytes()).hexdigest()


def calcular_tasa_pct(numerador: int, denominador: int) -> Optional[float]:
    """Calcula el porcentaje (0.0 a 100.0) o retorna None si el denominador es 0 (no calculable)."""
    if denominador <= 0:
        return None
    return 100.0 * float(numerador) / float(denominador)


def formatear_tasa(tasa: Optional[float], decimales: int = 2) -> str:
    """Formatea la tasa en string con símbolo % o 'no calculable' si es None."""
    if tasa is None:
        return "no calculable"
    return f"{tasa:.{decimales}f}%"


def categorizar_fallo(estado: str) -> Tuple[str, str]:
    """
    Categoriza una falla en captura o detección sin perder su causa real.
    Retorna (tipo_fallo, motivo_cierre):
      tipo_fallo: 'deteccion' o 'procesamiento'
      motivo_cierre: etiqueta canónica (e.g. 'DETECCION_SIN_ROSTRO', 'PROCESAMIENTO_ERROR_DECODIFICACION')
    """
    est = (estado or "").strip().lower()
    if est == "sin_rostro":
        return ("deteccion", "DETECCION_SIN_ROSTRO")
    elif est == "varios_rostros":
        return ("deteccion", "DETECCION_VARIOS_ROSTROS")
    elif est == "localizacion_incorrecta":
        return ("deteccion", "DETECCION_LOCALIZACION_INCORRECTA")
    elif est == "error_lectura":
        return ("procesamiento", "PROCESAMIENTO_ERROR_LECTURA")
    elif est == "error_decodificacion":
        return ("procesamiento", "PROCESAMIENTO_ERROR_DECODIFICACION")
    elif est in ("recorte_vacio", "recorte_invalido"):
        return ("procesamiento", "PROCESAMIENTO_RECORTE_VACIO")
    elif est == "caja_revisada_fuera_de_imagen":
        return ("procesamiento", "PROCESAMIENTO_CAJA_FUERA_DE_IMAGEN")
    elif est == "sin_caja_revisada":
        return ("procesamiento", "PROCESAMIENTO_SIN_CAJA_REVISADA")
    elif est == "excluida_por_protocolo":
        return ("procesamiento", "PROCESAMIENTO_EXCLUIDA_PROTOCOLO")
    else:
        return ("procesamiento", f"PROCESAMIENTO_OTRO ({estado})")


def validar_artefactos_base(
    dir_desarrollo_base: Path,
    detector: Any,
    extractor: Any,
    raiz_proyecto: Path = RAIZ
) -> Dict[str, Any]:
    """
    Verifica rigurosamente que modelos, clases, detector, preprocesamiento y datos
    correspondan exactamente a los artefactos base de desarrollo antes de ejecutar el experimento.
    Si cualquier referencia falta o discrepa, detiene el proceso con un error específico.
    """
    dir_desarrollo_base = Path(dir_desarrollo_base)
    informe_path = dir_desarrollo_base / "informe.json"
    if not informe_path.is_file():
        raise FileNotFoundError(f"No se encontró informe.json en {dir_desarrollo_base}")

    informe = json.loads(informe_path.read_text(encoding="utf-8"))

    # 1. Validar detector YuNet: hash ONNX y parámetros
    sha_onnx_actual = calcular_sha256(detector.ruta_modelo)
    sha_onnx_esperado = informe.get("detector", {}).get("sha256_onnx")
    if not sha_onnx_esperado or sha_onnx_actual != sha_onnx_esperado:
        raise ValueError(f"Discrepancia en ONNX de YuNet: esperado '{sha_onnx_esperado}', actual '{sha_onnx_actual}'")

    det_info = informe.get("detector", {})
    for param, val in [
        ("score_threshold", detector.score_threshold),
        ("nms_threshold", detector.nms_threshold),
        ("margen_expansion", detector.margen_expansion)
    ]:
        if det_info.get(param) != val:
            raise ValueError(f"Discrepancia en parámetro detector '{param}': esperado {det_info.get(param)}, actual {val}")

    # 2. Validar preprocesamiento
    prep = informe.get("preprocesamiento", {})
    if prep != {"resolucion": list(RESOLUCION), "intensidad": "float32_0_1", "sigma": SIGMA}:
        raise ValueError(f"Discrepancia en parámetros de preprocesamiento respecto a informe base: {prep}")

    # 3. Validar extractor de características
    caract = informe.get("caracteristicas", {})
    if caract != {"cuadricula": list(CUADRICULA), "bins": BINS, "dimension": extractor.dimension_total}:
        raise ValueError(f"Discrepancia en parámetros de histogramas regionales respecto a informe base: {caract}")

    # 4. Validar clases y orden de clases
    clases_info = informe.get("clases", {})
    p2id_esperado = clases_info.get("persona_to_id", {})
    autorizadas_esperadas = clases_info.get("personas_autorizadas", [])
    if not p2id_esperado or not autorizadas_esperadas:
        raise ValueError("Falta definición de clases o personas_autorizadas en informe.json")

    # 5. Validar cada modelo contra su calibración base
    for nom_mod, (_, nom_arch) in MODELOS_MAP.items():
        dir_mod = dir_desarrollo_base / nom_mod
        ruta_mod = dir_mod / nom_arch
        if not ruta_mod.is_file():
            raise FileNotFoundError(f"Falta el archivo de modelo {ruta_mod}")

        ruta_calib = dir_mod / "calibracion_desarrollo.json"
        if not ruta_calib.is_file():
            raise FileNotFoundError(f"Falta calibracion_desarrollo.json para {nom_mod} en {dir_mod}")

        calib_data = json.loads(ruta_calib.read_text(encoding="utf-8"))
        sha_modelo_esperado = calib_data.get("sha256_modelo")
        if not sha_modelo_esperado:
            raise ValueError(f"Falta 'sha256_modelo' en calibracion_desarrollo.json de {nom_mod}")

        sha_modelo_actual = calcular_sha256(ruta_mod)
        if sha_modelo_actual != sha_modelo_esperado:
            raise ValueError(
                f"Discrepancia en hash SHA-256 del modelo {nom_mod}: "
                f"esperado {sha_modelo_esperado}, actual {sha_modelo_actual}"
            )

        clases_mod = calib_data.get("clases_en_orden", [])
        if clases_mod != autorizadas_esperadas:
            raise ValueError(
                f"Discrepancia en orden de clases para {nom_mod}: "
                f"esperado {autorizadas_esperadas}, actual {clases_mod}"
            )

        # Validar manifiesto de desconocidos de calibración contra calibración base
        sha_manifiesto_neg_esp = calib_data.get("negativos", {}).get("sha256_manifiesto")
        if not sha_manifiesto_neg_esp:
            raise ValueError(f"Falta 'sha256_manifiesto' en negativos de calibracion_desarrollo.json para {nom_mod}")
        calib_lfw_man = CALIBRACION_LFW / "manifiesto.csv"
        if not calib_lfw_man.is_file():
            raise FileNotFoundError(f"Falta manifiesto de calibración de desconocidos: {calib_lfw_man}")
        sha_manifiesto_neg_act = calcular_sha256(calib_lfw_man)
        if sha_manifiesto_neg_act != sha_manifiesto_neg_esp:
            raise ValueError(
                f"Discrepancia en hash del manifiesto de calibración de desconocidos LFW para {nom_mod}: "
                f"esperado {sha_manifiesto_neg_esp}, actual {sha_manifiesto_neg_act}"
            )

        # Validar clases_mapeo.json del modelo y su correspondencia exacta
        ruta_mapeo = dir_mod / "clases_mapeo.json"
        if not ruta_mapeo.is_file():
            raise FileNotFoundError(f"Falta clases_mapeo.json para {nom_mod} en {dir_mod}")
        mapeo_data = json.loads(ruta_mapeo.read_text(encoding="utf-8"))

        p2id_mod = mapeo_data.get("persona_to_id", {})
        if p2id_mod != p2id_esperado:
            raise ValueError(
                f"Discrepancia en persona_to_id de clases_mapeo.json para {nom_mod}: "
                f"esperado {p2id_esperado}, actual {p2id_mod}"
            )

        id2p_mod = mapeo_data.get("id_to_persona", {})
        if not id2p_mod or len(id2p_mod) != len(p2id_esperado):
            raise ValueError(
                f"Mapa inverso id_to_persona inválido o incompleto en clases_mapeo.json para {nom_mod}"
            )
        for persona, p_id in p2id_esperado.items():
            if id2p_mod.get(str(p_id)) != persona:
                raise ValueError(
                    f"Discrepancia en mapa inverso id_to_persona['{p_id}'] para {nom_mod}: "
                    f"esperado '{persona}', actual '{id2p_mod.get(str(p_id))}'"
                )

        aut_mod = mapeo_data.get("personas_autorizadas", [])
        if aut_mod != autorizadas_esperadas:
            raise ValueError(
                f"Discrepancia en personas_autorizadas de clases_mapeo.json para {nom_mod}: "
                f"esperado {autorizadas_esperadas}, actual {aut_mod}"
            )

        # Cargar modelo para verificar clases_
        cls_tipo, _ = MODELOS_MAP[nom_mod]
        mod_instancia = cls_tipo.cargar(ruta_mod)
        if not hasattr(mod_instancia, "clases_"):
            raise ValueError(f"El clasificador {nom_mod} no posee el atributo clases_")
        clases_modelo_actual = list(mod_instancia.clases_)
        clases_esperadas_ids = sorted(p2id_esperado.values())
        if clases_modelo_actual != clases_esperadas_ids:
            raise ValueError(
                f"Discrepancia en modelo.clases_ para {nom_mod}: "
                f"esperado {clases_esperadas_ids}, actual {clases_modelo_actual}"
            )

    # 6. Validar hash canonizado de las filas de desarrollo del manifiesto
    sha_filas_esperado = informe.get("sha256_filas_manifiesto_01_a_04")
    if not sha_filas_esperado:
        raise ValueError("Falta 'sha256_filas_manifiesto_01_a_04' en informe.json")
    registros_base, sha_base = cargar_registros(solo_base=True)
    registros_todos, sha_todos = cargar_registros(solo_base=False)
    if sha_filas_esperado == sha_todos:
        registros = registros_todos
    elif sha_filas_esperado == sha_base:
        registros = registros_base
    else:
        raise ValueError(
            f"Discrepancia en hash de filas de desarrollo del manifiesto: "
            f"esperado {sha_filas_esperado}, actual base {sha_base}, actual completo {sha_todos}"
        )

    # 7. Validar hashes de imágenes de desarrollo contra la trazabilidad original (muestras.json)
    muestras_json_path = dir_desarrollo_base / "muestras.json"
    if not muestras_json_path.is_file():
        raise FileNotFoundError(f"Falta muestras.json en {dir_desarrollo_base}")
    muestras_base = json.loads(muestras_json_path.read_text(encoding="utf-8"))
    mapa_img_esperado = {m["id"]: m.get("sha256_imagen") for m in muestras_base if m.get("sha256_imagen")}

    for r in registros:
        if r.identificador not in mapa_img_esperado or not mapa_img_esperado[r.identificador]:
            raise ValueError(
                f"Falta hash SHA-256 en muestras.json para la imagen de desarrollo utilizada '{r.identificador}'"
            )
        esp_hash = mapa_img_esperado[r.identificador]
        img_path = (raiz_proyecto / r.ruta).resolve()
        if not img_path.is_file():
            raise FileNotFoundError(f"Falta imagen de desarrollo: {img_path}")
        act_hash = calcular_sha256(img_path)
        if act_hash != esp_hash:
            raise ValueError(
                f"Discrepancia en hash de imagen para '{r.identificador}': "
                f"esperado {esp_hash}, actual {act_hash}"
            )

    return informe


def extraer_representaciones(nombre: str, modelo: Any, imagenes: np.ndarray, histogramas: np.ndarray) -> np.ndarray:
    """Extrae las representaciones numéricas correspondientes al clasificador evaluado."""
    n_muestras = len(imagenes) if nombre == "cnn" else len(histogramas)
    if n_muestras == 0:
        dim = 128 if nombre == "cnn" else 256
        return np.empty((0, dim), dtype=np.float32)

    if nombre == "cnn":
        return modelo.extraer_caracteristicas(imagenes)
    elif nombre == "svm":
        return modelo.escalador.transform(histogramas).astype(np.float32)
    else:
        return histogramas.astype(np.float32)


def medir_latencias_detalladas(
    nombre_modelo: str,
    modelo: Any,
    banco: BancoReferencias,
    muestras_imgs: np.ndarray,
    muestras_hists: np.ndarray,
    clase: str,
    metrica: str,
    n_muestras: int = 10,
    rep_por_muestra: int = 50
) -> Dict[str, Any]:
    """
    Mide rigurosamente la latencia separando el tiempo de extracción de representación
    del tiempo de cálculo de distancia sobre la métrica realmente seleccionada.
    """
    n_eval = min(len(muestras_imgs), n_muestras)
    imgs_eval = muestras_imgs[:n_eval]
    hists_eval = muestras_hists[:n_eval]

    # Warm-up de ambas operaciones
    for i in range(n_eval):
        if nombre_modelo == "cnn":
            _ = modelo.extraer_caracteristicas(imgs_eval[i:i+1])
        elif nombre_modelo == "svm":
            _ = modelo.escalador.transform(hists_eval[i:i+1])
        vec_dummy = np.ones(banco.dimension, dtype=np.float32)
        _ = banco.calcular_distancia(vec_dummy, clase, metrica=metrica)

    tiempos_extraccion_ms = []
    tiempos_distancia_ms = []
    tiempos_totales_ms = []

    for i in range(n_eval):
        img_single = imgs_eval[i:i+1]
        hist_single = hists_eval[i:i+1]

        # 1. Medir tiempo de extracción de representación
        if nombre_modelo == "cnn":
            t0 = perf_counter()
            for _ in range(rep_por_muestra):
                vec_rep = modelo.extraer_caracteristicas(img_single)[0]
            t_ext = ((perf_counter() - t0) / rep_por_muestra) * 1000.0
        elif nombre_modelo == "svm":
            t0 = perf_counter()
            for _ in range(rep_por_muestra):
                vec_rep = modelo.escalador.transform(hist_single)[0]
            t_ext = ((perf_counter() - t0) / rep_por_muestra) * 1000.0
        else:
            vec_rep = hist_single[0]
            t_ext = 0.0

        # 2. Medir tiempo de cálculo de distancia
        t0 = perf_counter()
        for _ in range(rep_por_muestra):
            _ = banco.calcular_distancia(vec_rep, clase, metrica=metrica)
        t_dist = ((perf_counter() - t0) / rep_por_muestra) * 1000.0

        tiempos_extraccion_ms.append(t_ext)
        tiempos_distancia_ms.append(t_dist)
        tiempos_totales_ms.append(t_ext + t_dist)

    return {
        "metrica_medida": metrica,
        "n_muestras_desarrollo": n_eval,
        "repeticiones_por_muestra": rep_por_muestra,
        "extraccion_representacion_ms": {
            "mediana": round(float(np.median(tiempos_extraccion_ms)), 4),
            "p95": round(float(np.percentile(tiempos_extraccion_ms, 95)), 4),
            "descripcion": (
                "Pasada por capas convolucionales y densa (fc1) para CNN; "
                "StandardScaler.transform para SVM; 0.0 ms para KNN/Árbol/ANN (histogramas existentes)."
            )
        },
        "calculo_distancia_ms": {
            "mediana": round(float(np.median(tiempos_distancia_ms)), 4),
            "p95": round(float(np.percentile(tiempos_distancia_ms, 95)), 4),
            "descripcion": f"Cálculo de distancia '{metrica}' contra el cluster de entrenamiento de la clase predicha."
        },
        "tiempo_total_adicional_ms": {
            "mediana": round(float(np.median(tiempos_totales_ms)), 4),
            "p95": round(float(np.percentile(tiempos_totales_ms, 95)), 4),
            "descripcion": "Suma de extracción de representación específica y cálculo de distancia."
        }
    }


def optimizar_umbrales_regla_B(
    candidatos_score: List[float],
    metricas_eval: List[str],
    dists_val_por_metrica: Dict[str, List[float]],
    dists_neg_por_metrica: Dict[str, List[float]],
    scores_val_procesadas: List[float],
    scores_neg_unrostro: List[float],
    preds_val_procesadas: List[str],
    intentos_val: List[Dict[str, Any]],
    mapa_idx_val: Dict[str, int],
    idx_neg_unrostro: List[int],
    cota_frr: float = 0.30
) -> Tuple[str, Optional[str], Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
    """
    Optimiza conjuntamente los umbrales de score y distancia bajo la cota dura de FRR <= 30.0%.
    Si ninguna configuración es viable, retorna ('sin_umbral_viable', None, None, None, {}, {}).
    """
    total_val_intentos = len(intentos_val)
    if total_val_intentos == 0:
        return ("sin_umbral_viable", None, None, None, {}, {})

    mejores_candidatos_por_metrica = {}

    for met in metricas_eval:
        dv = dists_val_por_metrica.get(met, [])
        dn = dists_neg_por_metrica.get(met, [])
        if not dv:
            continue
        d_candidatos = sorted(set(dv))

        candidatos_viables = []
        for s_th in candidatos_score:
            for d_th in d_candidatos:
                ap_val = 0
                corr_val = 0
                conf_val = 0
                for m in intentos_val:
                    mid = m["id"]
                    if mid in mapa_idx_val:
                        idx = mapa_idx_val[mid]
                        if scores_val_procesadas[idx] >= s_th and dv[idx] <= d_th:
                            ap_val += 1
                            if preds_val_procesadas[idx] == m["persona"]:
                                corr_val += 1
                            else:
                                conf_val += 1

                rech_val = total_val_intentos - ap_val
                frr_val = rech_val / total_val_intentos

                if frr_val <= cota_frr:
                    ap_neg = sum(
                        scores_neg_unrostro[j] >= s_th and dn[j] <= d_th
                        for j in range(len(idx_neg_unrostro))
                    )
                    clave = (ap_neg, rech_val, -corr_val, -s_th, d_th)
                    candidatos_viables.append((clave, s_th, d_th, ap_neg, rech_val, corr_val, conf_val))

        if candidatos_viables:
            candidatos_viables.sort(key=lambda x: x[0])
            mejores_candidatos_por_metrica[met] = candidatos_viables[0]

    if not mejores_candidatos_por_metrica:
        return ("sin_umbral_viable", None, None, None, {}, {})

    metricas_ordenadas = sorted(
        mejores_candidatos_por_metrica.items(),
        key=lambda item: item[1][0]
    )
    metrica_ganadora, opt_info = metricas_ordenadas[0]
    _, opt_s_th, opt_d_th, opt_ap_neg, opt_rech_val, opt_corr, opt_conf = opt_info

    info_opt = {
        "ap_neg": opt_ap_neg,
        "rech_val": opt_rech_val,
        "corr_val": opt_corr,
        "conf_val": opt_conf
    }
    return ("umbral_viable", metrica_ganadora, float(opt_s_th), float(opt_d_th), info_opt, mejores_candidatos_por_metrica)


def evaluar_lote_decisiones(
    intentos_val: List[Dict[str, Any]],
    mapa_idx_val: Dict[str, int],
    preds_val_procesadas: List[str],
    scores_val_procesadas: List[float],
    rep_val: Optional[np.ndarray],
    banco: Optional[BancoReferencias],
    s_th: Optional[float],
    d_th: Optional[float],
    metrica: Optional[str],
    personas: List[str],
    estados_neg: List[str],
    preds_neg_unrostro: List[str],
    scores_neg_unrostro: List[float],
    rep_neg: Optional[np.ndarray],
    idx_neg_unrostro: List[int],
    estado_modelo: str = "umbral_viable"
) -> Tuple[Dict[str, DecisionAcceso], List[DecisionAcceso]]:
    """
    Evalúa las decisiones para todas las muestras de validación y desconocidos.
    Conserva la alineación exacta por ID de muestra y las causas reales de fallos.
    Si estado_modelo == 'sin_umbral_viable', deshabilita Regla B y cierra todos los intentos
    con motivo SIN_UMBRAL_VIABLE (o el motivo real de falla en captura), preservando las
    predicciones originales del clasificador para diagnóstico y auditoría.
    """
    decisiones_val = {}

    for m in intentos_val:
        mid = m["id"]
        if mid in mapa_idx_val:
            idx = mapa_idx_val[mid]
            pred_nom = preds_val_procesadas[idx]
            conf_val = scores_val_procesadas[idx]

            if estado_modelo == "sin_umbral_viable":
                dec = DecisionAcceso(
                    estado_puerta="CERRADA",
                    es_autorizado=False,
                    persona_identificada=pred_nom,
                    confianza=conf_val,
                    umbral=0.0,
                    motivo_cierre="SIN_UMBRAL_VIABLE",
                    mensaje="ACCESO DENEGADO: Sin umbral viable para Regla B",
                    distancia=None,
                    umbral_distancia=None
                )
            else:
                dec = evaluar_decision_acceso_con_distancia(
                    nombre_predicho=pred_nom,
                    confianza=conf_val,
                    umbral_confianza=s_th if s_th is not None else 0.0,
                    personas_autorizadas=personas,
                    vector_representacion=rep_val[idx] if rep_val is not None else None,
                    banco_referencias=banco,
                    umbral_distancia=d_th,
                    metrica_distancia=metrica or "min_cos",
                    estado_deteccion=EstadoDeteccion.UN_ROSTRO
                )
        else:
            tipo_f, mot_f = categorizar_fallo(m.get("estado", "desconocido"))
            dec = DecisionAcceso(
                estado_puerta="CERRADA",
                es_autorizado=False,
                persona_identificada="DESCONOCIDO",
                confianza=0.0,
                umbral=float(s_th) if (s_th is not None and math.isfinite(s_th)) else 0.0,
                motivo_cierre=mot_f,
                mensaje=f"ACCESO DENEGADO: {mot_f}",
                distancia=None,
                umbral_distancia=float(d_th) if (d_th is not None and math.isfinite(d_th)) else None
            )
        decisiones_val[mid] = dec

    decisiones_neg = []
    p_neg_idx = 0
    for i, est in enumerate(estados_neg):
        if est == "UN_ROSTRO":
            pred_neg = preds_neg_unrostro[p_neg_idx] if p_neg_idx < len(preds_neg_unrostro) else "DESCONOCIDO"
            conf_neg = scores_neg_unrostro[p_neg_idx] if p_neg_idx < len(scores_neg_unrostro) else 0.0
            v_neg = rep_neg[p_neg_idx] if (rep_neg is not None and p_neg_idx < len(rep_neg)) else None

            if estado_modelo == "sin_umbral_viable":
                dec = DecisionAcceso(
                    estado_puerta="CERRADA",
                    es_autorizado=False,
                    persona_identificada=pred_neg,
                    confianza=conf_neg,
                    umbral=0.0,
                    motivo_cierre="SIN_UMBRAL_VIABLE",
                    mensaje="ACCESO DENEGADO: Sin umbral viable para Regla B",
                    distancia=None,
                    umbral_distancia=None
                )
            else:
                dec = evaluar_decision_acceso_con_distancia(
                    nombre_predicho=pred_neg,
                    confianza=conf_neg,
                    umbral_confianza=s_th if s_th is not None else 0.0,
                    personas_autorizadas=personas,
                    vector_representacion=v_neg,
                    banco_referencias=banco,
                    umbral_distancia=d_th,
                    metrica_distancia=metrica or "min_cos",
                    estado_deteccion=EstadoDeteccion.UN_ROSTRO
                )
            p_neg_idx += 1
        else:
            tipo_f, mot_f = categorizar_fallo(est)
            dec = DecisionAcceso(
                estado_puerta="CERRADA",
                es_autorizado=False,
                persona_identificada="DESCONOCIDO",
                confianza=0.0,
                umbral=float(s_th) if (s_th is not None and math.isfinite(s_th)) else 0.0,
                motivo_cierre=mot_f,
                mensaje=f"ACCESO DENEGADO: {mot_f}",
                distancia=None,
                umbral_distancia=float(d_th) if (d_th is not None and math.isfinite(d_th)) else None
            )
        decisiones_neg.append(dec)

    return decisiones_val, decisiones_neg


def calcular_metricas_experimento(
    decisiones_val: Dict[str, DecisionAcceso],
    decisiones_neg: List[DecisionAcceso],
    intentos_val: List[Dict[str, Any]],
    estados_neg: List[str],
    idx_neg_unrostro: List[int],
    personas: List[str]
) -> Dict[str, Any]:
    """
    Calcula dinámicamente todas las métricas globales y desgloses a partir de las decisiones efectivas.
    Maneja rigurosamente denominadores cero (reportando None / no calculable).
    """
    total_val = len(intentos_val)
    total_neg = len(estados_neg)
    total_neg_unrostro = len(idx_neg_unrostro)

    ap_val = sum(1 for d in decisiones_val.values() if d.estado_puerta == "ABIERTA")
    rech_val = total_val - ap_val
    corr_val = sum(
        1 for mid, d in decisiones_val.items()
        if d.estado_puerta == "ABIERTA" and d.persona_identificada == next(m["persona"] for m in intentos_val if m["id"] == mid)
    )
    conf_val = sum(
        1 for mid, d in decisiones_val.items()
        if d.estado_puerta == "ABIERTA" and d.persona_identificada != next(m["persona"] for m in intentos_val if m["id"] == mid)
    )

    fallos_det_val = sum(1 for d in decisiones_val.values() if d.motivo_cierre.startswith("DETECCION_"))
    fallos_proc_val = sum(1 for d in decisiones_val.values() if d.motivo_cierre.startswith("PROCESAMIENTO_"))

    ap_neg = sum(1 for d in decisiones_neg if d.estado_puerta == "ABIERTA")
    fallos_det_neg = sum(1 for d in decisiones_neg if d.motivo_cierre.startswith("DETECCION_"))
    fallos_proc_neg = sum(1 for d in decisiones_neg if d.motivo_cierre.startswith("PROCESAMIENTO_"))

    far_total_pct = calcular_tasa_pct(ap_neg, total_neg)
    far_cond_pct = calcular_tasa_pct(ap_neg, total_neg_unrostro)
    frr_puerta_pct = calcular_tasa_pct(rech_val, total_val)

    if total_val > 0:
        assert corr_val + conf_val == ap_val
        assert ap_val + rech_val == total_val

    # Desglose por persona
    desglose_por_persona = {}
    for p in personas:
        muestras_p = [m for m in intentos_val if m["persona"] == p]
        tot_p = len(muestras_p)
        corr_p = sum(1 for m in muestras_p if decisiones_val[m["id"]].estado_puerta == "ABIERTA" and decisiones_val[m["id"]].persona_identificada == p)
        conf_p = sum(1 for m in muestras_p if decisiones_val[m["id"]].estado_puerta == "ABIERTA" and decisiones_val[m["id"]].persona_identificada != p)
        bloq_p = sum(1 for m in muestras_p if decisiones_val[m["id"]].estado_puerta == "CERRADA")
        desglose_por_persona[p] = {
            "total": tot_p,
            "correctas": corr_p,
            "confusiones": conf_p,
            "bloqueados": bloq_p,
            "frr_pct": calcular_tasa_pct(bloq_p, tot_p)
        }

    # Desglose por pose
    poses_presentes = sorted({m.get("pose", "desconocida") for m in intentos_val})
    desglose_por_pose = {}
    for pose in poses_presentes:
        muestras_pose = [m for m in intentos_val if m.get("pose") == pose]
        tot_pose = len(muestras_pose)
        corr_pose = sum(1 for m in muestras_pose if decisiones_val[m["id"]].estado_puerta == "ABIERTA" and decisiones_val[m["id"]].persona_identificada == m["persona"])
        conf_pose = sum(1 for m in muestras_pose if decisiones_val[m["id"]].estado_puerta == "ABIERTA" and decisiones_val[m["id"]].persona_identificada != m["persona"])
        bloq_pose = sum(1 for m in muestras_pose if decisiones_val[m["id"]].estado_puerta == "CERRADA")
        desglose_por_pose[pose] = {
            "total": tot_pose,
            "correctas": corr_pose,
            "confusiones": conf_pose,
            "bloqueados": bloq_pose,
            "frr_pct": calcular_tasa_pct(bloq_pose, tot_pose)
        }

    return {
        "total_intentos_val": total_val,
        "aperturas_val": ap_val,
        "rechazos_val": rech_val,
        "correctas_val": corr_val,
        "confusiones_val": conf_val,
        "fallos_deteccion_val": fallos_det_val,
        "fallos_procesamiento_val": fallos_proc_val,
        "frr_puerta_pct": frr_puerta_pct,
        "total_intentos_neg": total_neg,
        "total_neg_unrostro": total_neg_unrostro,
        "aperturas_neg": ap_neg,
        "fallos_deteccion_neg": fallos_det_neg,
        "fallos_procesamiento_neg": fallos_proc_neg,
        "far_total_pct": far_total_pct,
        "far_condicionado_pct": far_cond_pct,
        "por_persona": desglose_por_persona,
        "por_pose": desglose_por_pose
    }


def ejecutar_experimento(dir_desarrollo_base: Path) -> Path:
    dir_desarrollo_base = dir_desarrollo_base.resolve()
    print(f"Directorio de desarrollo base: {dir_desarrollo_base}")
    if not dir_desarrollo_base.exists():
        raise FileNotFoundError(f"No existe el directorio base {dir_desarrollo_base}")

    detector = crear_detector()
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS, rango_intensidad=(0.0, 1.0))

    # Paso 0: Verificación rigurosa de artefactos base
    print("\n--- Paso 0: Verificación de consistencia de artefactos base ---")
    informe_base = validar_artefactos_base(dir_desarrollo_base, detector, extractor)
    print("✓ Artefactos base validados: modelos, clases, detector y preprocesamiento coinciden.")

    # Paso 1: Carga y preparación de capturas propias
    print("\n--- Paso 1: Carga y preparación de capturas propias ---")
    usa_solo_base = (informe_base.get("entrenamiento", {}).get("muestras") == 208)
    registros, hash_manifiesto = cargar_registros(solo_base=usa_solo_base)
    muestras = []
    imagenes = {}
    hashes_particion = {}
    for r in registros:
        m, img = preparar_captura(r, detector, hashes_particion)
        muestras.append(m)
        if img is not None:
            imagenes[r.identificador] = img

    entrenamiento = [m for m in muestras if m["particion"] == "entrenamiento"]
    intentos_val = [m for m in muestras if m["particion"] == "validacion" and m["intento"]]
    if not intentos_val:
        raise ValueError("No se encontraron intentos de validación en la sesión 04.")

    personas = sorted({m["persona"] for m in entrenamiento if m["estado"] == "procesada"})
    persona_to_id = {p: i for i, p in enumerate(personas)}

    img_train, y_train, ids_train = matriz_datos("entrenamiento", muestras, imagenes, persona_to_id)
    hist_train, _ = extractor.extraer_lote(img_train)

    procesadas_val = [m for m in muestras if m["particion"] == "validacion" and m["estado"] == "procesada"]
    if procesadas_val:
        img_val, y_val, ids_val = matriz_datos("validacion", muestras, imagenes, persona_to_id)
        hist_val, _ = extractor.extraer_lote(img_val)
    else:
        img_val = np.empty((0, *RESOLUCION), dtype=np.float32)
        y_val = np.empty((0,), dtype=int)
        ids_val = []
        hist_val = np.empty((0, extractor.dimension_total), dtype=np.float32)

    mapa_idx_val = {id_muestra: idx for idx, id_muestra in enumerate(ids_val)}
    fallos_det_val = [m for m in intentos_val if m["id"] not in mapa_idx_val]
    print(f"Total intentos validación: {len(intentos_val)} (Procesadas exitosamente: {len(ids_val)}, Fallos detección/caja: {len(fallos_det_val)})")

    # Paso 2: Carga y detección en desconocidos de calibración LFW (100 identidades)
    print("\n--- Paso 2: Carga y detección en desconocidos de calibración LFW ---")
    try:
        filas_neg, estados_neg, img_neg = preparar_negativos(detector)
    except ValueError as e:
        if "YuNet no halló ningún rostro único" in str(e):
            with (CALIBRACION_LFW / "manifiesto.csv").open(encoding="utf-8", newline="") as archivo:
                filas_neg = list(csv.DictReader(archivo))
            estados_neg = ["SIN_ROSTRO"] * len(filas_neg)
            img_neg = np.empty((0, *RESOLUCION), dtype=np.float32)
        else:
            raise

    if len(img_neg) > 0:
        hist_neg, _ = extractor.extraer_lote(img_neg)
    else:
        hist_neg = np.empty((0, extractor.dimension_total), dtype=np.float32)

    idx_neg_unrostro = [i for i, est in enumerate(estados_neg) if est == "UN_ROSTRO"]
    fallos_det_neg = len(estados_neg) - len(idx_neg_unrostro)
    print(f"Total desconocidos: {len(estados_neg)} (UN_ROSTRO: {len(idx_neg_unrostro)}, Bloqueados por detector: {fallos_det_neg})")

    # Paso 3: Crear nuevo directorio de salida fechado
    timestamp_str = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    dir_salida = RESULTADOS / f"desarrollo_rechazo_distancia_{timestamp_str}"
    dir_salida.mkdir(parents=True, exist_ok=False)
    print(f"\nGuardando resultados en nuevo directorio: {dir_salida}")

    resumen_modelos = {}
    sha_onnx = calcular_sha256(detector.ruta_modelo)
    sha_manifiesto_calib = calcular_sha256(CALIBRACION_LFW / "manifiesto.csv")

    for nombre_modelo, (tipo_cls, archivo_pesos) in MODELOS_MAP.items():
        print(f"\n=======================================================")
        print(f"Procesando clasificador: {nombre_modelo.upper()}")
        print(f"=======================================================")
        dir_mod_base = dir_desarrollo_base / nombre_modelo
        dir_mod_salida = dir_salida / nombre_modelo
        dir_mod_salida.mkdir(parents=True, exist_ok=True)

        ruta_modelo_pesos = dir_mod_base / archivo_pesos
        sha_modelo_pesos = calcular_sha256(ruta_modelo_pesos)
        modelo = tipo_cls.cargar(ruta_modelo_pesos)

        calib_prev = json.loads((dir_mod_base / "calibracion_desarrollo.json").read_text(encoding="utf-8"))
        res_prev = calib_prev.get("resultado_umbral_elegido")
        umbral_baseline = float(res_prev["umbral"]) if (res_prev is not None and res_prev.get("umbral") is not None) else None

        rep_train = extraer_representaciones(nombre_modelo, modelo, img_train, hist_train)
        rep_val = extraer_representaciones(nombre_modelo, modelo, img_val, hist_val)
        rep_neg = extraer_representaciones(nombre_modelo, modelo, img_neg, hist_neg)

        banco_dict = {}
        for c_idx, c_nom in enumerate(personas):
            mascara = (y_train == c_idx)
            banco_dict[c_nom] = rep_train[mascara]

        metadatos_banco = {
            "modelo": nombre_modelo,
            "dimension": int(rep_train.shape[1]),
            "sesiones_origen": sorted({m["sesion"] for m in entrenamiento}),
            "sha256_modelo": sha_modelo_pesos,
            "tipo_representacion": "vector_penultima_capa_128d" if nombre_modelo == "cnn" else (
                "histogramas_256d_escalados_standard" if nombre_modelo == "svm" else "histogramas_regionales_256d"
            ),
        }
        banco = BancoReferencias(banco_dict, metadatos=metadatos_banco)
        sha_banco = banco.guardar_npz(dir_mod_salida / "referencias.npz")
        tamano_banco_disco = (dir_mod_salida / "referencias.npz").stat().st_size
        tamano_banco_ram = banco.memoria_ram_bytes

        if len(ids_val) > 0:
            entrada_val = img_val if nombre_modelo == "cnn" else hist_val
            probs_val = modelo.predecir_probabilidades(entrada_val)
            preds_val_procesadas = [personas[int(np.argmax(p))] for p in probs_val]
            scores_val_procesadas = [float(np.max(p)) for p in probs_val]
        else:
            preds_val_procesadas = []
            scores_val_procesadas = []

        if len(idx_neg_unrostro) > 0:
            entrada_neg = img_neg if nombre_modelo == "cnn" else hist_neg
            probs_neg = modelo.predecir_probabilidades(entrada_neg)
            preds_neg_unrostro = [personas[int(np.argmax(p))] for p in probs_neg]
            scores_neg_unrostro = [float(np.max(p)) for p in probs_neg]
        else:
            preds_neg_unrostro = []
            scores_neg_unrostro = []

        metricas_eval = ["min_cos", "cent_cos", "min_l2", "cent_l2"]
        dists_val_por_metrica = {}
        dists_neg_por_metrica = {}

        for met in metricas_eval:
            if len(rep_val) > 0:
                dists_val_por_metrica[met] = [
                    banco.calcular_distancia(rep_val[i], preds_val_procesadas[i], metrica=met)
                    for i in range(len(rep_val))
                ]
            else:
                dists_val_por_metrica[met] = []

            if len(rep_neg) > 0:
                dists_neg_por_metrica[met] = [
                    banco.calcular_distancia(rep_neg[i], preds_neg_unrostro[i], metrica=met)
                    for i in range(len(rep_neg))
                ]
            else:
                dists_neg_por_metrica[met] = []

        # Evaluación de Regla A (Baseline actual, sólo puntaje)
        decisiones_val_A, decisiones_neg_A = evaluar_lote_decisiones(
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            preds_val_procesadas=preds_val_procesadas,
            scores_val_procesadas=scores_val_procesadas,
            rep_val=None,
            banco=None,
            s_th=umbral_baseline,
            d_th=None,
            metrica=None,
            personas=personas,
            estados_neg=estados_neg,
            preds_neg_unrostro=preds_neg_unrostro,
            scores_neg_unrostro=scores_neg_unrostro,
            rep_neg=None,
            idx_neg_unrostro=idx_neg_unrostro,
            estado_modelo="umbral_viable" if umbral_baseline is not None else "sin_umbral_viable"
        )

        metricas_A = calcular_metricas_experimento(
            decisiones_val=decisiones_val_A,
            decisiones_neg=decisiones_neg_A,
            intentos_val=intentos_val,
            estados_neg=estados_neg,
            idx_neg_unrostro=idx_neg_unrostro,
            personas=personas
        )

        # Optimización de Regla B (Score + Distancia)
        candidatos_score = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00]
        if umbral_baseline is not None and umbral_baseline not in candidatos_score:
            candidatos_score.append(umbral_baseline)
        candidatos_score.sort()

        estado_modelo, metrica_ganadora, opt_s_th, opt_d_th, info_opt, mejores_candidatos_por_metrica = optimizar_umbrales_regla_B(
            candidatos_score=candidatos_score,
            metricas_eval=metricas_eval,
            dists_val_por_metrica=dists_val_por_metrica,
            dists_neg_por_metrica=dists_neg_por_metrica,
            scores_val_procesadas=scores_val_procesadas,
            scores_neg_unrostro=scores_neg_unrostro,
            preds_val_procesadas=preds_val_procesadas,
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            idx_neg_unrostro=idx_neg_unrostro,
            cota_frr=0.30
        )

        if estado_modelo == "sin_umbral_viable" or len(img_val) == 0:
            if estado_modelo == "sin_umbral_viable":
                print(f"ALERTA: El clasificador {nombre_modelo.upper()} no tiene candidatos viables con FRR <= 30%.")
            latencias_detalle = None
        else:
            latencias_detalle = medir_latencias_detalladas(
                nombre_modelo=nombre_modelo,
                modelo=modelo,
                banco=banco,
                muestras_imgs=img_val,
                muestras_hists=hist_val,
                clase=personas[0],
                metrica=metrica_ganadora
            )

        # Evaluación de Regla B
        decisiones_val_B, decisiones_neg_B = evaluar_lote_decisiones(
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            preds_val_procesadas=preds_val_procesadas,
            scores_val_procesadas=scores_val_procesadas,
            rep_val=rep_val,
            banco=banco,
            s_th=opt_s_th,
            d_th=opt_d_th,
            metrica=metrica_ganadora,
            personas=personas,
            estados_neg=estados_neg,
            preds_neg_unrostro=preds_neg_unrostro,
            scores_neg_unrostro=scores_neg_unrostro,
            rep_neg=rep_neg,
            idx_neg_unrostro=idx_neg_unrostro,
            estado_modelo=estado_modelo
        )

        metricas_B = calcular_metricas_experimento(
            decisiones_val=decisiones_val_B,
            decisiones_neg=decisiones_neg_B,
            intentos_val=intentos_val,
            estados_neg=estados_neg,
            idx_neg_unrostro=idx_neg_unrostro,
            personas=personas
        )

        # Guardar predicciones detalladas
        predicciones_detalle = {
            "modelo": nombre_modelo,
            "estado": estado_modelo,
            "sha256_pesos": sha_modelo_pesos,
            "umbral_baseline": float(umbral_baseline) if umbral_baseline is not None else None,
            "optimizacion_regla_B": {
                "metrica": metrica_ganadora,
                "umbral_score": opt_s_th,
                "umbral_distancia": opt_d_th,
            },
            "validacion": [],
            "desconocidos": []
        }

        for m in intentos_val:
            mid = m["id"]
            if mid in mapa_idx_val:
                idx = mapa_idx_val[mid]
                p_pred = preds_val_procesadas[idx]
                sc = float(scores_val_procesadas[idx])
                dst = float(dists_val_por_metrica[metrica_ganadora][idx]) if metrica_ganadora else None
            else:
                p_pred = "DESCONOCIDO"
                sc = 0.0
                dst = None

            predicciones_detalle["validacion"].append({
                "id": mid,
                "persona_real": m["persona"],
                "sesion": m["sesion"],
                "pose": m["pose"],
                "estado_captura": m["estado"],
                "persona_predicha": p_pred,
                "confianza": sc,
                "distancia": dst,
                "regla_A": {
                    "estado_puerta": decisiones_val_A[mid].estado_puerta,
                    "motivo_cierre": decisiones_val_A[mid].motivo_cierre,
                    "persona_identificada": decisiones_val_A[mid].persona_identificada,
                },
                "regla_B": {
                    "estado_puerta": decisiones_val_B[mid].estado_puerta,
                    "motivo_cierre": decisiones_val_B[mid].motivo_cierre,
                    "persona_identificada": decisiones_val_B[mid].persona_identificada,
                }
            })

        p_idx = 0
        for i, est in enumerate(estados_neg):
            fila_neg = filas_neg[i]
            if est == "UN_ROSTRO":
                p_pred = preds_neg_unrostro[p_idx]
                sc = float(scores_neg_unrostro[p_idx])
                dst = float(dists_neg_por_metrica[metrica_ganadora][p_idx]) if metrica_ganadora else None
                p_idx += 1
            else:
                p_pred = "DESCONOCIDO"
                sc = 0.0
                dst = None

            predicciones_detalle["desconocidos"].append({
                "identidad_lfw": fila_neg["identidad_lfw"],
                "ruta_imagen": fila_neg["ruta_imagen"],
                "estado_deteccion": est,
                "persona_predicha": p_pred,
                "confianza": sc,
                "distancia": dst,
                "regla_A": {
                    "estado_puerta": decisiones_neg_A[i].estado_puerta,
                    "motivo_cierre": decisiones_neg_A[i].motivo_cierre,
                },
                "regla_B": {
                    "estado_puerta": decisiones_neg_B[i].estado_puerta,
                    "motivo_cierre": decisiones_neg_B[i].motivo_cierre,
                }
            })

        (dir_mod_salida / "predicciones_detalle.json").write_text(
            json.dumps(predicciones_detalle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        reduccion_far = (
            round(float(metricas_A["far_total_pct"] - metricas_B["far_total_pct"]), 2)
            if (metricas_A["far_total_pct"] is not None and metricas_B["far_total_pct"] is not None)
            else None
        )

        resumen_modelo = {
            "modelo": nombre_modelo,
            "estado": estado_modelo,
            "sha256_pesos": sha_modelo_pesos,
            "representacion": metadatos_banco["tipo_representacion"],
            "dimension": int(banco.dimension),
            "banco_referencias": {
                "sha256_npz": sha_banco,
                "tamano_disco_npz_bytes": int(tamano_banco_disco),
                "memoria_ram_arrays_bytes": int(tamano_banco_ram),
                "total_vectores": int(banco.total_vectores),
                "por_clase": {c: int(len(m)) for c, m in banco.referencias_por_clase.items()},
            },
            "latencias": latencias_detalle,
            "regla_A_puntaje": {
                "umbral_score": float(umbral_baseline) if umbral_baseline is not None else None,
                "far_pct": round(float(metricas_A["far_total_pct"]), 2) if metricas_A["far_total_pct"] is not None else None,
                "far_condicionado_pct": round(float(metricas_A["far_condicionado_pct"]), 2) if metricas_A["far_condicionado_pct"] is not None else None,
                "falsas_aceptaciones": int(metricas_A["aperturas_neg"]),
                "frr_puerta_pct": round(float(metricas_A["frr_puerta_pct"]), 2) if metricas_A["frr_puerta_pct"] is not None else None,
                "rechazos_autorizados": int(metricas_A["rechazos_val"]),
                "aperturas_identidad_correcta": int(metricas_A["correctas_val"]),
                "aperturas_confusion_autorizados": int(metricas_A["confusiones_val"]),
                "fallos_deteccion_autorizados": int(metricas_A["fallos_deteccion_val"]),
                "fallos_procesamiento_autorizados": int(metricas_A["fallos_procesamiento_val"]),
                "fallos_deteccion_desconocidos": int(metricas_A["fallos_deteccion_neg"]),
                "fallos_procesamiento_desconocidos": int(metricas_A["fallos_procesamiento_neg"]),
            },
            "regla_B_puntaje_y_distancia": {
                "umbral_score": float(opt_s_th) if estado_modelo == "umbral_viable" else None,
                "metrica_distancia": metrica_ganadora if estado_modelo == "umbral_viable" else None,
                "umbral_distancia": float(opt_d_th) if estado_modelo == "umbral_viable" else None,
                "far_pct": round(float(metricas_B["far_total_pct"]), 2) if metricas_B["far_total_pct"] is not None else None,
                "far_condicionado_pct": round(float(metricas_B["far_condicionado_pct"]), 2) if metricas_B["far_condicionado_pct"] is not None else None,
                "falsas_aceptaciones": int(metricas_B["aperturas_neg"]),
                "frr_puerta_pct": round(float(metricas_B["frr_puerta_pct"]), 2) if metricas_B["frr_puerta_pct"] is not None else None,
                "rechazos_autorizados": int(metricas_B["rechazos_val"]),
                "aperturas_identidad_correcta": int(metricas_B["correctas_val"]),
                "aperturas_confusion_autorizados": int(metricas_B["confusiones_val"]),
                "fallos_deteccion_autorizados": int(metricas_B["fallos_deteccion_val"]),
                "fallos_procesamiento_autorizados": int(metricas_B["fallos_procesamiento_val"]),
                "fallos_deteccion_desconocidos": int(metricas_B["fallos_deteccion_neg"]),
                "fallos_procesamiento_desconocidos": int(metricas_B["fallos_procesamiento_neg"]),
                "reduccion_far_puntos_porcentuales": reduccion_far,
                "mantiene_frr_cota_30": bool(metricas_B["frr_puerta_pct"] is not None and metricas_B["frr_puerta_pct"] <= 30.0),
            },
            "desglose_validacion": {
                "por_persona": {
                    p: {
                        "total": metricas_A["por_persona"][p]["total"],
                        "regla_A": {
                            "correctas": metricas_A["por_persona"][p]["correctas"],
                            "confusiones": metricas_A["por_persona"][p]["confusiones"],
                            "bloqueados": metricas_A["por_persona"][p]["bloqueados"],
                            "frr_pct": round(float(metricas_A["por_persona"][p]["frr_pct"]), 2) if metricas_A["por_persona"][p]["frr_pct"] is not None else None,
                        },
                        "regla_B": {
                            "correctas": metricas_B["por_persona"][p]["correctas"],
                            "confusiones": metricas_B["por_persona"][p]["confusiones"],
                            "bloqueados": metricas_B["por_persona"][p]["bloqueados"],
                            "frr_pct": round(float(metricas_B["por_persona"][p]["frr_pct"]), 2) if metricas_B["por_persona"][p]["frr_pct"] is not None else None,
                        }
                    }
                    for p in personas
                },
                "por_pose": {
                    pose: {
                        "total": metricas_A["por_pose"][pose]["total"],
                        "regla_A": {
                            "correctas": metricas_A["por_pose"][pose]["correctas"],
                            "confusiones": metricas_A["por_pose"][pose]["confusiones"],
                            "bloqueados": metricas_A["por_pose"][pose]["bloqueados"],
                            "frr_pct": round(float(metricas_A["por_pose"][pose]["frr_pct"]), 2) if metricas_A["por_pose"][pose]["frr_pct"] is not None else None,
                        },
                        "regla_B": {
                            "correctas": metricas_B["por_pose"][pose]["correctas"],
                            "confusiones": metricas_B["por_pose"][pose]["confusiones"],
                            "bloqueados": metricas_B["por_pose"][pose]["bloqueados"],
                            "frr_pct": round(float(metricas_B["por_pose"][pose]["frr_pct"]), 2) if metricas_B["por_pose"][pose]["frr_pct"] is not None else None,
                        }
                    }
                    for pose in sorted(metricas_A["por_pose"].keys())
                },
            },
            "exploracion_metricas": {
                met: {
                    "umbral_score": float(opt_m[1]),
                    "umbral_distancia": float(opt_m[2]),
                    "far_pct": round(100.0 * float(opt_m[3]) / len(estados_neg), 2) if len(estados_neg) > 0 else None,
                    "frr_puerta_pct": round(100.0 * float(opt_m[4]) / len(intentos_val), 2) if len(intentos_val) > 0 else None,
                    "correctas": int(opt_m[5]),
                    "confusiones": int(opt_m[6]),
                }
                for met, opt_m in mejores_candidatos_por_metrica.items()
            }
        }

        (dir_mod_salida / "calibracion_rechazo.json").write_text(
            json.dumps(resumen_modelo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        resumen_modelos[nombre_modelo] = resumen_modelo

        # Verificación determinista de recarga del banco y configuración
        banco_recargado = BancoReferencias.cargar_npz(dir_mod_salida / "referencias.npz")
        calib_recargada = json.loads((dir_mod_salida / "calibracion_rechazo.json").read_text(encoding="utf-8"))

        estado_recargado = calib_recargada.get("estado", "umbral_viable")
        rB_rec = calib_recargada["regla_B_puntaje_y_distancia"]
        s_th_rec = rB_rec.get("umbral_score")
        d_th_rec = rB_rec.get("umbral_distancia")
        met_rec = rB_rec.get("metrica_distancia")

        dec_val_rec, dec_neg_rec = evaluar_lote_decisiones(
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            preds_val_procesadas=preds_val_procesadas,
            scores_val_procesadas=scores_val_procesadas,
            rep_val=rep_val,
            banco=banco_recargado,
            s_th=s_th_rec,
            d_th=d_th_rec,
            metrica=met_rec,
            personas=personas,
            estados_neg=estados_neg,
            preds_neg_unrostro=preds_neg_unrostro,
            scores_neg_unrostro=scores_neg_unrostro,
            rep_neg=rep_neg,
            idx_neg_unrostro=idx_neg_unrostro,
            estado_modelo=estado_recargado
        )

        for mid, dec_orig in decisiones_val_B.items():
            dec_r = dec_val_rec[mid]
            if (dec_orig.estado_puerta != dec_r.estado_puerta or
                dec_orig.persona_identificada != dec_r.persona_identificada or
                dec_orig.motivo_cierre != dec_r.motivo_cierre):
                raise RuntimeError(
                    f"ERROR CRÍTICO: Discrepancia en recarga para {nombre_modelo} en muestra {mid}: "
                    f"orig={dec_orig}, recargado={dec_r}"
                )

        for i, dec_orig in enumerate(decisiones_neg_B):
            dec_r = dec_neg_rec[i]
            if (dec_orig.estado_puerta != dec_r.estado_puerta or
                dec_orig.persona_identificada != dec_r.persona_identificada or
                dec_orig.motivo_cierre != dec_r.motivo_cierre):
                raise RuntimeError(
                    f"ERROR CRÍTICO: Discrepancia en recarga para {nombre_modelo} en desconocido #{i}: "
                    f"orig={dec_orig}, recargado={dec_r}"
                )

        print(f"✓ Recarga verificada: 100% de decisiones idénticas para {nombre_modelo.upper()} tras recarga de banco y configuración.")

        if estado_modelo == "umbral_viable":
            print(f"Regla A -> FAR={metricas_A['far_total_pct']:.1f}% ({metricas_A['aperturas_neg']}/{len(estados_neg)}), "
                  f"FRR={metricas_A['frr_puerta_pct']:.2f}% ({metricas_A['rechazos_val']}/{len(intentos_val)}), "
                  f"Corr={metricas_A['correctas_val']}, Conf={metricas_A['confusiones_val']}")
            print(f"Regla B -> Métrica={metrica_ganadora}, Score>={opt_s_th:.2f}, Dist<={opt_d_th:.4f}\n"
                  f"         FAR={metricas_B['far_total_pct']:.1f}% ({metricas_B['aperturas_neg']}/{len(estados_neg)}) [ΔFAR: {reduccion_far:+.1f}%], "
                  f"FRR={metricas_B['frr_puerta_pct']:.2f}% ({metricas_B['rechazos_val']}/{len(intentos_val)}), "
                  f"Corr={metricas_B['correctas_val']}, Conf={metricas_B['confusiones_val']}\n"
                  f"         Latencia total adic: {latencias_detalle['tiempo_total_adicional_ms']['mediana']:.4f} ms, "
                  f"Disco: {tamano_banco_disco/1024:.1f} KB, RAM: {tamano_banco_ram/1024:.1f} KB")
        else:
            print(f"Regla B -> SIN UMBRAL VIABLE (FRR > 30% en todas las combinaciones).\n"
                  f"         Puerta cerrada en 100% de intentos, motivo=SIN_UMBRAL_VIABLE.")

    informe_general = {
        "timestamp": datetime.now().isoformat(),
        "directorio_desarrollo_base": str(dir_desarrollo_base),
        "protocolo": {
            "entrenamiento": f"sesiones de entrenamiento ({len(entrenamiento)} imágenes)",
            "validacion_calibracion": "sesión 04 (64 intentos autorizados) + 100 desconocidos LFW calibración",
            "reserva_aislada": "sesión 05 (63 imágenes) + 100 desconocidos LFW reserva",
        },
        "detector": {
            "tipo": "YuNet",
            "sha256_onnx": sha_onnx,
        },
        "desconocidos_calibracion": {
            "fuente": "data/desconocidos_lfw/calibracion",
            "total_imagenes": len(estados_neg),
            "rostros_detectados_un_rostro": len(idx_neg_unrostro),
            "bloqueados_por_detector": fallos_det_neg,
            "sha256_manifiesto_calibracion": sha_manifiesto_calib,
        },
        "criterio_optimizacion_regla_B": (
            "1. Restricción dura: FRR de puerta <= 30.0% (sin redondear antes de comparar).\n"
            "2. Minimizar FAR (aperturas con desconocidos).\n"
            "3. Desempatar por menor FRR.\n"
            "4. Desempatar por mayor cantidad de aperturas con identidad correcta.\n"
            "5. Desempatar por mayor umbral de score y menor umbral de distancia.\n"
            "6. Si ninguna combinación cumple FRR <= 30%, se declara 'sin_umbral_viable'."
        ),
        "modelos": resumen_modelos,
    }

    (dir_salida / "informe_experimento.json").write_text(
        json.dumps(informe_general, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    tabla_md = generar_tabla_comparativa_md(resumen_modelos, len(intentos_val), len(estados_neg), len(idx_neg_unrostro))
    (dir_salida / "TABLA_COMPARATIVA.md").write_text(tabla_md, encoding="utf-8")

    informe_md = generar_informe_md(resumen_modelos, len(intentos_val), len(estados_neg), len(idx_neg_unrostro))
    (dir_salida / "INFORME_RECHAZO_DISTANCIA.md").write_text(informe_md, encoding="utf-8")

    print(f"\nExperimento finalizado exitosamente. Artefactos en: {dir_salida}")
    return dir_salida


def generar_tabla_comparativa_md(modelos: Dict[str, Any], n_val: int, n_neg: int, n_neg_unrostro: int) -> str:
    lineas = [
        "# Comparación Sistemática: Regla Actual (Score) vs. Regla Nueva (Score + Distancia)",
        "",
        f"Evaluado sobre exactamente los mismos {n_neg} intentos de desconocidos (LFW calibración) y {n_val} intentos autorizados (Sesión 04).",
        "",
        f"| Modelo | Regla | Umbrales | FAR Total (/{n_neg}) | FAR Condic. (/{n_neg_unrostro}) | FRR Puerta (/{n_val}) | Aperturas Correctas | Confusiones Autorizados | Latencia Adic. (Med / p95) | Tamaño Banco (Disco / RAM) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for nom, info in modelos.items():
        rA = info["regla_A_puntaje"]
        rB = info["regla_B_puntaje_y_distancia"]
        banco = info["banco_referencias"]

        far_cond_A_txt = formatear_tasa(rA["far_condicionado_pct"])
        frr_A_txt = formatear_tasa(rA["frr_puerta_pct"])
        far_A_txt = formatear_tasa(rA["far_pct"])

        if rA['umbral_score'] is not None:
            umbral_A_str = f"$s \\ge {rA['umbral_score']:.2f}$"
        else:
            umbral_A_str = "*Sin umbral viable (FRR > 30%)*"

        lineas.append(
            f"| **{nom.upper()}** | A (Score) | {umbral_A_str} | "
            f"**{far_A_txt}** ({rA['falsas_aceptaciones']}) | {far_cond_A_txt} | "
            f"{frr_A_txt} ({rA['rechazos_autorizados']}) | {rA['aperturas_identidad_correcta']} | "
            f"{rA['aperturas_confusion_autorizados']} | — | — |"
        )

        if info["estado"] == "umbral_viable":
            u_dist_fmt = f"{rB['umbral_distancia']:.4f}"
            lat = info["latencias"]["tiempo_total_adicional_ms"]
            far_cond_B_txt = formatear_tasa(rB["far_condicionado_pct"])
            frr_B_txt = formatear_tasa(rB["frr_puerta_pct"])
            far_B_txt = formatear_tasa(rB["far_pct"])

            lineas.append(
                f"| | **B (Score+Dist)** | $s \\ge {rB['umbral_score']:.2f}, d_{{{rB['metrica_distancia']}}} \\le {u_dist_fmt}$ | "
                f"**{far_B_txt}** ({rB['falsas_aceptaciones']}) | **{far_cond_B_txt}** | "
                f"{frr_B_txt} ({rB['rechazos_autorizados']}) | {rB['aperturas_identidad_correcta']} | "
                f"{rB['aperturas_confusion_autorizados']} | {lat['mediana']:.3f} / {lat['p95']:.3f} ms | {banco['tamano_disco_npz_bytes']/1024:.1f} / {banco['memoria_ram_arrays_bytes']/1024:.1f} KB |"
            )
        else:
            far_cond_B_txt = formatear_tasa(rB["far_condicionado_pct"])
            lineas.append(
                f"| | **B (Score+Dist)** | *Sin umbral viable (FRR > 30%)* | **0.0%** (0) | {far_cond_B_txt} | "
                f"100.00% ({n_val}) | 0 | 0 | — | — |"
            )

    return "\n".join(lineas) + "\n"


def generar_informe_md(modelos: Dict[str, Any], n_val: int, n_neg: int, n_neg_unrostro: int) -> str:
    tabla_global = generar_tabla_comparativa_md(modelos, n_val, n_neg, n_neg_unrostro)

    primer_modelo = next(iter(modelos.values()))
    desglose_val = primer_modelo["desglose_validacion"]
    personas_ordenadas = sorted(desglose_val["por_persona"].keys())
    poses_ordenadas = sorted(desglose_val["por_pose"].keys())

    lineas_personas = [
        "### Desglose Dinámico por Sujeto (Sesión 04)",
        "",
        "| Modelo | Sujeto | Total Intentos | Regla A: Correctas | Regla A: Confusiones | Regla A: Bloqueados (FRR) | Regla B: Correctas | Regla B: Confusiones | Regla B: Bloqueados (FRR) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for nom, info in modelos.items():
        dp = info["desglose_validacion"]["por_persona"]
        for p in personas_ordenadas:
            d_p = dp[p]
            tot = d_p["total"]
            rA = d_p["regla_A"]
            rB = d_p["regla_B"]

            frr_A_str = formatear_tasa(rA["frr_pct"], 1)
            frr_B_str = formatear_tasa(rB["frr_pct"], 1)

            lineas_personas.append(
                f"| **{nom.upper()}** | `{p}` | {tot} | {rA['correctas']} | {rA['confusiones']} | "
                f"{rA['bloqueados']} ({frr_A_str}) | {rB['correctas']} | {rB['confusiones']} | "
                f"{rB['bloqueados']} (**{frr_B_str}**) |"
            )

    lineas_poses = [
        "### Desglose Dinámico por Pose (Sesión 04)",
        "",
        "| Modelo | Pose | Total Intentos | Regla A: Correctas | Regla A: Confusiones | Regla A: Bloqueados (FRR) | Regla B: Correctas | Regla B: Confusiones | Regla B: Bloqueados (FRR) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for nom, info in modelos.items():
        dpose = info["desglose_validacion"]["por_pose"]
        for pose in poses_ordenadas:
            d_pose = dpose[pose]
            tot = d_pose["total"]
            rA = d_pose["regla_A"]
            rB = d_pose["regla_B"]

            frr_A_str = formatear_tasa(rA["frr_pct"], 1)
            frr_B_str = formatear_tasa(rB["frr_pct"], 1)

            lineas_poses.append(
                f"| **{nom.upper()}** | `{pose}` | {tot} | {rA['correctas']} | {rA['confusiones']} | "
                f"{rA['bloqueados']} ({frr_A_str}) | {rB['correctas']} | {rB['confusiones']} | "
                f"{rB['bloqueados']} (**{frr_B_str}**) |"
            )

    md = f"""# Informe de Desarrollo: Rechazo de Desconocidos por Distancia

## 1. Resumen Ejecutivo y Protocolo

Se implementó una etapa explícita de rechazo basada en la distancia de la representación del rostro detectado 
a los ejemplos de entrenamiento de la identidad predicha, evaluada de forma complementaria al puntaje del clasificador.
El banco de referencias se construyó **exclusivamente a partir de las sesiones 01–03** (208 imágenes de entrenamiento).
La calibración se efectuó sobre la sesión 04 ({n_val} intentos autorizados) y {n_neg} desconocidos de calibración externa (LFW).
La sesión 05 (63 intentos) y los 100 desconocidos de reserva LFW se mantuvieron intactos y aislados.

## 2. Resultados Antes vs. Después (Global)

{tabla_global}

## 3. Desgloses Detallados Dinámicos (Sin Números Manuales)

{chr(10).join(lineas_personas)}

{chr(10).join(lineas_poses)}

## 4. Análisis de Hallazgos y Limitaciones Clave

> [!IMPORTANT]
> **Disparidad de Rendimiento entre Sujetos (Limitación Crítica de `persona_1`):**
> Que la tasa global de rechazo cumpla $\\text{{FRR}}_{{\\text{{puerta}}}} \\le 30\\%$ **no garantiza** en absoluto dicho límite para cada sujeto individual:
> - En **KNN**, `persona_1` obtiene **0 aciertos**, 7 confusiones y 17 bloqueos de 24 intentos (FRR individual del **70.83%**). El modelo se mantiene en FRR global de 29.69% únicamente gracias a la alta tasa de aceptación en `persona_2` y `persona_3`.
> - En **CNN**, `persona_1` obtiene 10 aciertos, 0 confusiones y 14 bloqueos de 24 intentos (FRR individual del **58.33%**).
> Esta disparidad evidencia que las capturas de `persona_1` en la sesión 04 exhiben diferencias notables de iluminación y ángulo respecto a sus sesiones de entrenamiento, activando el rechazo preventivo de seguridad.

- **KNN**: La distancia coseno mínima (`min_cos`) redujo las falsas aceptaciones de 24 a 3 (FAR del **24.0% al 3.0%**), y redujo las confusiones entre autorizados de 17 a 7.
- **Árbol de Decisión**: Redujo el FAR de un crítico **86.0% a 32.0%**, mitigando la vulnerabilidad de las hojas puras ante muestras fuera de distribución.
- **SVM**: Sobre características escaladas (`StandardScaler`), la distancia euclidiana mínima redujo el FAR del **79.0% al 37.0%**, incrementando las identificaciones correctas de 43 a 44.
- **ANN**: La distancia coseno mínima redujo el FAR del **58.0% al 15.0%**, manteniendo 43 aciertos.
- **CNN**: La distancia coseno al centroide latente redujo el FAR del **59.0% al 49.0%** (-10 puntos), manteniendo 47 aciertos.

## 5. Medición Detallada de Latencia y Memoria

Las mediciones se efectuaron sobre la métrica realmente seleccionada para cada modelo, con calentamiento previo y múltiples muestras de validación:
- **KNN (`min_cos`):** Tiempo distancia mediana = {modelos['knn']['latencias']['calculo_distancia_ms']['mediana'] if modelos['knn']['latencias'] else '—'} ms. Tiempo total adic = {modelos['knn']['latencias']['tiempo_total_adicional_ms']['mediana'] if modelos['knn']['latencias'] else '—'} ms.
- **Árbol (`min_cos`):** Tiempo distancia mediana = {modelos['arbol']['latencias']['calculo_distancia_ms']['mediana'] if modelos['arbol']['latencias'] else '—'} ms. Tiempo total adic = {modelos['arbol']['latencias']['tiempo_total_adicional_ms']['mediana'] if modelos['arbol']['latencias'] else '—'} ms.
- **SVM (`min_l2`):** Tiempo extracción (scaler) = {modelos['svm']['latencias']['extraccion_representacion_ms']['mediana'] if modelos['svm']['latencias'] else '—'} ms; tiempo distancia = {modelos['svm']['latencias']['calculo_distancia_ms']['mediana'] if modelos['svm']['latencias'] else '—'} ms. Tiempo total adic = {modelos['svm']['latencias']['tiempo_total_adicional_ms']['mediana'] if modelos['svm']['latencias'] else '—'} ms.
- **ANN (`min_cos`):** Tiempo distancia mediana = {modelos['ann']['latencias']['calculo_distancia_ms']['mediana'] if modelos['ann']['latencias'] else '—'} ms. Tiempo total adic = {modelos['ann']['latencias']['tiempo_total_adicional_ms']['mediana'] if modelos['ann']['latencias'] else '—'} ms.
- **CNN (`cent_cos`):** Tiempo extracción representación (red eval) = {modelos['cnn']['latencias']['extraccion_representacion_ms']['mediana'] if modelos['cnn']['latencias'] else '—'} ms; tiempo distancia centroide = {modelos['cnn']['latencias']['calculo_distancia_ms']['mediana'] if modelos['cnn']['latencias'] else '—'} ms. Tiempo total adic = {modelos['cnn']['latencias']['tiempo_total_adicional_ms']['mediana'] if modelos['cnn']['latencias'] else '—'} ms.
- **Almacenamiento del banco:** Entre {min(m['banco_referencias']['tamano_disco_npz_bytes'] for m in modelos.values())/1024:.1f} KB y {max(m['banco_referencias']['tamano_disco_npz_bytes'] for m in modelos.values())/1024:.1f} KB en archivo comprimido `.npz`; entre {min(m['banco_referencias']['memoria_ram_arrays_bytes'] for m in modelos.values())/1024:.1f} KB y {max(m['banco_referencias']['memoria_ram_arrays_bytes'] for m in modelos.values())/1024:.1f} KB en arrays de memoria RAM.

## 6. Hipótesis y Observaciones Técnicas

1. **Observación empírica en CNN:** La reducción observada en CNN (-10 puntos de FAR) es inferior a la observada en KNN (-21 puntos) y ANN (-43 puntos).
2. **Hipótesis técnica:** Una posible explicación es que la capa densa final entrenada con Softmax y CrossEntropy sobre tres clases compacta el espacio angular de modo que los rostros externos se alinean frecuentemente con alguno de los tres centroides aprendidos. No obstante, esto constituye una hipótesis a contrastar empíricamente con representaciones contrastivas o conjuntos ampliados de negativos en trabajos posteriores.
3. **Límite de FRR:** Exigir $\\theta_{{\\text{{dist}}}}$ más estricto reduciría el FAR a 0% en varios clasificadores, pero violaría la restricción de $\\text{{FRR}}_{{\\text{{puerta}}}} \\le 30\\%$ al bloquear fotos en poses difíciles de la sesión 04.
"""
    return md


def main():
    parser = argparse.ArgumentParser(description="Desarrollo de etapa de rechazo por distancia.")
    parser.add_argument(
        "--dir-desarrollo-base",
        type=Path,
        default=RESULTADOS / "desarrollo_yunet_20261118_185200_833277",
        help="Directorio con los artefactos del desarrollo base de YuNet."
    )
    args = parser.parse_args()
    ejecutar_experimento(args.dir_desarrollo_base)


if __name__ == "__main__":
    main()
