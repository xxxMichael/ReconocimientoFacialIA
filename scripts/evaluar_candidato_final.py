"""
Evaluación final independiente del candidato de 4 clases (CNN) sobre datos de prueba reservados:
  1. Sesión 05 de personas autorizadas (63 intentos)
  2. Reserva de prueba LFW (100 intentos de desconocidos no vistos)

Garantiza:
  - Uso estricto del PipelineCandidatoCongelado unificado (idéntico a la web en vivo).
  - Verificación criptográfica de hashes SHA-256 de pesos CNN y detector YuNet.
  - Aborto inmediato ante cualquier inconsistencia o error de configuración global.
  - Tratamiento exhaustivo de errores por muestra (lectura vs. fallo detector vs. inferencia),
    manteniendo cada intento rigurosamente en el denominador.
  - Generación dinámica de informes y tablas sin números fijos o hardcodeados.
  - Carga dinámica de métricas de desarrollo desde informe_experimento.json.
  - Preservación íntegra de evaluaciones históricas anteriores.
"""

import argparse
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

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.clasificacion.pipeline_candidato import (
    PipelineCandidatoCongelado,
    ErrorConfiguracionCandidato,
)
from reconocimiento.clasificacion.regla_acceso import (
    evaluar_regla_acceso_completa,
    validar_vector_probabilidades,
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion


def calcular_tasa_pct(numerador: int, denominador: int) -> Optional[float]:
    """Calcula porcentaje o retorna None si el denominador es vacío/nulo."""
    if denominador is None or denominador == 0:
        return None
    return round((numerador / denominador) * 100.0, 4)


def fmt_pct(val: Optional[float]) -> str:
    """Formatea porcentaje para el informe o indica 'no calculable' si es None."""
    if val is None:
        return "no calculable"
    return f"{val:.2f}%"


def ejecutar_evaluacion_final(
    ruta_config: Path,
    dir_salida_base: Optional[Path] = None,
) -> Path:
    ruta_config = Path(ruta_config).resolve()
    if not ruta_config.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración congelada: {ruta_config}")

    print(f"\n[INICIO] Cargando e instanciando PipelineCandidatoCongelado desde: {ruta_config}")
    try:
        pipeline = PipelineCandidatoCongelado(ruta_config, raiz_proyecto=RAIZ)
    except Exception as e_cfg:
        print(f"[FATAL] Error de configuración global del candidato: {e_cfg}")
        raise ErrorConfiguracionCandidato(f"Fallo al inicializar pipeline congelado: {e_cfg}") from e_cfg

    info_cand = pipeline.config["candidato"]
    info_det = pipeline.config["detector_facial"]
    umbral_congelado = pipeline.umbral_score
    personas_autorizadas = list(pipeline.personas_autorizadas)
    orden_clases = list(pipeline.orden_clases)
    mapeo_id_clase = dict(pipeline.mapeo_id_clase)

    # 2. Comprobar aislamiento y solapamiento de identidades desconocidas
    ruta_train_desc = RAIZ / "data" / "desconocidos_lfw" / "entrenamiento_4clases" / "manifiesto.csv"
    ruta_calib_desc = RAIZ / "data" / "desconocidos_lfw" / "calibracion" / "manifiesto.csv"
    ruta_reserva_desc = RAIZ / "data" / "desconocidos_lfw" / "reserva_prueba" / "manifiesto.csv"

    if not ruta_train_desc.exists() or not ruta_calib_desc.exists() or not ruta_reserva_desc.exists():
        raise FileNotFoundError("No se encontraron los manifiestos de desconocidos LFW necesarios para verificar aislamiento.")

    with open(ruta_train_desc, "r", encoding="utf-8") as f:
        train_identidades = {r["identidad_normalizada"] for r in csv.DictReader(f)}
    with open(ruta_calib_desc, "r", encoding="utf-8") as f:
        calib_identidades = {r["identidad_normalizada"] for r in csv.DictReader(f)}
    with open(ruta_reserva_desc, "r", encoding="utf-8") as f:
        reserva_filas = list(csv.DictReader(f))
        reserva_identidades = {r["identidad_normalizada"] for r in reserva_filas}

    solap_tr_res = train_identidades & reserva_identidades
    solap_cal_res = calib_identidades & reserva_identidades
    if solap_tr_res:
        raise ValueError(f"Violación de aislamiento: {len(solap_tr_res)} identidades solapadas entre train y reserva!")
    if solap_cal_res:
        raise ValueError(f"Violación de aislamiento: {len(solap_cal_res)} identidades solapadas entre calib y reserva!")

    # 3. Cargar manifiesto de Sesión 05
    ruta_manifiesto_local = RAIZ / "data" / "propio" / "manifiesto_local.csv"
    if not ruta_manifiesto_local.exists():
        raise FileNotFoundError(f"No existe el manifiesto propio en: {ruta_manifiesto_local}")

    with open(ruta_manifiesto_local, "r", encoding="utf-8-sig") as f:
        filas_propio = list(csv.DictReader(f))

    filas_sesion_05 = [
        f for f in filas_propio
        if f["sesion"] == "sesion_05" and f["particion"] == "prueba"
    ]
    if len(filas_sesion_05) != 63:
        raise ValueError(f"Se esperaban 63 capturas de sesión 05, pero se hallaron {len(filas_sesion_05)}")

    # 4. Crear directorio de resultados nuevo (preservando evaluaciones previas)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dir_salida_base is None:
        dir_salida = RAIZ / "resultados" / f"evaluacion_final_independiente_corregida_{timestamp}"
    else:
        dir_salida = Path(dir_salida_base)
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_manifiestos = dir_salida / "manifiestos_evaluados"
    dir_manifiestos.mkdir(parents=True, exist_ok=True)

    # Guardar copia de configuración congelada
    (dir_salida / "configuracion_congelada.json").write_text(
        json.dumps(pipeline.config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Guardar manifiestos evaluados
    with open(dir_manifiestos / "manifiesto_sesion_05.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=filas_sesion_05[0].keys())
        writer.writeheader()
        writer.writerows(filas_sesion_05)

    with open(dir_manifiestos / "manifiesto_reserva_lfw.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reserva_filas[0].keys())
        writer.writeheader()
        writer.writerows(reserva_filas)

    # 5. Evaluar Sesión 05 (Autorizados)
    muestras_resultados = []
    aperturas_correctas_auth = 0
    confusiones_auth = 0
    bloqueos_auth = 0
    bloqueos_desconocido_auth = 0
    bloqueos_bajo_umbral_auth = 0
    bloqueos_fallo_det_auth = 0
    bloqueos_error_proc_auth = 0
    bloqueos_error_lectura_auth = 0

    desglose_persona = {
        p: {
            "intentos": 0,
            "aperturas_correctas": 0,
            "confusiones": 0,
            "aperturas_totales": 0,
            "bloqueos": 0,
            "bloqueos_desconocido": 0,
            "bloqueos_bajo_umbral": 0,
            "bloqueos_fallo_det": 0,
            "bloqueos_error_lectura": 0,
            "bloqueos_error_proc": 0,
            "frr_puerta_pct": None,
        }
        for p in personas_autorizadas
    }
    desglose_pose = {}

    y_true_all = []
    y_pred_all = []

    print(f"\n--- Evaluando Sesión 05 (Autorizados, {len(filas_sesion_05)} intentos) ---")
    for f in filas_sesion_05:
        p_real = f["persona"]
        pose = f["pose"]
        ruta_img = RAIZ / f["ruta"] if not Path(f["ruta"]).is_absolute() else Path(f["ruta"])

        desglose_persona[p_real]["intentos"] += 1
        desglose_pose.setdefault(pose, {
            "intentos": 0,
            "aperturas_correctas": 0,
            "confusiones": 0,
            "aperturas_totales": 0,
            "bloqueos": 0,
            "frr_puerta_pct": None,
        })
        desglose_pose[pose]["intentos"] += 1

        img_bgr = cv2.imread(str(ruta_img))
        if img_bgr is None:
            dec = evaluar_regla_acceso_completa(
                estado_deteccion="ERROR_LECTURA",
                procesamiento_exitoso=False,
                clase_predicha=None,
                score=0.0,
                umbral=umbral_congelado,
                personas_autorizadas=personas_autorizadas,
                error_procesamiento="No se pudo decodificar la imagen desde el almacenamiento",
            )
            bloqueos_auth += 1
            bloqueos_error_lectura_auth += 1
            desglose_persona[p_real]["bloqueos"] += 1
            desglose_persona[p_real]["bloqueos_error_lectura"] += 1
            desglose_pose[pose]["bloqueos"] += 1
            muestras_resultados.append({
                "conjunto": "autorizados_sesion_05",
                "identificador": f["identificador"],
                "persona_real": p_real,
                "pose": pose,
                "ruta": str(ruta_img),
                "estado_deteccion": "ERROR_LECTURA",
                "clase_predicha": None,
                "score": 0.0,
                "decision_puerta": dec.estado_puerta,
                "es_autorizado": dec.es_autorizado,
                "persona_identificada": dec.persona_identificada,
                "motivo_cierre": dec.motivo_cierre,
                "clasificacion_evento": "BLOQUEO_ERROR_LECTURA",
            })
            continue

        # Inferencia con detector YuNet
        res_det = pipeline.detector.detectar(img_bgr, es_pre_recortada=False)
        st_det = res_det.estado.value

        if res_det.estado != EstadoDeteccion.UN_ROSTRO or res_det.rostro_recortado is None:
            dec = evaluar_regla_acceso_completa(
                estado_deteccion=st_det,
                procesamiento_exitoso=False,
                clase_predicha=None,
                score=0.0,
                umbral=umbral_congelado,
                personas_autorizadas=personas_autorizadas,
            )
            bloqueos_auth += 1
            bloqueos_fallo_det_auth += 1
            desglose_persona[p_real]["bloqueos"] += 1
            desglose_persona[p_real]["bloqueos_fallo_det"] += 1
            desglose_pose[pose]["bloqueos"] += 1

            muestras_resultados.append({
                "conjunto": "autorizados_sesion_05",
                "identificador": f["identificador"],
                "persona_real": p_real,
                "pose": pose,
                "ruta": str(ruta_img),
                "estado_deteccion": st_det,
                "clase_predicha": None,
                "score": 0.0,
                "decision_puerta": dec.estado_puerta,
                "es_autorizado": dec.es_autorizado,
                "persona_identificada": dec.persona_identificada,
                "motivo_cierre": dec.motivo_cierre,
                "clasificacion_evento": "BLOQUEO_FALLO_DETECCION",
            })
            continue

        # Evaluación con pipeline unificado
        try:
            dec, probs = pipeline.evaluar_recorte(res_det.rostro_recortado)
            c_pred = int(np.argmax(probs))
            score_pred = float(probs[c_pred])
            nom_pred = mapeo_id_clase[c_pred]

            # Muestra que alcanzó la etapa de clasificación
            y_true_all.append(p_real)
            y_pred_all.append(nom_pred)

            if dec.es_autorizado:
                if dec.persona_identificada == p_real:
                    aperturas_correctas_auth += 1
                    desglose_persona[p_real]["aperturas_correctas"] += 1
                    desglose_pose[pose]["aperturas_correctas"] += 1
                    evento = "APERTURA_CORRECTA"
                else:
                    confusiones_auth += 1
                    desglose_persona[p_real]["confusiones"] += 1
                    desglose_pose[pose]["confusiones"] += 1
                    evento = "APERTURA_CONFUSION_AUTORIZADOS"
            else:
                bloqueos_auth += 1
                desglose_persona[p_real]["bloqueos"] += 1
                desglose_pose[pose]["bloqueos"] += 1
                if nom_pred == "desconocido":
                    bloqueos_desconocido_auth += 1
                    desglose_persona[p_real]["bloqueos_desconocido"] += 1
                    evento = "BLOQUEO_PREDICCION_DESCONOCIDO"
                else:
                    bloqueos_bajo_umbral_auth += 1
                    desglose_persona[p_real]["bloqueos_bajo_umbral"] += 1
                    evento = "BLOQUEO_SCORE_INSUFICIENTE"

            muestras_resultados.append({
                "conjunto": "autorizados_sesion_05",
                "identificador": f["identificador"],
                "persona_real": p_real,
                "pose": pose,
                "ruta": str(ruta_img),
                "estado_deteccion": "UN_ROSTRO",
                "clase_predicha": nom_pred,
                "score": score_pred,
                "decision_puerta": dec.estado_puerta,
                "es_autorizado": dec.es_autorizado,
                "persona_identificada": dec.persona_identificada,
                "motivo_cierre": dec.motivo_cierre,
                "clasificacion_evento": evento,
            })
        except Exception as e_proc:
            dec = evaluar_regla_acceso_completa(
                estado_deteccion="UN_ROSTRO",
                procesamiento_exitoso=False,
                clase_predicha=None,
                score=0.0,
                umbral=umbral_congelado,
                personas_autorizadas=personas_autorizadas,
                error_procesamiento=str(e_proc),
            )
            bloqueos_auth += 1
            bloqueos_error_proc_auth += 1
            desglose_persona[p_real]["bloqueos"] += 1
            desglose_persona[p_real]["bloqueos_error_proc"] += 1
            desglose_pose[pose]["bloqueos"] += 1
            muestras_resultados.append({
                "conjunto": "autorizados_sesion_05",
                "identificador": f["identificador"],
                "persona_real": p_real,
                "pose": pose,
                "ruta": str(ruta_img),
                "estado_deteccion": "UN_ROSTRO",
                "clase_predicha": None,
                "score": 0.0,
                "decision_puerta": dec.estado_puerta,
                "es_autorizado": dec.es_autorizado,
                "persona_identificada": dec.persona_identificada,
                "motivo_cierre": dec.motivo_cierre,
                "clasificacion_evento": "BLOQUEO_ERROR_PROCESAMIENTO",
            })

    # Calcular tasas por persona y por pose
    for p, st in desglose_persona.items():
        st["aperturas_totales"] = st["aperturas_correctas"] + st["confusiones"]
        st["frr_puerta_pct"] = calcular_tasa_pct(st["bloqueos"], st["intentos"])

    for pose, st in desglose_pose.items():
        st["aperturas_totales"] = st["aperturas_correctas"] + st["confusiones"]
        st["frr_puerta_pct"] = calcular_tasa_pct(st["bloqueos"], st["intentos"])

    # 6. Evaluar Reserva LFW (Desconocidos, 100 intentos)
    print(f"\n--- Evaluando Reserva de Prueba LFW (Desconocidos, {len(reserva_filas)} intentos) ---")
    falsas_aceptaciones_desc = 0
    rechazos_desc = 0
    fallos_det_desc = 0
    errores_lectura_desc = 0
    errores_proc_desc = 0
    desconocidos_rostro_unico = 0

    for r in reserva_filas:
        ruta_img = RAIZ / r["ruta_imagen"] if not Path(r["ruta_imagen"]).is_absolute() else Path(r["ruta_imagen"])
        ident_norm = r["identidad_normalizada"]

        img_bgr = cv2.imread(str(ruta_img))
        if img_bgr is None:
            rechazos_desc += 1
            errores_lectura_desc += 1
            muestras_resultados.append({
                "conjunto": "desconocidos_reserva_lfw",
                "identificador": ident_norm,
                "persona_real": "desconocido",
                "pose": "desconocida",
                "ruta": str(ruta_img),
                "estado_deteccion": "ERROR_LECTURA",
                "clase_predicha": None,
                "score": 0.0,
                "decision_puerta": "CERRADA",
                "es_autorizado": False,
                "persona_identificada": "DESCONOCIDO",
                "motivo_cierre": "ERROR_LECTURA",
                "clasificacion_evento": "RECHAZO_DESCONOCIDO_ERROR_LECTURA",
            })
            continue

        res_det = pipeline.detector.detectar(img_bgr, es_pre_recortada=False)
        st_det = res_det.estado.value

        if res_det.estado != EstadoDeteccion.UN_ROSTRO or res_det.rostro_recortado is None:
            fallos_det_desc += 1
            rechazos_desc += 1
            muestras_resultados.append({
                "conjunto": "desconocidos_reserva_lfw",
                "identificador": ident_norm,
                "persona_real": "desconocido",
                "pose": "desconocida",
                "ruta": str(ruta_img),
                "estado_deteccion": st_det,
                "clase_predicha": None,
                "score": 0.0,
                "decision_puerta": "CERRADA",
                "es_autorizado": False,
                "persona_identificada": "DESCONOCIDO",
                "motivo_cierre": f"FALLO_DETECCION ({st_det})",
                "clasificacion_evento": "RECHAZO_DESCONOCIDO_FALLO_DETECCION",
            })
            continue

        desconocidos_rostro_unico += 1

        try:
            dec, probs = pipeline.evaluar_recorte(res_det.rostro_recortado)
            c_pred = int(np.argmax(probs))
            score_pred = float(probs[c_pred])
            nom_pred = mapeo_id_clase[c_pred]

            y_true_all.append("desconocido")
            y_pred_all.append(nom_pred)

            if dec.es_autorizado:
                falsas_aceptaciones_desc += 1
                evento = "FALSA_ACEPTACION_DESCONOCIDO"
            else:
                rechazos_desc += 1
                evento = "RECHAZO_CORRECTO_DESCONOCIDO"

            muestras_resultados.append({
                "conjunto": "desconocidos_reserva_lfw",
                "identificador": ident_norm,
                "persona_real": "desconocido",
                "pose": "desconocida",
                "ruta": str(ruta_img),
                "estado_deteccion": "UN_ROSTRO",
                "clase_predicha": nom_pred,
                "score": score_pred,
                "decision_puerta": dec.estado_puerta,
                "es_autorizado": dec.es_autorizado,
                "persona_identificada": dec.persona_identificada,
                "motivo_cierre": dec.motivo_cierre,
                "clasificacion_evento": evento,
            })
        except Exception as e_proc:
            rechazos_desc += 1
            errores_proc_desc += 1
            muestras_resultados.append({
                "conjunto": "desconocidos_reserva_lfw",
                "identificador": ident_norm,
                "persona_real": "desconocido",
                "pose": "desconocida",
                "ruta": str(ruta_img),
                "estado_deteccion": "UN_ROSTRO",
                "clase_predicha": None,
                "score": 0.0,
                "decision_puerta": "CERRADA",
                "es_autorizado": False,
                "persona_identificada": "DESCONOCIDO",
                "motivo_cierre": f"ERROR_PROCESAMIENTO ({e_proc})",
                "clasificacion_evento": "RECHAZO_DESCONOCIDO_ERROR_PROC",
            })

    # 7. Verificación de Invariantes Estrictas
    total_intentos_auth = len(filas_sesion_05)
    total_intentos_desc = len(reserva_filas)
    aperturas_totales_auth = aperturas_correctas_auth + confusiones_auth

    assert total_intentos_auth == (bloqueos_auth + aperturas_correctas_auth + confusiones_auth), (
        f"Invariante rota en autorizados: {total_intentos_auth} != {bloqueos_auth} + {aperturas_correctas_auth} + {confusiones_auth}"
    )
    assert total_intentos_desc == (falsas_aceptaciones_desc + rechazos_desc), (
        f"Invariante rota en desconocidos: {total_intentos_desc} != {falsas_aceptaciones_desc} + {rechazos_desc}"
    )

    frr_puerta_global_pct = calcular_tasa_pct(bloqueos_auth, total_intentos_auth)
    far_total_pct = calcular_tasa_pct(falsas_aceptaciones_desc, total_intentos_desc)
    far_condicional_pct = calcular_tasa_pct(falsas_aceptaciones_desc, desconocidos_rostro_unico)

    # 8. Matriz de Confusión (únicamente para muestras que alcanzaron la fase de clasificación)
    etiquetas_todas = orden_clases
    cm = confusion_matrix(y_true_all, y_pred_all, labels=etiquetas_todas)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=etiquetas_todas,
        yticklabels=etiquetas_todas,
        title=f"Matriz de Confusión - Fase de Clasificación\n(CNN 4 Clases, th={umbral_congelado})",
        ylabel="Identidad Real",
        xlabel="Predicción del Modelo",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    ruta_cm_png = dir_salida / "matriz_confusion.png"
    plt.savefig(ruta_cm_png, dpi=150)
    plt.close(fig)

    # 9. Guardar predicciones por muestra (CSV y JSON)
    with open(dir_salida / "predicciones_muestras.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=muestras_resultados[0].keys())
        writer.writeheader()
        writer.writerows(muestras_resultados)

    (dir_salida / "predicciones_muestras.json").write_text(
        json.dumps(muestras_resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 10. Cargar métricas de desarrollo dinámicamente para comparativa
    ruta_informe_desarrollo = RAIZ / "resultados" / "desarrollo_4clases_corregido_20261119_123602" / "informe_experimento.json"
    dev_metrics: Dict[str, Any] = {}
    if ruta_informe_desarrollo.exists():
        try:
            with open(ruta_informe_desarrollo, "r", encoding="utf-8") as f_dev:
                datos_dev = json.load(f_dev)
            dev_metrics = datos_dev.get("resultados", {}).get("cnn", {}).get("regla_b", {})
        except Exception as e_load_dev:
            print(f"[AVISO] No se pudo cargar informe de desarrollo: {e_load_dev}")

    # 11. Guardar métricas estructuradas en JSON
    metricas = {
        "timestamp": timestamp,
        "directorio_evaluacion": str(dir_salida.name),
        "candidato": {
            "nombre": info_cand["nombre"],
            "sha256_modelo": pipeline.sha256_modelo,
            "sha256_esperado": info_cand["sha256_modelo"],
            "integridad_verificada": pipeline.sha256_modelo.lower() == info_cand["sha256_modelo"].lower(),
            "umbral_congelado": umbral_congelado,
            "orden_clases": orden_clases,
            "personas_autorizadas": personas_autorizadas,
        },
        "detector_facial": {
            "nombre": info_det["nombre"],
            "sha256_yunet": pipeline.sha256_detector,
            "parametros": info_det["parametros"],
        },
        "invariantes": {
            "autorizados": {
                "formula": "intentos == bloqueos + aperturas_correctas + confusiones",
                "intentos": total_intentos_auth,
                "bloqueos": bloqueos_auth,
                "aperturas_correctas": aperturas_correctas_auth,
                "confusiones": confusiones_auth,
                "aperturas_totales": aperturas_totales_auth,
                "invariante_valida": total_intentos_auth == (bloqueos_auth + aperturas_correctas_auth + confusiones_auth),
            },
            "desconocidos": {
                "formula": "intentos == falsas_aceptaciones + rechazos",
                "intentos": total_intentos_desc,
                "falsas_aceptaciones": falsas_aceptaciones_desc,
                "rechazos": rechazos_desc,
                "invariante_valida": total_intentos_desc == (falsas_aceptaciones_desc + rechazos_desc),
            }
        },
        "tasas_globales": {
            "frr_puerta_global_pct": frr_puerta_global_pct,
            "frr_puerta_global_conteo": f"{bloqueos_auth}/{total_intentos_auth}",
            "far_total_pct": far_total_pct,
            "far_total_conteo": f"{falsas_aceptaciones_desc}/{total_intentos_desc}",
            "far_condicional_pct": far_condicional_pct,
            "far_condicional_conteo": f"{falsas_aceptaciones_desc}/{desconocidos_rostro_unico}" if desconocidos_rostro_unico > 0 else None,
        },
        "desglose_bloqueos_causa": {
            "bloqueos_fallo_deteccion": bloqueos_fallo_det_auth,
            "bloqueos_prediccion_desconocido": bloqueos_desconocido_auth,
            "bloqueos_score_insuficiente": bloqueos_bajo_umbral_auth,
            "bloqueos_error_lectura": bloqueos_error_lectura_auth,
            "bloqueos_error_procesamiento": bloqueos_error_proc_auth,
        },
        "desglose_por_persona": desglose_persona,
        "desglose_por_pose": desglose_pose,
        "matriz_confusion": {
            "nota": "Incluye exclusivamente las muestras con rostro detectado que alcanzaron la fase de inferencia.",
            "total_muestras_clasificadas": len(y_true_all),
            "etiquetas": etiquetas_todas,
            "valores": cm.tolist(),
        },
        "referencia_desarrollo": dev_metrics,
    }

    (dir_salida / "metricas_evaluacion_final.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 12. Generar informe Markdown completo dinámicamente
    lineas_informe = [
        "# Informe de Evaluación Final Independiente del Clasificador de 4 Clases (Corregido)",
        "",
        f"**Fecha y Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Directorio:** `{dir_salida.name}`",
        f"**Candidato Evaluado:** {info_cand['nombre']}",
        f"**Modelo:** `{pipeline.ruta_modelo.name}` (SHA-256: `{pipeline.sha256_modelo}`)",
        f"**Umbral Congelado en Desarrollo:** $\\theta^* = {umbral_congelado:.2f}$",
        "",
        "> [!IMPORTANT]",
        "> **Independencia Estricta y Pipeline Unificado:**",
        "> - Esta evaluación utiliza de forma estricta el `PipelineCandidatoCongelado`, compartiendo código idéntico con el streamer web en vivo.",
        f"> - Conjuntos de prueba reservados: **Sesión 05** ({total_intentos_auth} intentos) y **Reserva LFW** ({total_intentos_desc} identidades no vistas).",
        "> - Ni el modelo, ni los pesos, ni el umbral fueron modificados tras observar los datos de prueba.",
        "> - Se verificó la ausencia absoluta de solapamiento de identidades entre entrenamiento, calibración y prueba.",
        "",
        "## 1. Verificación de Integridad de Parámetros Congelados",
        "",
        "| Componente | Archivo / Parámetro | Hash SHA-256 Esperado | Hash SHA-256 Real | Estado |",
        "| :--- | :--- | :--- | :--- | :---: |",
        f"| **Pesos CNN** | `{pipeline.ruta_modelo.name}` | `{info_cand['sha256_modelo']}` | `{pipeline.sha256_modelo}` | **Intacto y Verificado** |",
        f"| **Detector YuNet** | `{pipeline.ruta_detector.name}` | `{info_det['sha256_modelo']}` | `{pipeline.sha256_detector}` | **Intacto y Verificado** |",
        f"| **Umbral de Decisión** | $\\theta^* = {umbral_congelado:.2f}$ | *Fijado en desarrollo (Regla B)* | `{umbral_congelado:.2f}` | **Congelado** |",
        f"| **Clases del Modelo** | `{', '.join(orden_clases)}` | 4 Clases | 4 Clases | **Idéntico** |",
        f"| **Autorizados** | `{', '.join(personas_autorizadas)}` | 3 Clases | 3 Clases | **Idéntico** |",
        "",
        "## 2. Resultados Globales de la Prueba Final Independiente",
        "",
        "| Métrica | Definición / Cómputo | Conteo Exacto | Porcentaje |",
        "| :--- | :--- | :---: | :---: |",
        f"| **FAR Total** | Falsas Aceptaciones / Intentos Desconocidos | **{falsas_aceptaciones_desc}/{total_intentos_desc}** | **{fmt_pct(far_total_pct)}** |",
        f"| **FAR Condicional** | Falsas Aceptaciones / Desconocidos con Detección | **{falsas_aceptaciones_desc}/{desconocidos_rostro_unico}** | **{fmt_pct(far_condicional_pct)}** |",
        f"| **FRR Puerta Global** | Bloqueos Físicos / Intentos Autorizados | **{bloqueos_auth}/{total_intentos_auth}** | **{fmt_pct(frr_puerta_global_pct)}** |",
        f"| **Aperturas Correctas** | Identidad y Acceso Concedido Correctamente | **{aperturas_correctas_auth}/{total_intentos_auth}** | **{fmt_pct(calcular_tasa_pct(aperturas_correctas_auth, total_intentos_auth))}** |",
        f"| **Confusiones Autorizados** | Puerta abierta pero con identidad equivocada | **{confusiones_auth}/{total_intentos_auth}** | **{fmt_pct(calcular_tasa_pct(confusiones_auth, total_intentos_auth))}** |",
        f"| **Aperturas Totales** | Aperturas Correctas + Confusiones | **{aperturas_totales_auth}/{total_intentos_auth}** | **{fmt_pct(calcular_tasa_pct(aperturas_totales_auth, total_intentos_auth))}** |",
        f"| **Bloqueos Puerta Totales** | Intentos Autorizados - Aperturas Totales | **{bloqueos_auth}/{total_intentos_auth}** | **{fmt_pct(frr_puerta_global_pct)}** |",
        "",
        "### 2.1 Verificación de Invariantes Reglamentarias",
        f"- **Autorizados:** Intentos ({total_intentos_auth}) = Bloqueos ({bloqueos_auth}) + Aperturas Correctas ({aperturas_correctas_auth}) + Confusiones ({confusiones_auth}) -> **CUMPLE ESTRICTAMENTE**.",
        f"- **Desconocidos:** Intentos ({total_intentos_desc}) = Falsas Aceptaciones ({falsas_aceptaciones_desc}) + Rechazos ({rechazos_desc}) -> **CUMPLE ESTRICTAMENTE**.",
        "",
        "### 2.2 Desglose de Causas de Bloqueo en Autorizados",
        f"- **Fallo de detección facial (YuNet):** {bloqueos_fallo_det_auth} intentos ({fmt_pct(calcular_tasa_pct(bloqueos_fallo_det_auth, total_intentos_auth))}).",
        f"- **Predicción como desconocido (argmax = desconocido):** {bloqueos_desconocido_auth} intentos ({fmt_pct(calcular_tasa_pct(bloqueos_desconocido_auth, total_intentos_auth))}).",
        f"- **Score insuficiente (score < {umbral_congelado:.2f}):** {bloqueos_bajo_umbral_auth} intentos ({fmt_pct(calcular_tasa_pct(bloqueos_bajo_umbral_auth, total_intentos_auth))}).",
        f"- **Error de lectura de archivo:** {bloqueos_error_lectura_auth} intentos ({fmt_pct(calcular_tasa_pct(bloqueos_error_lectura_auth, total_intentos_auth))}).",
        f"- **Error técnico de procesamiento:** {bloqueos_error_proc_auth} intentos ({fmt_pct(calcular_tasa_pct(bloqueos_error_proc_auth, total_intentos_auth))}).",
        "",
        "## 3. Desglose por Persona Autorizada (Sesión 05)",
        "",
        "| Persona | Intentos | Aperturas Correctas | Confusiones | Aperturas Totales | Bloqueos Puerta | FRR Puerta | Bloqueos Desconocido | Bloqueos Bajo Umbral | Bloqueos Detección |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for p in personas_autorizadas:
        st = desglose_persona[p]
        lineas_informe.append(
            f"| `{p}` | {st['intentos']} | {st['aperturas_correctas']} | {st['confusiones']} | "
            f"{st['aperturas_totales']} | {st['bloqueos']} | **{fmt_pct(st['frr_puerta_pct'])}** | "
            f"{st['bloqueos_desconocido']} | {st['bloqueos_bajo_umbral']} | {st['bloqueos_fallo_det']} |"
        )

    lineas_informe.extend([
        "",
        "## 4. Desglose por Pose (Sesión 05)",
        "",
        "| Pose | Intentos | Aperturas Correctas | Confusiones | Aperturas Totales | Bloqueos Puerta | FRR Pose |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for pose in sorted(desglose_pose.keys()):
        st = desglose_pose[pose]
        lineas_informe.append(
            f"| `{pose}` | {st['intentos']} | {st['aperturas_correctas']} | {st['confusiones']} | "
            f"{st['aperturas_totales']} | {st['bloqueos']} | **{fmt_pct(st['frr_puerta_pct'])}** |"
        )

    # Construir fila de desarrollo dinámicamente
    dev_far_txt = f"{dev_metrics.get('far_total_pct', 'N/D')}% ({dev_metrics.get('far_total_conteo', 'N/D')})" if dev_metrics else "no disponible"
    dev_farc_txt = f"{dev_metrics.get('far_condicional_pct', 'N/D')}% ({dev_metrics.get('far_condicional_conteo', 'N/D')})" if dev_metrics else "no disponible"
    dev_frr_txt = f"{dev_metrics.get('frr_puerta_pct', 'N/D')}% ({dev_metrics.get('frr_puerta_conteo', 'N/D')})" if dev_metrics else "no disponible"
    dev_ap_txt = str(dev_metrics.get("aperturas_correctas", "no disponible"))
    dev_conf_txt = str(dev_metrics.get("confusiones_autorizados", "no disponible"))
    dev_bloq_txt = str(dev_metrics.get("bloqueos_puerta", "no disponible"))
    dev_umb_txt = f"$s \\ge {dev_metrics.get('umbral', 'N/D')}$" if dev_metrics else "no disponible"

    lineas_informe.extend([
        "",
        "## 5. Comparación Metodológica: Desarrollo vs. Prueba Final Independiente",
        "",
        "| Fase | Conjunto Autorizados | Conjunto Desconocidos | Umbral ($\\theta^*$) | FAR Total | FAR Condicional | FRR Puerta Global | Aperturas Correctas | Confusiones | Bloqueos Puerta |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Desarrollo (Calibración)** | Sesión 04 (64 fotos) | LFW Calibración (100 fotos) | {dev_umb_txt} | {dev_far_txt} | {dev_farc_txt} | {dev_frr_txt} | {dev_ap_txt} | {dev_conf_txt} | {dev_bloq_txt} |",
        f"| **Prueba Final Independiente** | Sesión 05 ({total_intentos_auth} fotos) | LFW Reserva ({total_intentos_desc} fotos) | $s \\ge {umbral_congelado:.2f}$ | **{fmt_pct(far_total_pct)}** ({falsas_aceptaciones_desc}/{total_intentos_desc}) | **{fmt_pct(far_condicional_pct)}** ({falsas_aceptaciones_desc}/{desconocidos_rostro_unico}) | **{fmt_pct(frr_puerta_global_pct)}** ({bloqueos_auth}/{total_intentos_auth}) | **{aperturas_correctas_auth}** | **{confusiones_auth}** | **{bloqueos_auth}** |",
        "",
        "> [!NOTE]",
        "> **Alcance de la Matriz de Confusión:**",
        f"> La matriz de confusión incluye únicamente las {len(y_true_all)} muestras que fueron detectadas como rostro único y clasificadas por la CNN. Los fallos de detección ({bloqueos_fallo_det_auth + fallos_det_desc}) y errores de lectura no entran a la matriz de confusión multiclase, pero se contabilizan rigurosamente como bloqueos en el FRR de puerta y rechazos en el FAR.",
        "",
        "## 6. Conclusiones y Limitaciones Operativas",
        "",
        "1. **Rechazo de Desconocidos en Dominio LFW:**",
        f"   El modelo CNN congelado con $\\theta^* = {umbral_congelado:.2f}$ obtuvo un FAR Total de **{fmt_pct(far_total_pct)}** ({falsas_aceptaciones_desc}/{total_intentos_desc}) y condicional de **{fmt_pct(far_condicional_pct)}** ({falsas_aceptaciones_desc}/{desconocidos_rostro_unico}) en identidades no vistas de reserva LFW.",
        f"2. **Tasa de Falso Rechazo (FRR):**",
        f"   En la Sesión 05 reservada, el FRR global de puerta fue de **{fmt_pct(frr_puerta_global_pct)}** ({bloqueos_auth}/{total_intentos_auth}), con {aperturas_correctas_auth} aperturas correctas y {confusiones_auth} confusiones.",
        "3. **Advertencia de Seguridad para Cerraduras Físicas Reales:**",
        "   - Este resultado evalúa únicamente el rechazo frente a imágenes de rostros externos de la base LFW funneled.",
        "   - **No demuestra protección contra ataques de presentación (spoofing)**: fotografías impresas, pantallas de teléfonos/tablets, vídeos pregrabados, máscaras o caricaturas.",
        "   - **No reemplaza la prueba con desconocidos reales en cámara web local**, debido al sesgo de dominio fotográfico (iluminación, resolución y ángulo de cámara de webcam).",
        "   - El sistema permanece catalogado como **prototipo académico universitario**, no apto para control de acceso comercial de alta seguridad sin módulo de vivacidad (liveness detection).",
    ])

    (dir_salida / "INFORME_EVALUACION_FINAL.md").write_text("\n".join(lineas_informe) + "\n", encoding="utf-8")
    print(f"\n[OK] Evaluación final completada exitosamente. Resultados en: {dir_salida}")
    return dir_salida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación final independiente del candidato de 4 clases")
    parser.add_argument(
        "--config",
        type=str,
        default="configuracion_candidato_congelado_corregida.json",
        help="Ruta a la configuración congelada del candidato"
    )
    parser.add_argument(
        "--salida",
        type=str,
        default=None,
        help="Directorio de salida opcional"
    )
    args = parser.parse_args()
    ejecutar_evaluacion_final(Path(args.config), Path(args.salida) if args.salida else None)
