"""
Script de evaluación independiente de candidatos congelados sobre fotografías reales de desconocidos.

Evalúa:
1. CNN original de cuatro clases (configuración corregida, umbral 0.46)
2. CNN alineada de cuatro clases (configuración alineada, umbral 0.84)
sobre los 75 intentos adquiridos de 5 voluntarios externos reales.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Asegurar que 'src' esté en sys.path
RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconocimiento.adquisicion.propio import ManifiestoDataset, RegistroCaptura, calcular_hash_archivo
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.detector import EstadoDeteccion


def calcular_sha256_bytes(datos: bytes) -> str:
    h = hashlib.sha256()
    h.update(datos)
    return h.hexdigest().lower()


def ejecutar_evaluacion_comparativa():
    ahora_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_salida = RAIZ / "resultados" / f"evaluacion_desconocidos_reales_{ahora_str}"
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_laminas = dir_salida / "laminas"
    dir_laminas.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" EVALUACIÓN COMPARATIVA DE CANDIDATOS CONGELADOS: DESCONOCIDOS REALES")
    print("=" * 75)
    print(f" Directorio de resultados: {dir_salida}")

    # 1. Cargar manifiesto con registros externos
    ruta_manifiesto = RAIZ / "data" / "propio" / "manifiesto_local.csv"
    if not ruta_manifiesto.exists():
        ruta_manifiesto = RAIZ / "data" / "propio" / "manifiesto.csv"

    manifiesto = ManifiestoDataset(ruta_manifiesto)
    registros_desconocidos = [
        r for r in manifiesto.registros
        if r.procedencia == "portal_web_externo" and r.particion == "prueba"
    ]

    print(f" Total de capturas externas encontradas en manifiesto: {len(registros_desconocidos)}")
    if not registros_desconocidos:
        raise RuntimeError("No se encontraron registros de capturas externas en el manifiesto.")

    # Ordenar determinísticamente por sujeto, pose, distancia
    orden_poses = {"frontal": 0, "izquierda": 1, "derecha": 2, "arriba": 3, "abajo": 4}
    orden_dist = {"cercana": 0, "media": 1, "lejana": 2}
    registros_desconocidos.sort(
        key=lambda r: (r.persona, orden_poses.get(r.pose, 99), orden_dist.get(r.distancia or "", 99))
    )

    sujetos_unicos = sorted(list({r.persona for r in registros_desconocidos}))
    print(f" Participantes únicos ({len(sujetos_unicos)}): {sujetos_unicos}")

    # 2. Cargar ambos pipelines candidatos
    ruta_cfg_orig = RAIZ / "configuracion_candidato_congelado_corregida.json"
    ruta_cfg_alin = RAIZ / "configuracion_candidato_alineado.json"

    pipe_orig = PipelineCandidatoCongelado(ruta_cfg_orig, raiz_proyecto=RAIZ)
    pipe_alin = PipelineCandidatoCongelado(ruta_cfg_alin, raiz_proyecto=RAIZ)

    print(f"\n[Modelo 1 - Original]")
    print(f"  Nombre: {pipe_orig.nombre}")
    print(f"  Umbral: {pipe_orig.umbral_score}")
    print(f"  Alineación: {pipe_orig.usar_alineacion}")

    print(f"\n[Modelo 2 - Alineado]")
    print(f"  Nombre: {pipe_alin.nombre}")
    print(f"  Umbral: {pipe_alin.umbral_score}")
    print(f"  Alineación: {pipe_alin.usar_alineacion}")
    print("-" * 75)

    # 3. Evaluar cada imagen
    resultados_orig = []
    resultados_alin = []
    pareados = []

    # Estructura para registrar decisiones individuales
    filas_csv_individuales = []

    for i, reg in enumerate(registros_desconocidos, 1):
        ruta_img = RAIZ / reg.ruta
        if not ruta_img.exists():
            raise FileNotFoundError(f"Archivo de imagen no encontrado: {ruta_img}")

        frame_bgr = cv2.imread(str(ruta_img))
        if frame_bgr is None:
            raise ValueError(f"No se pudo decodificar la imagen: {ruta_img}")

        h, w = frame_bgr.shape[:2]
        hash_img = calcular_hash_archivo(ruta_img)

        # Evaluar con Modelo Original
        t0 = time.perf_counter()
        det_orig, dec_orig, probs_orig = pipe_orig.evaluar_fotograma_completo(frame_bgr)
        lat_orig = (time.perf_counter() - t0) * 1000.0

        # Evaluar con Modelo Alineado
        t0 = time.perf_counter()
        det_alin, dec_alin, probs_alin = pipe_alin.evaluar_fotograma_completo(frame_bgr)
        lat_alin = (time.perf_counter() - t0) * 1000.0

        def categorizar(det, dec, probs, pipe):
            if det.estado != EstadoDeteccion.UN_ROSTRO:
                return "fallo_deteccion"
            motivo = dec.motivo_cierre or dec.mensaje
            if motivo and "error" in motivo.lower():
                return "error_tecnico"
            if dec.estado_puerta == "ABIERTA":
                return "falsa_aceptacion"
            if probs is not None:
                pred_idx = int(np.argmax(probs))
                pred_nombre = pipe.mapeo_id_clase.get(pred_idx, "desconocido")
                if pred_nombre == "desconocido":
                    return "clasificacion_desconocido"
                else:
                    return "rechazo_umbral"
            return "clasificacion_desconocido"

        cat_orig = categorizar(det_orig, dec_orig, probs_orig, pipe_orig)
        cat_alin = categorizar(det_alin, dec_alin, probs_alin, pipe_alin)

        # Extraer probabilidades
        def extraer_info_probs(probs, pipe):
            if probs is None:
                return None, 0.0, [0.0, 0.0, 0.0, 0.0]
            pred_idx = int(np.argmax(probs))
            score = float(probs[pred_idx])
            pred_nombre = pipe.mapeo_id_clase.get(pred_idx, "desconocido")
            return pred_nombre, score, [float(x) for x in probs]

        pred_orig, score_orig, pvec_orig = extraer_info_probs(probs_orig, pipe_orig)
        pred_alin, score_alin, pvec_alin = extraer_info_probs(probs_alin, pipe_alin)

        motivo_orig = dec_orig.motivo_cierre if dec_orig.estado_puerta == "CERRADA" else "Acceso concedido por superar umbral"
        motivo_alin = dec_alin.motivo_cierre if dec_alin.estado_puerta == "CERRADA" else "Acceso concedido por superar umbral"

        res_item_orig = {
            "identificador_captura": reg.identificador,
            "participante": reg.persona,
            "pose": reg.pose,
            "distancia": reg.distancia,
            "ruta": reg.ruta,
            "hash_sha256": hash_img,
            "dimensiones": [w, h],
            "dispositivo": "dispositivo no documentado",
            "modelo": pipe_orig.nombre,
            "umbral": pipe_orig.umbral_score,
            "estado_deteccion": det_orig.estado.value,
            "total_rostros": det_orig.total_rostros_detectados,
            "recuadro": det_orig.recuadro,
            "clase_argmax": pred_orig,
            "score_argmax": score_orig,
            "probabilidades": pvec_orig,
            "decision_puerta": dec_orig.estado_puerta,
            "categoria_resultado": cat_orig,
            "motivo_exacto": motivo_orig,
            "latencia_ms": lat_orig,
            "rostro_recortado": det_orig.rostro_recortado,
        }

        res_item_alin = {
            "identificador_captura": reg.identificador,
            "participante": reg.persona,
            "pose": reg.pose,
            "distancia": reg.distancia,
            "ruta": reg.ruta,
            "hash_sha256": hash_img,
            "dimensiones": [w, h],
            "dispositivo": "dispositivo no documentado",
            "modelo": pipe_alin.nombre,
            "umbral": pipe_alin.umbral_score,
            "estado_deteccion": det_alin.estado.value,
            "total_rostros": det_alin.total_rostros_detectados,
            "recuadro": det_alin.recuadro,
            "clase_argmax": pred_alin,
            "score_argmax": score_alin,
            "probabilidades": pvec_alin,
            "decision_puerta": dec_alin.estado_puerta,
            "categoria_resultado": cat_alin,
            "motivo_exacto": motivo_alin,
            "latencia_ms": lat_alin,
            "rostro_recortado": det_alin.recorte_alineado if det_alin.recorte_alineado is not None else det_alin.rostro_recortado,
        }

        resultados_orig.append(res_item_orig)
        resultados_alin.append(res_item_alin)

        # Comparación pareada
        ambos_rechazan = (dec_orig.estado_puerta == "CERRADA") and (dec_alin.estado_puerta == "CERRADA")
        solo_orig_acepta = (dec_orig.estado_puerta == "ABIERTA") and (dec_alin.estado_puerta == "CERRADA")
        solo_alin_acepta = (dec_orig.estado_puerta == "CERRADA") and (dec_alin.estado_puerta == "ABIERTA")
        ambos_aceptan = (dec_orig.estado_puerta == "ABIERTA") and (dec_alin.estado_puerta == "ABIERTA")

        pareados.append({
            "captura": reg.identificador,
            "sujeto": reg.persona,
            "pose": reg.pose,
            "distancia": reg.distancia,
            "orig_decision": dec_orig.estado_puerta,
            "orig_cat": cat_orig,
            "orig_pred": pred_orig,
            "orig_score": score_orig,
            "alin_decision": dec_alin.estado_puerta,
            "alin_cat": cat_alin,
            "alin_pred": pred_alin,
            "alin_score": score_alin,
            "ambos_rechazan": ambos_rechazan,
            "solo_orig_acepta": solo_orig_acepta,
            "solo_alin_acepta": solo_alin_acepta,
            "ambos_aceptan": ambos_aceptan,
        })

        # Filas para CSV individual
        for r_item, p_pipe in ((res_item_orig, pipe_orig), (res_item_alin, pipe_alin)):
            p_dict = {
                f"prob_{p_pipe.mapeo_id_clase[k]}": round(r_item["probabilidades"][k], 4)
                for k in range(4)
            }
            fila = {
                "identificador_captura": r_item["identificador_captura"],
                "participante": r_item["participante"],
                "pose": r_item["pose"],
                "distancia": r_item["distancia"],
                "modelo": r_item["modelo"],
                "umbral": r_item["umbral"],
                "estado_deteccion": r_item["estado_deteccion"],
                "total_rostros": r_item["total_rostros"],
                "clase_argmax": r_item["clase_argmax"],
                "score_argmax": round(r_item["score_argmax"], 4),
                **p_dict,
                "decision_puerta": r_item["decision_puerta"],
                "categoria_resultado": r_item["categoria_resultado"],
                "motivo_exacto": r_item["motivo_exacto"],
                "latencia_ms": round(r_item["latencia_ms"], 2),
                "dispositivo": r_item["dispositivo"],
                "hash_sha256": r_item["hash_sha256"],
            }
            filas_csv_individuales.append(fila)

        if i % 15 == 0 or i == len(registros_desconocidos):
            print(f"  Evaluadas {i}/{len(registros_desconocidos)} imágenes...")

    # Guardar CSV de decisiones individuales
    ruta_csv_ind = dir_salida / "decisiones_individuales.csv"
    with open(ruta_csv_ind, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_csv_individuales[0].keys()))
        writer.writeheader()
        writer.writerows(filas_csv_individuales)

    # 4. Calcular métricas completas
    def calcular_metricas_modelo(lista_res, pipe):
        total = len(lista_res)
        sujetos = sorted(list({r["participante"] for r in lista_res}))
        
        falsas_aceptaciones = sum(1 for r in lista_res if r["decision_puerta"] == "ABIERTA")
        llegaron_clasificador = sum(1 for r in lista_res if r["estado_deteccion"] == "UN_ROSTRO")
        
        far_e2e_num = falsas_aceptaciones
        far_e2e_den = total
        far_e2e_pct = (far_e2e_num / far_e2e_den * 100.0) if far_e2e_den > 0 else 0.0

        far_cls_num = falsas_aceptaciones
        far_cls_den = llegaron_clasificador
        far_cls_pct = (far_cls_num / far_cls_den * 100.0) if far_cls_den > 0 else 0.0

        clasif_desconocido = sum(1 for r in lista_res if r["categoria_resultado"] == "clasificacion_desconocido")
        rechazo_umbral = sum(1 for r in lista_res if r["categoria_resultado"] == "rechazo_umbral")
        fallos_deteccion = sum(1 for r in lista_res if r["categoria_resultado"] == "fallo_deteccion")
        errores_tecnicos = sum(1 for r in lista_res if r["categoria_resultado"] == "error_tecnico")

        # Sujetos con al menos una falsa aceptación
        sujetos_con_fa = sorted(list({
            r["participante"] for r in lista_res if r["decision_puerta"] == "ABIERTA"
        }))

        # Desglose por participante
        desglose_participante = {}
        for s in sujetos:
            sub = [r for r in lista_res if r["participante"] == s]
            sub_fa = sum(1 for r in sub if r["decision_puerta"] == "ABIERTA")
            sub_descon = sum(1 for r in sub if r["categoria_resultado"] == "clasificacion_desconocido")
            sub_rech = sum(1 for r in sub if r["categoria_resultado"] == "rechazo_umbral")
            sub_det_fallo = sum(1 for r in sub if r["categoria_resultado"] == "fallo_deteccion")
            desglose_participante[s] = {
                "intentos": len(sub),
                "falsas_aceptaciones": sub_fa,
                "far_pct": (sub_fa / len(sub) * 100.0) if sub else 0.0,
                "clasif_desconocido": sub_descon,
                "rechazo_umbral": sub_rech,
                "fallo_deteccion": sub_det_fallo,
            }

        # Desglose por pose
        poses = ["frontal", "izquierda", "derecha", "arriba", "abajo"]
        desglose_pose = {}
        for p in poses:
            sub = [r for r in lista_res if r["pose"] == p]
            sub_fa = sum(1 for r in sub if r["decision_puerta"] == "ABIERTA")
            desglose_pose[p] = {
                "intentos": len(sub),
                "falsas_aceptaciones": sub_fa,
                "far_pct": (sub_fa / len(sub) * 100.0) if sub else 0.0,
            }

        # Desglose por distancia
        distancias = ["cercana", "media", "lejana"]
        desglose_distancia = {}
        for d in distancias:
            sub = [r for r in lista_res if r["distancia"] == d]
            sub_fa = sum(1 for r in sub if r["decision_puerta"] == "ABIERTA")
            desglose_distancia[d] = {
                "intentos": len(sub),
                "falsas_aceptaciones": sub_fa,
                "far_pct": (sub_fa / len(sub) * 100.0) if sub else 0.0,
            }

        latencias = [r["latencia_ms"] for r in lista_res]
        lat_media = float(np.mean(latencias)) if latencias else 0.0
        lat_p95 = float(np.percentile(latencias, 95)) if latencias else 0.0

        return {
            "modelo": pipe.nombre,
            "configuracion": str(pipe.ruta_config.name),
            "umbral": pipe.umbral_score,
            "intentos_totales": total,
            "participantes_distintos": len(sujetos),
            "llegaron_clasificador": llegaron_clasificador,
            "falsas_aceptaciones": falsas_aceptaciones,
            "far_e2e_numerador": far_e2e_num,
            "far_e2e_denominador": far_e2e_den,
            "far_e2e_pct": round(far_e2e_pct, 2),
            "far_cls_numerador": far_cls_num,
            "far_cls_denominador": far_cls_den,
            "far_cls_pct": round(far_cls_pct, 2),
            "clasificaciones_desconocido": clasif_desconocido,
            "rechazos_por_umbral": rechazo_umbral,
            "fallos_deteccion": fallos_deteccion,
            "errores_tecnicos": errores_tecnicos,
            "participantes_con_falsa_aceptacion_conteo": len(sujetos_con_fa),
            "participantes_con_falsa_aceptacion_lista": sujetos_con_fa,
            "latencia_media_ms": round(lat_media, 2),
            "latencia_p95_ms": round(lat_p95, 2),
            "desglose_participante": desglose_participante,
            "desglose_pose": desglose_pose,
            "desglose_distancia": desglose_distancia,
        }

    met_orig = calcular_metricas_modelo(resultados_orig, pipe_orig)
    met_alin = calcular_metricas_modelo(resultados_alin, pipe_alin)

    # Tabla pareada
    conteo_ambos_rechazan = sum(1 for p in pareados if p["ambos_rechazan"])
    conteo_solo_orig_acepta = sum(1 for p in pareados if p["solo_orig_acepta"])
    conteo_solo_alin_acepta = sum(1 for p in pareados if p["solo_alin_acepta"])
    conteo_ambos_aceptan = sum(1 for p in pareados if p["ambos_aceptan"])

    tabla_pareada = {
        "ambos_modelos_rechazan": conteo_ambos_rechazan,
        "solo_original_acepta": conteo_solo_orig_acepta,
        "solo_alineado_acepta": conteo_solo_alin_acepta,
        "ambos_modelos_aceptan": conteo_ambos_aceptan,
        "total_pareados": len(pareados),
    }

    # Guardar resumen métricas
    resumen_completo = {
        "fecha_evaluacion": datetime.now(timezone.utc).isoformat(),
        "directorio_capturas": str(RAIZ / "data" / "recepcion_externa" / "fotos"),
        "total_imagenes_evaluadas": len(registros_desconocidos),
        "participantes_evaluados": len(sujetos_unicos),
        "aclaracion_independencia": "Se evaluaron 15 capturas por voluntario en diversas poses y distancias. Varias fotos de un mismo sujeto no equivalen a identidades estadísticamente independientes.",
        "modelo_original": met_orig,
        "modelo_alineado": met_alin,
        "tabla_pareada": tabla_pareada,
    }

    with open(dir_salida / "resumen_comparativo.json", "w", encoding="utf-8") as f:
        json.dump(resumen_completo, f, indent=2, ensure_ascii=False)

    # 5. Generar Lámina de Inspección de Errores
    # Identificar casos de interés: falsas aceptaciones o fallos de detección en cualquiera de los dos modelos
    casos_interes = []
    for orig_item, alin_item in zip(resultados_orig, resultados_alin):
        es_error_orig = (orig_item["decision_puerta"] == "ABIERTA") or (orig_item["categoria_resultado"] == "fallo_deteccion")
        es_error_alin = (alin_item["decision_puerta"] == "ABIERTA") or (alin_item["categoria_resultado"] == "fallo_deteccion")
        if es_error_orig or es_error_alin:
            casos_interes.append((orig_item, alin_item))

    print(f"\n Total de incidencias (falsas aceptaciones o fallos de detección): {len(casos_interes)}")

    # Si no hubiera incidencias o hubiera pocas, añadir casos representativos para inspección
    muestras_a_graficar = list(casos_interes)
    if len(muestras_a_graficar) < 6:
        # Añadir muestras limpias representativas de distintos participantes
        for orig_item, alin_item in zip(resultados_orig, resultados_alin):
            if (orig_item, alin_item) not in muestras_a_graficar:
                muestras_a_graficar.append((orig_item, alin_item))
                if len(muestras_a_graficar) >= 6:
                    break

    # Crear lámina gráfica de inspección
    n_casos = min(len(muestras_a_graficar), 8)
    fig, axes = plt.subplots(n_casos, 3, figsize=(15, 3.8 * n_casos))
    if n_casos == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(n_casos):
        o_item, a_item = muestras_a_graficar[idx]
        frame_orig = cv2.imread(str(RAIZ / o_item["ruta"]))
        frame_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)

        # Dibujar recuadro si existe
        caja_o = o_item["recuadro"]
        if caja_o:
            x, y, w_box, h_box = caja_o
            cv2.rectangle(frame_rgb, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)

        # Columna 1: Fotograma original con recuadro
        axes[idx, 0].imshow(frame_rgb)
        axes[idx, 0].set_title(
            f"Captura: {o_item['identificador_captura']}\n({o_item['participante']} | {o_item['pose']} | {o_item['distancia']})",
            fontsize=10, fontweight="bold"
        )
        axes[idx, 0].axis("off")

        # Columna 2: Recorte Original (sin alineación)
        recorte_o = o_item["rostro_recortado"]
        if recorte_o is not None and recorte_o.size > 0:
            rec_o_rgb = cv2.cvtColor(recorte_o, cv2.COLOR_BGR2RGB)
            axes[idx, 1].imshow(rec_o_rgb)
        else:
            axes[idx, 1].text(0.5, 0.5, "SIN RECORTE\n(Fallo Detección)", ha="center", va="center", color="red")
        
        color_dec_o = "red" if o_item["decision_puerta"] == "ABIERTA" else "green"
        axes[idx, 1].set_title(
            f"Original (Umbral {o_item['umbral']})\nDecisión: {o_item['decision_puerta']} ({o_item['categoria_resultado']})\n"
            f"Argmax: {o_item['clase_argmax']} (Score: {o_item['score_argmax']:.3f})",
            fontsize=9, color=color_dec_o
        )
        axes[idx, 1].axis("off")

        # Columna 3: Recorte Alineado (5 puntos)
        recorte_a = a_item["rostro_recortado"]
        if recorte_a is not None and recorte_a.size > 0:
            rec_a_rgb = cv2.cvtColor(recorte_a, cv2.COLOR_BGR2RGB) if len(recorte_a.shape) == 3 else recorte_a
            axes[idx, 2].imshow(rec_a_rgb, cmap="gray" if len(recorte_a.shape) == 2 else None)
        else:
            axes[idx, 2].text(0.5, 0.5, "SIN RECORTE\n(Fallo Detección)", ha="center", va="center", color="red")

        color_dec_a = "red" if a_item["decision_puerta"] == "ABIERTA" else "green"
        axes[idx, 2].set_title(
            f"Alineado (Umbral {a_item['umbral']})\nDecisión: {a_item['decision_puerta']} ({a_item['categoria_resultado']})\n"
            f"Argmax: {a_item['clase_argmax']} (Score: {a_item['score_argmax']:.3f})",
            fontsize=9, color=color_dec_a
        )
        axes[idx, 2].axis("off")

    plt.tight_layout()
    ruta_lamina = dir_laminas / "inspeccion_errores_desconocidos_reales.png"
    plt.savefig(ruta_lamina, dpi=160, bbox_inches="tight")
    plt.close()
    print(f" Lámina de inspección guardada en: {ruta_lamina}")

    # 6. Hashes y manifiesto evaluado
    hashes_dict = {
        "configuracion_original": calcular_hash_archivo(pipe_orig.ruta_config),
        "modelo_original": calcular_hash_archivo(pipe_orig.ruta_modelo),
        "configuracion_alineado": calcular_hash_archivo(pipe_alin.ruta_config),
        "modelo_alineado": calcular_hash_archivo(pipe_alin.ruta_modelo),
        "detector_yunet": calcular_hash_archivo(pipe_orig.ruta_detector),
        "decisiones_individuales_csv": calcular_hash_archivo(ruta_csv_ind),
        "lamina_inspeccion_png": calcular_hash_archivo(ruta_lamina),
    }
    with open(dir_salida / "hashes_sha256.json", "w", encoding="utf-8") as f:
        json.dump(hashes_dict, f, indent=2)

    # Imprimir resumen en consola
    print("\n" + "=" * 75)
    print(" RESUMEN DE RESULTADOS: DESCONOCIDOS REALES (75 INTENTOS / 5 SUJETOS)")
    print("=" * 75)
    print(f"{'Métrica':<35} | {'CNN Original (0.46)':<17} | {'CNN Alineada (0.84)':<17}")
    print("-" * 75)
    print(f"{'Intentos evaluados':<35} | {met_orig['intentos_totales']:<17} | {met_alin['intentos_totales']:<17}")
    print(f"{'Participantes distintos':<35} | {met_orig['participantes_distintos']:<17} | {met_alin['participantes_distintos']:<17}")
    print(f"{'FAR Extremo a Extremo':<35} | {met_orig['far_e2e_numerador']}/{met_orig['far_e2e_denominador']} ({met_orig['far_e2e_pct']}%) | {met_alin['far_e2e_numerador']}/{met_alin['far_e2e_denominador']} ({met_alin['far_e2e_pct']}%)")
    print(f"{'FAR sobre Clasificador':<35} | {met_orig['far_cls_numerador']}/{met_orig['far_cls_denominador']} ({met_orig['far_cls_pct']}%) | {met_alin['far_cls_numerador']}/{met_alin['far_cls_denominador']} ({met_alin['far_cls_pct']}%)")
    print(f"{'Clasif. explícita Desconocido':<35} | {met_orig['clasificaciones_desconocido']:<17} | {met_alin['clasificaciones_desconocido']:<17}")
    print(f"{'Rechazo por umbral (score)':<35} | {met_orig['rechazos_por_umbral']:<17} | {met_alin['rechazos_por_umbral']:<17}")
    print(f"{'Fallos de detección (YuNet)':<35} | {met_orig['fallos_deteccion']:<17} | {met_alin['fallos_deteccion']:<17}")
    print(f"{'Sujetos con >=1 Falsa Aceptación':<35} | {met_orig['participantes_con_falsa_aceptacion_conteo']}/5 ({', '.join(met_orig['participantes_con_falsa_aceptacion_lista']) or 'Ninguno'}) | {met_alin['participantes_con_falsa_aceptacion_conteo']}/5 ({', '.join(met_alin['participantes_con_falsa_aceptacion_lista']) or 'Ninguno'})")
    print(f"{'Latencia Media / P95 (ms)':<35} | {met_orig['latencia_media_ms']} / {met_orig['latencia_p95_ms']} ms | {met_alin['latencia_media_ms']} / {met_alin['latencia_p95_ms']} ms")
    print("-" * 75)
    print(" TABLA PAREADA DE DECISIONES DE ACCESO:")
    print(f"  - Ambos modelos rechazan (Puerta Cerrada): {tabla_pareada['ambos_modelos_rechazan']} / 75")
    print(f"  - Solo el modelo Original acepta:         {tabla_pareada['solo_original_acepta']} / 75")
    print(f"  - Solo el modelo Alineado acepta:         {tabla_pareada['solo_alineado_acepta']} / 75")
    print(f"  - Ambos modelos aceptan:                  {tabla_pareada['ambos_modelos_aceptan']} / 75")
    print("=" * 75 + "\n")

    return resumen_completo, dir_salida


if __name__ == "__main__":
    ejecutar_evaluacion_comparativa()
