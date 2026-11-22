"""
Módulo de inspección visual paso a paso del pipeline de reconocimiento facial.
Muestra estrictamente los pasos reales ejecutados por la configuración y el modelo activo:
1. Adquisición y detección facial (YuNet + 5 puntos faciales).
2. Preprocesamiento:
   - Recorte seguro o Alineación 2D de 5 puntos a plantilla canónica.
   - Escala de grises y normalización 64x64 (INTER_AREA / INTER_LINEAR).
   - Filtro suavizado gaussiano (sigma=0.5, kernel 5x5) y diferencia de filtrado paso-bajo.
   - Normalización de intensidades float32 [0.0, 1.0].
3. Identificación de rasgos faciales según el modelo activo:
   - Si es CNN: Mapas de activación reales de Bloque 1 (32x32), Bloque 2 (16x16) y Bloque 3 (8x8).
   - Si es Árbol/KNN/SVM/ANN: Cuadrícula 4x4 e histogramas regionales de 256 dimensiones.
4. Clasificación y regla de acceso.
"""

import base64
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from reconocimiento.adquisicion.propio import ManifiestoDataset
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.preprocesamiento.alineador import alinear_rostro
from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion,
)


def _a_png_base64(img: Optional[np.ndarray]) -> str:
    """Codifica un array BGR o escala de grises a PNG base64 sin pérdidas."""
    if img is None or img.size == 0:
        return ""
    if img.dtype != np.uint8:
        arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = img
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _dibujar_deteccion_y_landmarks(frame_bgr: np.ndarray, det: ResultadoDeteccion) -> np.ndarray:
    """Dibuja el recuadro de detección y los 5 puntos faciales de YuNet sobre el fotograma."""
    vis = frame_bgr.copy()
    if det.recuadro is not None:
        x, y, w, h = det.recuadro
        cv2.rectangle(vis, (x, y), (x + w, y + h), (16, 185, 129), 3)

    if det.puntos_faciales is not None and len(det.puntos_faciales) >= 5:
        colores = [
            (255, 100, 100),  # OD
            (255, 200, 50),   # OI
            (50, 220, 255),   # Nariz
            (180, 100, 255),  # BD
            (100, 140, 255),  # BI
        ]
        etiquetas = ["OD", "OI", "N", "BD", "BI"]
        radio = max(3, int(min(vis.shape[:2]) * 0.008))
        for idx, (px, py) in enumerate(det.puntos_faciales[:5]):
            cx, cy = int(round(px)), int(round(py))
            col = colores[idx]
            cv2.circle(vis, (cx, cy), radio + 2, (0, 0, 0), -1)
            cv2.circle(vis, (cx, cy), radio, col, -1)
            cv2.putText(
                vis, etiquetas[idx], (cx + radio + 3, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA
            )
    return vis


def _generar_visualizacion_cuadricula_y_histogramas(img_64_float: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera cuadrícula 4x4 y heatmap del vector 16x16=256 histogramas regionales."""
    extractor = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16)
    res_feat = extractor.extraer(img_64_float)
    vec_256 = res_feat.vector_caracteristicas

    u8 = np.clip(img_64_float * 255.0, 0, 255).astype(np.uint8)
    vis_grid = cv2.resize(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR), (192, 192), interpolation=cv2.INTER_NEAREST)
    paso = 48
    for i in range(1, 4):
        cv2.line(vis_grid, (i * paso, 0), (i * paso, 192), (0, 212, 255), 1)
        cv2.line(vis_grid, (0, i * paso), (192, i * paso), (0, 212, 255), 1)
    for r in range(4):
        for c in range(4):
            idx_celda = r * 4 + c
            cv2.putText(
                vis_grid, f"R{idx_celda+1}", (c * paso + 4, r * paso + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 220, 80), 1, cv2.LINE_AA
            )

    matriz = vec_256.reshape(16, 16)
    max_val = float(np.max(matriz)) if np.max(matriz) > 0 else 1.0
    norm_mat = np.clip((matriz / max_val) * 255.0, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(norm_mat, cv2.COLORMAP_VIRIDIS)
    heatmap_grande = cv2.resize(heatmap_color, (256, 256), interpolation=cv2.INTER_NEAREST)
    for i in range(1, 16):
        cv2.line(heatmap_grande, (0, i * 16), (256, i * 16), (30, 30, 30), 1)

    return vis_grid, heatmap_grande, vec_256


def _generar_mapas_activacion_cnn(
    modelo_cnn: ClasificadorCNN, img_64_float: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae activaciones reales de los 3 bloques convolucionales de la CNN activa:
    - Bloque 1: 16 filtros (32x32)
    - Bloque 2: 32 filtros (16x16) -> Top 16
    - Bloque 3: 64 filtros (8x8)   -> Top 16
    """
    import torch

    net = modelo_cnn.red_wrapper.net
    net.eval()
    tensor_in = torch.tensor(img_64_float, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        x1 = net.b1_pool(net.b1_relu(net.b1_bn(net.b1_conv(tensor_in))))  # [1, 16, 32, 32]
        x2 = net.b2_pool(net.b2_relu(net.b2_bn(net.b2_conv(x1))))         # [1, 32, 16, 16]
        x3 = net.b3_pool(net.b3_relu(net.b3_bn(net.b3_conv(x2))))         # [1, 64, 8, 8]

    def construir_mosaico(fmaps_tensor, n_mostrar: int = 16, celda_px: int = 48) -> np.ndarray:
        arr = fmaps_tensor[0].cpu().numpy()
        energias = np.mean(arr, axis=(1, 2))
        indices = np.argsort(-energias)[:n_mostrar]

        mosaico = np.zeros((4 * celda_px, 4 * celda_px, 3), dtype=np.uint8)
        for pos, idx_f in enumerate(indices):
            fmap = arr[idx_f]
            f_min, f_max = float(np.min(fmap)), float(np.max(fmap))
            if f_max > f_min:
                f_u8 = np.clip(((fmap - f_min) / (f_max - f_min)) * 255.0, 0, 255).astype(np.uint8)
            else:
                f_u8 = np.zeros_like(fmap, dtype=np.uint8)
            f_col = cv2.applyColorMap(f_u8, cv2.COLORMAP_INFERNO)
            f_res = cv2.resize(f_col, (celda_px - 2, celda_px - 2), interpolation=cv2.INTER_NEAREST)
            r, c = divmod(pos, 4)
            mosaico[r * celda_px + 1 : (r + 1) * celda_px - 1, c * celda_px + 1 : (c + 1) * celda_px - 1] = f_res
        return mosaico

    return construir_mosaico(x1, 16, 48), construir_mosaico(x2, 16, 48), construir_mosaico(x3, 16, 48)


def inspeccionar_fotograma_pipeline(
    frame_bgr: np.ndarray,
    gestor: Any,
    origen_etiqueta: str = "camara",
) -> Dict[str, Any]:
    """
    Ejecuta paso a paso las 4 fases reales sobre `frame_bgr` usando exclusivamente
    los pasos y el extractor de rasgos del modelo activo en `gestor`.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {"status": "error", "mensaje": "Fotograma vacío o nulo."}

    pipe_cand = getattr(gestor, "pipeline_candidato", None)
    detector = pipe_cand.detector if pipe_cand is not None else gestor.detector
    usar_alineacion = getattr(pipe_cand, "usar_alineacion", False) if pipe_cand is not None else False
    sigma_gauss = float(getattr(pipe_cand, "sigma_gaussiano", 0.5)) if pipe_cand is not None else 0.5

    # Fase 1: Detección facial YuNet
    res_det = detector.detectar(frame_bgr, es_pre_recortada=False)
    vis_deteccion = _dibujar_deteccion_y_landmarks(frame_bgr, res_det)

    # Identificar modelo activo y tipo de rasgos que realmente utiliza
    modelo_cnn_activo: Optional[ClasificadorCNN] = None
    modelo_habilitado = False
    nombre_modelo = "Sin modelo activo"

    if pipe_cand is not None:
        modelo_habilitado = True
        nombre_modelo = pipe_cand.nombre
        if isinstance(pipe_cand.clasificador, ClasificadorCNN):
            modelo_cnn_activo = pipe_cand.clasificador
    elif getattr(gestor, "modelo", None) is not None and getattr(gestor, "modelo_habilitado", False):
        modelo_habilitado = True
        nombre_modelo = getattr(gestor, "nombre_modelo_activo", "Modelo Activo")
        if isinstance(gestor.modelo, ClasificadorCNN):
            modelo_cnn_activo = gestor.modelo

    if not modelo_habilitado:
        tipo_rasgos = "ninguno"
    elif modelo_cnn_activo is not None:
        tipo_rasgos = "cnn"
    else:
        tipo_rasgos = "histogramas_4x4x16"

    respuesta: Dict[str, Any] = {
        "status": "ok",
        "origen": origen_etiqueta,
        "fase_1_deteccion": {
            "estado": res_det.estado.value,
            "total_rostros": res_det.total_rostros_detectados,
            "recuadro": list(res_det.recuadro) if res_det.recuadro else None,
            "angulo_orientacion": getattr(res_det, "angulo_orientacion", 0),
            "tiene_puntos_faciales": res_det.puntos_faciales is not None,
            "imagen_deteccion_b64": _a_png_base64(vis_deteccion),
        },
        "fase_2_preprocesamiento": {
            "usa_alineacion": usar_alineacion,
            "metodo_extraccion": "Alineación 2D Similitud (5 Puntos)" if usar_alineacion else "Recorte Seguro (Margen +10%)",
            "angulo_alineacion_deg": None,
            "escala_alineacion": None,
            "sigma_gaussiano": sigma_gauss,
            "kernel_gaussiano": f"{int(2 * np.ceil(3 * sigma_gauss) + 1)}x{int(2 * np.ceil(3 * sigma_gauss) + 1)}",
            "recorte_usado_b64": "",
            "gris_64_b64": "",
            "gaussiano_64_b64": "",
            "diff_gaussiano_b64": "",
            "estadisticas_tensor": None,
        },
        "fase_3_rasgos": {
            "tipo_rasgos": tipo_rasgos,
            "modelo_nombre": nombre_modelo,
            "cuadricula_4x4_b64": "",
            "heatmap_histogramas_256_b64": "",
            "norma_l2_vector_256": None,
            "cnn_bloque1_b64": "",
            "cnn_bloque2_b64": "",
            "cnn_bloque3_b64": "",
        },
        "fase_4_clasificacion": {
            "modelo_activo": nombre_modelo if modelo_habilitado else None,
            "umbral": None,
            "decision_puerta": "CERRADA",
            "clase_argmax": None,
            "score_argmax": None,
            "probabilidades": None,
            "motivo": "Sin rostro válido detectado" if res_det.estado != EstadoDeteccion.UN_ROSTRO else "Sin modelo activo",
        },
    }

    if res_det.estado != EstadoDeteccion.UN_ROSTRO or res_det.rostro_recortado is None:
        return respuesta

    # Fase 2: Preprocesamiento paso a paso real
    recorte_raw = res_det.rostro_recortado

    if usar_alineacion:
        if res_det.recorte_alineado is not None:
            rec_elegido = res_det.recorte_alineado
            # Obtener ángulo y escala de alineación para mostrar métricas exactas
            if res_det.puntos_faciales is not None:
                m_rem = getattr(pipe_cand, "metodo_remuestreo", "estandar") if pipe_cand is not None else "estandar"
                _, meta_alin = alinear_rostro(frame_bgr, puntos_faciales=res_det.puntos_faciales, recuadro=res_det.recuadro, resolucion=(64, 64), metodo_remuestreo=m_rem)
                respuesta["fase_2_preprocesamiento"]["angulo_alineacion_deg"] = round(float(meta_alin.angulo_grados), 2)
                respuesta["fase_2_preprocesamiento"]["escala_alineacion"] = round(float(meta_alin.escala), 3)
        elif res_det.puntos_faciales is not None:
            m_rem = getattr(pipe_cand, "metodo_remuestreo", "estandar") if pipe_cand is not None else "estandar"
            rec_elegido, meta_alin = alinear_rostro(frame_bgr, puntos_faciales=res_det.puntos_faciales, recuadro=res_det.recuadro, resolucion=(64, 64), metodo_remuestreo=m_rem)
            respuesta["fase_2_preprocesamiento"]["angulo_alineacion_deg"] = round(float(meta_alin.angulo_grados), 2)
            respuesta["fase_2_preprocesamiento"]["escala_alineacion"] = round(float(meta_alin.escala), 3)
        else:
            rec_elegido = recorte_raw
    else:
        rec_elegido = recorte_raw

    respuesta["fase_2_preprocesamiento"]["recorte_usado_b64"] = _a_png_base64(
        cv2.resize(rec_elegido, (160, 160), interpolation=cv2.INTER_NEAREST if usar_alineacion else cv2.INTER_LINEAR)
    )

    # Paso 2.2: Escala de grises + normalización 64x64 + float32 [0,1] (ANTES de Gaussiano)
    gris = a_escala_de_grises(rec_elegido) if len(rec_elegido.shape) == 3 else rec_elegido
    norm_res = normalizar_resolucion(gris, (64, 64), usar_antialiasing=True)
    norm_int = normalizar_intensidades(norm_res, "float32_0_1")

    u8_gris = np.clip(norm_int * 255.0, 0, 255).astype(np.uint8)
    respuesta["fase_2_preprocesamiento"]["gris_64_b64"] = _a_png_base64(
        cv2.resize(u8_gris, (160, 160), interpolation=cv2.INTER_NEAREST)
    )

    # Paso 2.3: Filtro Suavizado Gaussiano (sigma=0.5, k=5x5)
    img_64_preproc = suavizado_gaussiano(norm_int, sigma=sigma_gauss)
    u8_gauss = np.clip(img_64_preproc * 255.0, 0, 255).astype(np.uint8)
    respuesta["fase_2_preprocesamiento"]["gaussiano_64_b64"] = _a_png_base64(
        cv2.resize(u8_gauss, (160, 160), interpolation=cv2.INTER_NEAREST)
    )

    # Residual del filtro paso-bajo (|antes - después| amplificado x8 para visibilidad)
    diff = np.abs(norm_int - img_64_preproc)
    diff_u8 = np.clip(diff * 8.0 * 255.0, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_MAGMA)
    respuesta["fase_2_preprocesamiento"]["diff_gaussiano_b64"] = _a_png_base64(
        cv2.resize(diff_color, (160, 160), interpolation=cv2.INTER_NEAREST)
    )

    respuesta["fase_2_preprocesamiento"]["estadisticas_tensor"] = {
        "forma": "64 × 64 (float32)",
        "min": round(float(np.min(img_64_preproc)), 4),
        "max": round(float(np.max(img_64_preproc)), 4),
        "media": round(float(np.mean(img_64_preproc)), 4),
        "desv_std": round(float(np.std(img_64_preproc)), 4),
    }

    # Fase 3: Mostrar ÚNICAMENTE el extractor de rasgos que usa el modelo activo
    if tipo_rasgos == "cnn" and modelo_cnn_activo is not None:
        try:
            b1_vis, b2_vis, b3_vis = _generar_mapas_activacion_cnn(modelo_cnn_activo, img_64_preproc)
            respuesta["fase_3_rasgos"]["cnn_bloque1_b64"] = _a_png_base64(b1_vis)
            respuesta["fase_3_rasgos"]["cnn_bloque2_b64"] = _a_png_base64(b2_vis)
            respuesta["fase_3_rasgos"]["cnn_bloque3_b64"] = _a_png_base64(b3_vis)
        except Exception as e_cnn:
            print(f"[INSPECTOR] Error extrayendo activaciones CNN: {e_cnn}")
    elif tipo_rasgos == "histogramas_4x4x16":
        vis_grid, heatmap_256, vec_256 = _generar_visualizacion_cuadricula_y_histogramas(img_64_preproc)
        respuesta["fase_3_rasgos"]["cuadricula_4x4_b64"] = _a_png_base64(vis_grid)
        respuesta["fase_3_rasgos"]["heatmap_histogramas_256_b64"] = _a_png_base64(heatmap_256)
        respuesta["fase_3_rasgos"]["norma_l2_vector_256"] = round(float(np.linalg.norm(vec_256)), 4)

    # Fase 4: Clasificación y decisión
    if pipe_cand is not None:
        dec, probs = pipe_cand.evaluar_recorte(rec_elegido)
        pred_idx = int(np.argmax(probs))
        clase_argmax = pipe_cand.mapeo_id_clase.get(pred_idx, f"clase_{pred_idx}")
        probs_dict = {
            pipe_cand.mapeo_id_clase.get(i, str(i)): round(float(probs[i]) * 100.0, 2)
            for i in range(len(probs))
        }
        respuesta["fase_4_clasificacion"] = {
            "modelo_activo": pipe_cand.nombre,
            "umbral": round(pipe_cand.umbral_score * 100.0, 1),
            "decision_puerta": dec.estado_puerta,
            "clase_argmax": clase_argmax,
            "score_argmax": round(float(probs[pred_idx]) * 100.0, 2),
            "probabilidades": probs_dict,
            "motivo": dec.motivo_cierre if dec.estado_puerta == "CERRADA" else dec.mensaje,
        }
    elif modelo_habilitado and getattr(gestor, "modelo", None) is not None:
        modelo = gestor.modelo
        if isinstance(modelo, ClasificadorCNN):
            probs = modelo.predecir_probabilidades(np.expand_dims(img_64_preproc, axis=0))[0]
        else:
            extractor = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16)
            vec = extractor.extraer(img_64_preproc).vector_caracteristicas
            probs = modelo.predecir_probabilidades(np.expand_dims(vec, axis=0))[0]
        pred_idx = int(np.argmax(probs))
        clases_map = getattr(gestor, "mapeo_clases", {})
        clase_argmax = clases_map.get(pred_idx, f"persona_{pred_idx}")
        probs_dict = {
            clases_map.get(i, f"clase_{i}"): round(float(probs[i]) * 100.0, 2)
            for i in range(len(probs))
        }
        score_pct = round(float(probs[pred_idx]) * 100.0, 2)
        umbral_pct = round(float(getattr(gestor, "umbral_rechazo", 0.65)) * 100.0, 1)
        puerta = "ABIERTA" if (clase_argmax != "desconocido" and score_pct >= umbral_pct) else "CERRADA"
        respuesta["fase_4_clasificacion"] = {
            "modelo_activo": nombre_modelo,
            "umbral": umbral_pct,
            "decision_puerta": puerta,
            "clase_argmax": clase_argmax,
            "score_argmax": score_pct,
            "probabilidades": probs_dict,
            "motivo": "Acceso concedido" if puerta == "ABIERTA" else "Rechazado por regla de acceso",
        }

    return respuesta


def listar_muestras_dataset_disponibles(raiz: Path, limite_por_grupo: int = 30) -> List[Dict[str, str]]:
    """
    Lista muestras representativas de los manifiestos del proyecto (Autorizados, Portal,
    Universidad desarrollo y LFW calibración) excluyendo estrictamente 'prueba_reservada'.
    """
    muestras: List[Dict[str, str]] = []

    ruta_propio = raiz / "data" / "propio" / "manifiesto_local.csv"
    if not ruta_propio.exists():
        ruta_propio = raiz / "data" / "propio" / "manifiesto.csv"

    if ruta_propio.exists():
        man = ManifiestoDataset(ruta_propio)
        conteos = {"autorizado": 0, "portal": 0}
        for r in man.registros:
            p_img = raiz / r.ruta
            if not p_img.is_file():
                continue
            if r.procedencia == "portal_web_externo":
                if conteos["portal"] < limite_por_grupo:
                    muestras.append({
                        "id": r.identificador,
                        "grupo": "Desconocido Portal",
                        "etiqueta": f"{r.persona} ({r.pose}/{r.distancia})",
                        "ruta": r.ruta,
                    })
                    conteos["portal"] += 1
            elif r.tipo_sujeto == "autorizado" and r.particion in ("entrenamiento", "validacion"):
                if conteos["autorizado"] < limite_por_grupo:
                    muestras.append({
                        "id": r.identificador,
                        "grupo": "Autorizado",
                        "etiqueta": f"{r.persona} | {r.sesion} ({r.pose})",
                        "ruta": r.ruta,
                    })
                    conteos["autorizado"] += 1

    ruta_univ = raiz / "data" / "universidad_desc" / "manifiesto_universidad.csv"
    if ruta_univ.exists():
        with ruta_univ.open(encoding="utf-8", newline="") as f:
            cnt_u = 0
            for fila in csv.DictReader(f):
                if fila.get("particion") in ("entrenamiento", "validacion") and cnt_u < limite_por_grupo:
                    if (raiz / fila["ruta"]).is_file():
                        muestras.append({
                            "id": fila["identificador"],
                            "grupo": "Desconocido Universidad",
                            "etiqueta": f"{fila['identificador']} ({fila['particion']})",
                            "ruta": fila["ruta"],
                        })
                        cnt_u += 1

    ruta_lfw = raiz / "data" / "desconocidos_lfw" / "calibracion" / "manifiesto.csv"
    if ruta_lfw.exists():
        with ruta_lfw.open(encoding="utf-8", newline="") as f:
            cnt_l = 0
            for fila in csv.DictReader(f):
                if cnt_l < 15:
                    if (raiz / fila["ruta_imagen"]).is_file():
                        muestras.append({
                            "id": f"lfw_{fila['identidad_normalizada']}",
                            "grupo": "Desconocido LFW",
                            "etiqueta": fila["identidad_normalizada"],
                            "ruta": fila["ruta_imagen"],
                        })
                        cnt_l += 1

    return muestras
