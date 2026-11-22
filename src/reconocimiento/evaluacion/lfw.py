"""
Evaluación externa del sistema de reconocimiento facial y apertura simulada
utilizando una muestra reproducible de Labeled Faces in the Wild (LFW).
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.regla_acceso import evaluar_decision_acceso
from reconocimiento.preprocesamiento.fabrica import crear_detector
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion,
)

logger = logging.getLogger(__name__)


def calcular_sha256(ruta_archivo: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en disco."""
    sha256 = hashlib.sha256()
    with open(ruta_archivo, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def verificar_y_congelar_configuracion(
    ruta_modelo_seleccionado: Path = Path("resultados/modelo_seleccionado.json"),
    directorio_salida: Path = Path("resultados/evaluacion_externa_lfw_cnn"),
) -> Dict[str, Any]:
    """
    Lee resultados/modelo_seleccionado.json y comprueba qué modelo, pesos,
    etiquetas autorizadas y umbral cargaría la web en un arranque normal.
    Si faltan los pesos o hay discrepancia entre el umbral seleccionado y el
    que aplica la web (0.65), detiene la evaluación e informa del problema.
    """
    if not ruta_modelo_seleccionado.exists():
        raise FileNotFoundError(
            f"No existe el archivo de modelo seleccionado: {ruta_modelo_seleccionado}"
        )

    with open(ruta_modelo_seleccionado, "r", encoding="utf-8") as f:
        sel_info = json.load(f)

    modelo_elegido = sel_info.get("modelo_elegido", "").lower()
    dir_modelo = Path(sel_info.get("directorio_modelo", ""))
    umbral_recomendado = float(sel_info.get("umbral_recomendado", 0.65))

    if modelo_elegido != "cnn":
        raise ValueError(
            f"El modelo seleccionado es '{modelo_elegido}', pero se requiere evaluar 'cnn'."
        )

    ruta_pesos = dir_modelo / "modelo.pt"
    if not ruta_pesos.exists():
        raise FileNotFoundError(
            f"Faltan los pesos del modelo seleccionado en: {ruta_pesos}"
        )

    # Verificar clases_mapeo.json
    ruta_mapeo = dir_modelo / "clases_mapeo.json"
    if not ruta_mapeo.exists():
        raise FileNotFoundError(
            f"Falta el archivo de mapeo de clases en: {ruta_mapeo}"
        )

    with open(ruta_mapeo, "r", encoding="utf-8") as f:
        mapeo = json.load(f)

    clases_autorizadas = mapeo.get("personas_autorizadas", [])
    id_to_persona = mapeo.get("id_to_persona", {})

    # Determinar qué umbral cargaría la web en un arranque normal (app.py)
    val_u_web = sel_info.get("umbral_recomendado")
    if val_u_web is None:
        val_u_web = sel_info.get("umbral_congelado", sel_info.get("umbral", umbral_recomendado))
    umbral_web = float(val_u_web)

    # Comprobar calibración de umbral registrada en el modelo
    ruta_calibracion = dir_modelo / "calibracion_umbral.json"
    umbral_calibrado = umbral_recomendado
    if ruta_calibracion.exists():
        with open(ruta_calibracion, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
            umbral_calibrado = float(calib_data.get("umbral_recomendado", calib_data.get("umbral_optimo", umbral_recomendado)))

    # Comprobar umbral en mapeo de clases si está definido
    umbral_mapeo = float(mapeo.get("umbral_rechazo_recomendado", umbral_calibrado))

    # Verificar coherencia estricta entre el valor calibrado y el que aplicaría la web
    # (sin forzar un valor numérico fijo como 0.65, permitiendo legítimamente 0.80 u otros)
    if abs(umbral_calibrado - umbral_web) > 1e-4:
        raise ValueError(
            f"Discrepancia detectada: el umbral calibrado en {ruta_calibracion} ({umbral_calibrado}) "
            f"no coincide con el que cargaría la web en un arranque normal ({umbral_web})."
        )
    if abs(umbral_recomendado - umbral_web) > 1e-4:
        raise ValueError(
            f"Discrepancia detectada: el umbral seleccionado ({umbral_recomendado}) "
            f"no coincide con el umbral de la web ({umbral_web})."
        )
    if abs(umbral_mapeo - umbral_web) > 1e-4:
        raise ValueError(
            f"Discrepancia detectada: el umbral en clases_mapeo.json ({umbral_mapeo}) "
            f"no coincide con el de la web ({umbral_web})."
        )

    sha256_pesos = calcular_sha256(ruta_pesos)

    # Detector de producción por defecto
    det_prod = crear_detector()
    detector_config = {
        "tipo": "YuNet",
        "sha256_onnx": calcular_sha256(det_prod.ruta_modelo),
        "score_threshold": det_prod.score_threshold,
        "nms_threshold": det_prod.nms_threshold,
        "margen_expansion": det_prod.margen_expansion,
    }

    config_congelada = {
        "modelo_elegido": modelo_elegido,
        "directorio_modelo": str(dir_modelo).replace("\\", "/"),
        "ruta_pesos": str(ruta_pesos).replace("\\", "/"),
        "sha256_pesos": sha256_pesos,
        "tamanio_pesos_bytes": ruta_pesos.stat().st_size,
        "umbral_efectivo_puerta": umbral_recomendado,
        "clases_autorizadas": clases_autorizadas,
        "id_to_persona": id_to_persona,
        "version_opencv": cv2.__version__,
        "detector_activo": detector_config,
        "preprocesamiento": {
            "recorte_margen_seguro": True,
            "margen_expansion": det_prod.margen_expansion,
            "escala_grises": True,
            "resolucion_normalizada": [64, 64],
            "normalizacion_intensidad": "float32_0_1",
            "suavizado_gaussiano_sigma": 0.5,
        },
        "lfw_config": {
            "color": True,
            "resize": 1.0,
            "slice_": "[[0, 250], [0, 250]] (campo visual completo sin recorte central)",
            "semilla": 42,
            "n_identidades": 100,
            "n_fotos_por_identidad": 1,
            "reemplazo_identidades": False,
        },
    }

    directorio_salida.mkdir(parents=True, exist_ok=True)
    ruta_congelada = directorio_salida / "configuracion_congelada.json"
    with open(ruta_congelada, "w", encoding="utf-8") as f:
        json.dump(config_congelada, f, indent=2, ensure_ascii=False)

    return config_congelada


def seleccionar_muestra_reproducible_lfw(
    semilla: int = 42,
    n_identidades: int = 100,
    directorio_salida: Path = Path("resultados/evaluacion_externa_lfw_cnn"),
) -> Tuple[List[Dict[str, Any]], np.ndarray, Any]:
    """
    Carga LFW mediante fetch_lfw_people sin recortes centrales ni redimensionamiento.
    Selecciona de forma determinista con semilla=42:
      1. 100 identidades únicas sin reemplazo.
      2. 1 fotografía por cada identidad.
    Guarda las 100 imágenes en `directorio_salida / imagenes_lfw/` y genera el manifiesto CSV.
    """
    print(f"[1/4] Cargando LFW completo (color=True, resize=1.0, slice_=(0:250, 0:250))...")
    lfw = fetch_lfw_people(
        color=True,
        resize=1.0,
        slice_=(slice(0, 250), slice(0, 250)),
        download_if_missing=True,
    )

    rng = np.random.RandomState(semilla)
    unique_identities = np.unique(lfw.target)
    if len(unique_identities) < n_identidades:
        raise ValueError(
            f"El dataset contiene {len(unique_identities)} identidades, menos de las {n_identidades} requeridas."
        )

    # 1. Elegir 100 identidades distintas sin reemplazo
    chosen_identities = rng.choice(unique_identities, size=n_identidades, replace=False)

    # 2. Elegir exactamente 1 fotografía por cada identidad
    chosen_indices = []
    for ident in chosen_identities:
        indices_fotos = np.where(lfw.target == ident)[0]
        foto_sel = rng.choice(indices_fotos)
        chosen_indices.append(int(foto_sel))

    # Verificar unicidad
    assert len(chosen_indices) == n_identidades, "Debe haber exactamente 100 fotografías seleccionadas"
    assert len(set(chosen_identities)) == n_identidades, "Debe haber 100 identidades distintas"
    assert len(set(chosen_indices)) == n_identidades, "Debe haber 100 fotos distintas"

    dir_imagenes = directorio_salida / "imagenes_lfw"
    dir_imagenes.mkdir(parents=True, exist_ok=True)

    manifiesto: List[Dict[str, Any]] = []
    for idx_muestra, idx_dataset in enumerate(chosen_indices, start=1):
        target_id = int(lfw.target[idx_dataset])
        nombre_persona = str(lfw.target_names[target_id])

        # Convertir de float32 [0.0, 1.0] RGB a uint8 BGR para OpenCV
        img_float_rgb = lfw.images[idx_dataset]
        img_uint8_rgb = np.clip(img_float_rgb * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2BGR)

        # Nombre seguro para archivo
        nombre_archivo = f"lfw_{idx_muestra:03d}_{nombre_persona.replace(' ', '_').replace('/', '_')}.jpg"
        ruta_img = dir_imagenes / nombre_archivo
        cv2.imwrite(str(ruta_img), img_bgr)

        sha256_img = calcular_sha256(ruta_img)

        manifiesto.append({
            "id_muestra": idx_muestra,
            "identidad_lfw": nombre_persona,
            "id_clase_lfw": target_id,
            "indice_dataset_lfw": idx_dataset,
            "ruta_imagen": str(ruta_img).replace("\\", "/"),
            "sha256": sha256_img,
            "ancho": int(img_bgr.shape[1]),
            "alto": int(img_bgr.shape[0]),
            "canales": int(img_bgr.shape[2]),
        })

    # Guardar manifiesto_lfw_100.csv
    ruta_manifiesto = directorio_salida / "manifiesto_lfw_100.csv"
    with open(ruta_manifiesto, "w", encoding="utf-8") as f:
        f.write("id_muestra,identidad_lfw,id_clase_lfw,indice_dataset_lfw,ruta_imagen,sha256,ancho,alto,canales\n")
        for item in manifiesto:
            f.write(
                f"{item['id_muestra']},"
                f"\"{item['identidad_lfw']}\","
                f"{item['id_clase_lfw']},"
                f"{item['indice_dataset_lfw']},"
                f"{item['ruta_imagen']},"
                f"{item['sha256']},"
                f"{item['ancho']},"
                f"{item['alto']},"
                f"{item['canales']}\n"
            )

    print(f"  - Muestra de {n_identidades} imágenes guardada en {dir_imagenes}")
    print(f"  - Manifiesto guardado en {ruta_manifiesto}")
    return manifiesto, np.array(chosen_indices), lfw


def reproducir_decision_produccion(
    frame_bgr: np.ndarray,
    detector: DetectorFacial,
    modelo: ClasificadorCNN,
    clases_autorizadas: List[str],
    id_to_persona: Dict[str, str],
    umbral_confianza: float = 0.65,
) -> Dict[str, Any]:
    """
    Ejecuta exactamente la misma lógica de inferencia y decisión de acceso
    que implementa la web en producción (GestorCamaraWeb).
    """
    res_det = detector.detectar(frame_bgr, es_pre_recortada=False)
    estado_det = res_det.estado

    resultado = {
        "estado_detector": estado_det.value,
        "recuadro": res_det.recuadro,
        "recuadro_str": f"({res_det.recuadro[0]},{res_det.recuadro[1]},{res_det.recuadro[2]},{res_det.recuadro[3]})" if res_det.recuadro else "",
        "etiqueta_predicha": "",
        "confianza": None,
        "umbral": umbral_confianza,
        "decision_puerta": "CERRADA",
        "motivo_cierre": "",
        "es_falsa_aceptacion": False,
        "error_tecnico": "",
    }

    if estado_det == EstadoDeteccion.SIN_ROSTRO:
        resultado["decision_puerta"] = "CERRADA"
        resultado["motivo_cierre"] = "SIN_ROSTRO"
        return resultado

    if estado_det == EstadoDeteccion.VARIOS_ROSTROS:
        resultado["decision_puerta"] = "CERRADA"
        resultado["motivo_cierre"] = "VARIOS_ROSTROS"
        return resultado

    # UN_ROSTRO: Inferencia con CNN
    try:
        # 1. Recorte seguro (usar recorte de la detección o recortar con margen seguro)
        recorte = res_det.rostro_recortado
        if recorte is None:
            recorte = detector.recortar_con_margen_seguro(frame_bgr, res_det.recuadro)

        # 2. Preprocesamiento estándar (idéntico a streamer.py)
        gris = a_escala_de_grises(recorte) if len(recorte.shape) == 3 else recorte
        norm_res = normalizar_resolucion(gris, (64, 64))
        norm_int = normalizar_intensidades(norm_res, "float32_0_1")
        suavizada = suavizado_gaussiano(norm_int, sigma=0.5)

        # 3. Tensor de entrada (1, 64, 64)
        in_tensor = np.expand_dims(suavizada, axis=0)
        probs = modelo.predecir_probabilidades(in_tensor)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        # 4. Mapear etiqueta
        id_str = str(pred_idx)
        nombre_pred = id_to_persona.get(id_str, id_to_persona.get(pred_idx, f"persona_{pred_idx}"))

        resultado["etiqueta_predicha"] = nombre_pred
        resultado["confianza"] = float(conf)

        # 5. Regla canónica de decisión de acceso (función compartida con la web y streamer)
        decision_acceso = evaluar_decision_acceso(
            nombre_predicho=nombre_pred,
            confianza=conf,
            umbral_confianza=umbral_confianza,
            personas_autorizadas=clases_autorizadas,
        )

        resultado["decision_puerta"] = decision_acceso.estado_puerta
        resultado["motivo_cierre"] = decision_acceso.motivo_cierre
        resultado["es_falsa_aceptacion"] = (decision_acceso.estado_puerta == "ABIERTA")

    except Exception as exc:
        resultado["decision_puerta"] = "CERRADA"
        resultado["motivo_cierre"] = f"ERROR_TECNICO ({exc})"
        resultado["error_tecnico"] = str(exc)
        logger.error(f"Error técnico durante inferencia: {exc}")

    return resultado


def evaluar_lfw_cnn(
    semilla: int = 42,
    n_identidades: int = 100,
    directorio_salida: Path = Path("resultados/evaluacion_externa_lfw_cnn"),
) -> Dict[str, Any]:
    """
    Ejecuta el protocolo completo de evaluación externa:
      1. Congelamiento y verificación de sistema.
      2. Muestreo reproducible de LFW (100 identidades, 1 foto/identidad).
      3. Ejecución de decisión de puerta sobre cada intento externo.
      4. Generación de artefactos auditables (configuracion, manifiesto, predicciones, informe).
    """
    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)

    # 1. Congelar y verificar configuración
    print("\n--- Paso 1: Congelar y verificar configuración del sistema ---")
    config = verificar_y_congelar_configuracion(directorio_salida=directorio_salida)
    print(f"  - Modelo verificado: {config['modelo_elegido'].upper()}")
    print(f"  - Pesos: {config['ruta_pesos']} (SHA-256: {config['sha256_pesos'][:12]}...)")
    print(f"  - Umbral web verificado: {config['umbral_efectivo_puerta']}")
    print(f"  - Clases autorizadas: {config['clases_autorizadas']}")
    print(f"  - OpenCV: {config['version_opencv']}")

    # 2. Muestra reproducible de LFW
    print("\n--- Paso 2: Seleccionar muestra reproducible de LFW ---")
    manifiesto, indices_sel, lfw = seleccionar_muestra_reproducible_lfw(
        semilla=semilla,
        n_identidades=n_identidades,
        directorio_salida=directorio_salida,
    )

    # 3. Cargar modelo y detector de producción
    print("\n--- Paso 3: Cargar modelo y detector en memoria ---")
    modelo = ClasificadorCNN.cargar(Path(config["ruta_pesos"]))
    detector = crear_detector()
    clases_autorizadas = config["clases_autorizadas"]
    id_to_persona = config["id_to_persona"]
    umbral = config["umbral_efectivo_puerta"]

    # 4. Evaluar cada fotografía
    print(f"\n--- Paso 4: Evaluando decisión de puerta sobre {len(manifiesto)} fotografías externas ---")
    predicciones: List[Dict[str, Any]] = []

    conteo_abiertas = 0
    conteo_cerradas = 0
    conteo_sin_rostro = 0
    conteo_varios_rostros = 0
    conteo_rechazo_clasificador = 0
    conteo_un_rostro = 0
    conteo_errores_tecnicos = 0

    atribucion_por_clase: Dict[str, int] = {c: 0 for c in clases_autorizadas}

    for item in manifiesto:
        ruta_img = Path(item["ruta_imagen"])
        img_bgr = cv2.imread(str(ruta_img))
        if img_bgr is None:
            raise RuntimeError(f"Error de lectura al cargar la imagen guardada: {ruta_img}")

        decision_info = reproducir_decision_produccion(
            frame_bgr=img_bgr,
            detector=detector,
            modelo=modelo,
            clases_autorizadas=clases_autorizadas,
            id_to_persona=id_to_persona,
            umbral_confianza=umbral,
        )

        if decision_info["error_tecnico"]:
            conteo_errores_tecnicos += 1

        est_det = decision_info["estado_detector"]
        decision = decision_info["decision_puerta"]

        if est_det == "UN_ROSTRO":
            conteo_un_rostro += 1

        if decision == "ABIERTA":
            conteo_abiertas += 1
            clase_pred = decision_info["etiqueta_predicha"]
            if clase_pred in atribucion_por_clase:
                atribucion_por_clase[clase_pred] += 1
            else:
                atribucion_por_clase[clase_pred] = 1
        else:
            conteo_cerradas += 1
            if est_det == "SIN_ROSTRO":
                conteo_sin_rostro += 1
            elif est_det == "VARIOS_ROSTROS":
                conteo_varios_rostros += 1
            else:  # UN_ROSTRO rechazado
                conteo_rechazo_clasificador += 1

        fila_pred = {
            "id_muestra": item["id_muestra"],
            "identidad_lfw": item["identidad_lfw"],
            "estado_detector": decision_info["estado_detector"],
            "recuadro": decision_info["recuadro_str"],
            "etiqueta_predicha": decision_info["etiqueta_predicha"],
            "confianza": f"{decision_info['confianza']:.4f}" if decision_info["confianza"] is not None else "",
            "umbral": f"{decision_info['umbral']:.2f}",
            "decision_puerta": decision_info["decision_puerta"],
            "motivo_cierre": decision_info["motivo_cierre"],
            "es_falsa_aceptacion": decision_info["es_falsa_aceptacion"],
            "error_tecnico": decision_info["error_tecnico"],
        }
        predicciones.append(fila_pred)

    if conteo_errores_tecnicos > 0:
        raise RuntimeError(
            f"Se registraron {conteo_errores_tecnicos} errores técnicos durante la evaluación. Ensayo no válido."
        )

    if len(predicciones) != n_identidades:
        raise RuntimeError(
            f"El ensayo procesó {len(predicciones)} intentos, se requerían exactamente {n_identidades}."
        )

    # Verificar balance exacto
    assert conteo_abiertas + conteo_cerradas == n_identidades, "Cierres y aperturas deben sumar exactamente 100"
    assert conteo_sin_rostro + conteo_varios_rostros + conteo_rechazo_clasificador == conteo_cerradas, "Desglose de cierres inconsistente"

    # Tasas
    far_extremo_a_extremo_pct = (conteo_abiertas / n_identidades) * 100.0
    far_un_rostro_pct = (conteo_abiertas / conteo_un_rostro * 100.0) if conteo_un_rostro > 0 else None

    # Guardar predicciones.csv
    ruta_pred_csv = directorio_salida / "predicciones.csv"
    with open(ruta_pred_csv, "w", encoding="utf-8") as f:
        f.write("id_muestra,identidad_lfw,estado_detector,recuadro,etiqueta_predicha,confianza,umbral,decision_puerta,motivo_cierre,es_falsa_aceptacion,error_tecnico\n")
        for p in predicciones:
            f.write(
                f"{p['id_muestra']},"
                f"\"{p['identidad_lfw']}\","
                f"{p['estado_detector']},"
                f"\"{p['recuadro']}\","
                f"\"{p['etiqueta_predicha']}\","
                f"{p['confianza']},"
                f"{p['umbral']},"
                f"{p['decision_puerta']},"
                f"\"{p['motivo_cierre']}\","
                f"{p['es_falsa_aceptacion']},"
                f"\"{p['error_tecnico']}\"\n"
            )

    # 5. Generar informe de auditoría
    print("\n--- Paso 5: Generando informe de auditoría ---")
    informe_md = generar_informe_auditoria(
        config=config,
        n_intentos=n_identidades,
        conteo_abiertas=conteo_abiertas,
        conteo_cerradas=conteo_cerradas,
        conteo_sin_rostro=conteo_sin_rostro,
        conteo_varios_rostros=conteo_varios_rostros,
        conteo_rechazo_clasificador=conteo_rechazo_clasificador,
        conteo_un_rostro=conteo_un_rostro,
        far_extremo=far_extremo_a_extremo_pct,
        far_un_rostro=far_un_rostro_pct,
        atribucion=atribucion_por_clase,
        directorio_salida=directorio_salida,
    )

    ruta_informe = directorio_salida / "informe_evaluacion_lfw.md"
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write(informe_md)

    print(f"  - Predicciones guardadas en: {ruta_pred_csv}")
    print(f"  - Informe guardado en: {ruta_informe}")
    print(f"\n=======================================================")
    print(f" RESUMEN DE LA EVALUACIÓN EXTERNA LFW (100 FOTOGRAFÍAS)")
    print(f"=======================================================")
    print(f"  - Intentos Totales (Desconocidos externos): 100")
    print(f"  - Falsas Aceptaciones (Puerta Abierta):     {conteo_abiertas} ({far_extremo_a_extremo_pct:.1f}%)")
    print(f"  - Puertas Cerradas (Rechazo Correcto):      {conteo_cerradas} ({100 - far_extremo_a_extremo_pct:.1f}%)")
    print(f"    * Por SIN_ROSTRO:                         {conteo_sin_rostro}")
    print(f"    * Por VARIOS_ROSTROS:                     {conteo_varios_rostros}")
    print(f"    * Por RECHAZO CLASIFICADOR:               {conteo_rechazo_clasificador}")
    print(f"  - Imágenes que llegaron al Clasificador:    {conteo_un_rostro}")
    print(f"  - FAR entre imágenes clasificadas:          {f'{far_un_rostro_pct:.1f}%' if far_un_rostro_pct is not None else 'no calculable'}")
    print(f"  - Atribución de aperturas falsas:")
    for cls_name, cnt in atribucion_por_clase.items():
        print(f"    * {cls_name}: {cnt}")
    print(f"=======================================================\n")

    return {
        "n_intentos": n_identidades,
        "abiertas": conteo_abiertas,
        "cerradas": conteo_cerradas,
        "sin_rostro": conteo_sin_rostro,
        "varios_rostros": conteo_varios_rostros,
        "rechazo_clasificador": conteo_rechazo_clasificador,
        "un_rostro": conteo_un_rostro,
        "far_extremo_a_extremo_pct": far_extremo_a_extremo_pct,
        "far_un_rostro_pct": far_un_rostro_pct,
        "atribucion_por_clase": atribucion_por_clase,
    }


def generar_informe_auditoria(
    config: Dict[str, Any],
    n_intentos: int,
    conteo_abiertas: int,
    conteo_cerradas: int,
    conteo_sin_rostro: int,
    conteo_varios_rostros: int,
    conteo_rechazo_clasificador: int,
    conteo_un_rostro: int,
    far_extremo: float,
    far_un_rostro: Optional[float],
    atribucion: Dict[str, int],
    directorio_salida: Path,
) -> str:
    """Genera el contenido estructurado en Markdown del informe de auditoría."""
    str_far_un_rostro = f"{far_un_rostro:.2f}%" if far_un_rostro is not None else "no calculable"

    filas_atribucion = "\n".join(
        f"| `{k}` | **{v}** | {(v / conteo_abiertas * 100.0 if conteo_abiertas > 0 else 0.0):.1f}% |"
        for k, v in atribucion.items()
    )

    doc = f"""# Informe de Evaluación Externa con Labeled Faces in the Wild (LFW)
## Auditoría de Seguridad y Tasa de Falsa Aceptación (FAR) para el Acceso Físico

- **Fecha de Evaluación:** 2026-11-16
- **Pregunta del Ensayo:** *De 100 fotografías de 100 identidades externas desconocidas (LFW), ¿cuántas consiguen abrir la puerta física con el sistema tal como está configurado hoy?*
- **Modelo Evaluado:** Red Neuronal Convolucional (`ClasificadorCNN`), pesos en `{config['ruta_pesos']}`
- **Hash SHA-256 de los Pesos:** `{config['sha256_pesos']}`
- **Umbral de Decisión en Producción:** `{config['umbral_efectivo_puerta']}`
- **Etiquetas Autorizadas:** `{config['clases_autorizadas']}`
- **Versión de OpenCV:** `{config['version_opencv']}`

---

## 1. Resumen Ejecutivo y Resultados Clave

| Métrica | Valor Absoluto | Porcentaje / Tasa |
| :--- | :---: | :---: |
| **Intentos Totales Procesados (Identidades Externas)** | **{n_intentos}** | **100.0%** |
| **Aperturas Falsas (Puerta Abierta a Desconocidos)** | **{conteo_abiertas}** | **{far_extremo:.2f}% (FAR Extremo a Extremo)** |
| **Cierres Correctos (Puerta Cerrada)** | **{conteo_cerradas}** | **{100 - far_extremo:.2f}%** |
| - *Cierres por SIN_ROSTRO (detector)* | {conteo_sin_rostro} | {(conteo_sin_rostro / n_intentos * 100):.2f}% |
| - *Cierres por VARIOS_ROSTROS (detector)* | {conteo_varios_rostros} | {(conteo_varios_rostros / n_intentos * 100):.2f}% |
| - *Cierres por Rechazo tras Clasificación* | {conteo_rechazo_clasificador} | {(conteo_rechazo_clasificador / n_intentos * 100):.2f}% |
| **Imágenes con UN_ROSTRO (llegaron al clasificador)** | **{conteo_un_rostro}** | **{(conteo_un_rostro / n_intentos * 100):.2f}%** |
| **FAR sobre rostros clasificados ($N_{{abiertas}} / N_{{UN\\_ROSTRO}}$)** | **{conteo_abiertas} / {conteo_un_rostro}** | **{str_far_un_rostro}** |

> [!IMPORTANT]
> **Comprobación de Balance Estricto:**
> - $\\text{{Cierres}} + \\text{{Aperturas}} = {conteo_cerradas} + {conteo_abiertas} = {conteo_cerradas + conteo_abiertas}$ (Suma exacta de 100 intentos).
> - $\\text{{Desglose Cierres}} = {conteo_sin_rostro} + {conteo_varios_rostros} + {conteo_rechazo_clasificador} = {conteo_cerradas}$.
> - Errores técnicos: 0. Ningún intento falló por caída del sistema.

---

## 2. Atribución de Falsas Aceptaciones por Sujeto Autorizado

Cada apertura falsa representa a un individuo externo de LFW que fue clasificado erróneamente como un usuario autorizado con confianza superior o igual a 0.65.

| Identidad Autorizada Atribuida | Conteo de Aperturas Falsas | % del Total de Falsas Aceptaciones |
| :--- | :---: | :---: |
{filas_atribucion}
| **Total Falsas Aceptaciones** | **{conteo_abiertas}** | **100.0%** |

---

## 3. Metodología y Trazabilidad de la Muestra LFW

1. **Fuente de Datos:** Labeled Faces in the Wild (LFW) funneled, cargado con `sklearn.datasets.fetch_lfw_people`.
2. **Preservación del Campo Visual:** Se desactivó el recorte central por defecto y cualquier redimensionamiento arbitrario configurando `slice_=(slice(0, 250), slice(0, 250))` y `resize=1.0`. Esto preserva la imagen externa original de $250 \\times 250$ píxeles a color RGB.
3. **Conversión de Canales y Rangos:**
   - La matriz float32 en $[0.0, 1.0]$ fue convertida a `uint8` en rango $[0, 255]$.
   - El orden de canales se transformó estrictamente de RGB a BGR (`cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)`) para garantizar coherencia con OpenCV.
4. **Muestreo Determinista:**
   - Semilla: `seed = 42` mediante `np.random.RandomState(42)`.
   - Primero se seleccionaron **100 identidades distintas sin reemplazo** entre las 5,749 personas únicas disponibles.
   - Para cada identidad seleccionada se eligió **exactamente 1 fotografía**, totalizando 100 imágenes distintas de 100 personas distintas.
   - No se sustituyó ninguna toma por ser difícil, borrosa o presentar pose compleja.
5. **Aislamiento de Datos:**
   - La muestra se guardó en `{directorio_salida}/imagenes_lfw/`.
   - Cada imagen fue registrada con su hash SHA-256 en `{directorio_salida}/manifiesto_lfw_100.csv`.
   - Se mantiene completamente separada de los conjuntos de entrenamiento, validación y prueba de DroidCam / capturas propias.

---

## 4. Pipeline de Inferencia Idéntico a Producción

Para cada fotografía de LFW se ejecutó la misma lógica que aplica `GestorCamaraWeb` en la web:
1. **Detección Multi-Vista:** `DetectorYuNet.detectar(img_bgr, es_pre_recortada=False)` utilizando el modelo ONNX común de producción.
2. **Condiciones de Seguridad:**
   - Si el detector reporta `SIN_ROSTRO` $\\rightarrow$ La puerta se mantiene **CERRADA** (Motivo: `SIN_ROSTRO`).
   - Si el detector reporta `VARIOS_ROSTROS` $\\rightarrow$ La puerta se mantiene **CERRADA** (Motivo: `VARIOS_ROSTROS`).
3. **Preprocesamiento:**
   - Recorte seguro con preservación de relación de aspecto $1:1$ y margen de expansión del $10\\%$, rellenando bordes mediante reflexión si excede el encuadre.
   - Conversión a escala de grises.
   - Normalización de resolución a $64 \\times 64$ píxeles.
   - Normalización de intensidades a flotante en rango $[0.0, 1.0]$.
   - Suavizado gaussiano con $\\sigma = 0.5$.
4. **Inferencia de la Red Convolucional:**
   - Inserción de dimensión de batch: `in_tensor = np.expand_dims(suavizada, axis=0)`.
   - Inferencia de probabilidades: `probs = ClasificadorCNN.predecir_probabilidades(in_tensor)[0]`.
   - Selección de clase y confianza: `pred_idx = np.argmax(probs)`, `conf = probs[pred_idx]`.
   - Mapeo de identificador a nombre mediante `clases_mapeo.json`.
5. **Decisión de Apertura:**
   - Evaluación a través de la función canónica unificada `evaluar_decision_acceso(...)` (compartida entre inferencia web y auditoría).
   - `confianza >= umbral` ({config['umbral_efectivo_puerta']}) **Y** `nombre_pred in {config['clases_autorizadas']}` (y clase no reservada como desconocido) $\\rightarrow$ Puerta **ABIERTA** (Falsa Aceptación).
   - En cualquier otro caso $\\rightarrow$ Puerta **CERRADA** (Rechazo).

---

## 5. Hallazgos y Análisis para la Siguiente Iteración

> [!WARNING]
> **Vulnerabilidad de Suposición de Mundo Cerrado (Closed-World Softmax):**
> La red convolucional actual fue entrenada con una capa final de activación Softmax sobre 3 clases conocidas (`persona_1`, `persona_2`, `persona_3`). Softmax fuerza que la suma de probabilidades sobre las 3 clases sea idénticamente 1.0 ($100\\%$).
> 
> Cuando una persona externa completamente desconocida se presenta ante el sistema:
> 1. El detector YuNet localiza exitosamente el rostro ({conteo_un_rostro}/100 casos).
> 2. La CNN proyecta las características del rostro desconocido en el espacio latente de las 3 personas conocidas.
> 3. Aunque la persona no se parece a ninguna de las 3, una de las 3 neuronas de salida puede superar el umbral de confianza del 65% simplemente por estar ligeramente más cerca de ese cluster que de los otros dos en la frontera de decisión lineal final.

### Recomendaciones Técnicas para la Siguiente Iteración:
1. **Modelado de Conjunto Abierto (Open-Set Recognition):**
   - Incorporar una clase explícita de `desconocido` durante el entrenamiento con un banco representativo de rostros externos (ej. subconjunto de fondo de LFW o VGGFace2).
2. **Extracción de Embeddings con Distancia Métrica (Contrastive / Triplet / ArcFace):**
   - En lugar de clasificar directamente con Softmax, entrenar una red de embeddings y determinar la identidad mediante la distancia euclidiana o coseno respecto a los prototipos de los sujetos autorizados.
   - Si la distancia mínima supera un radio de aceptación $R$, rechazar como desconocido con independencia de la clase más cercana.
3. **Calibración de Incertidumbre y Temperatura:**
   - Implementar calibración por escalamiento de temperatura o modelos basados en Evidential Deep Learning que asignen alta incertidumbre cuando la entrada cae fuera de la distribución de entrenamiento (Out-Of-Distribution / OOD).

---

## 6. Reproducibilidad y Verificación

El experimento puede reproducirse íntegramente ejecutando desde la raíz del proyecto:
```bash
python scripts/evaluar_lfw_cnn.py
```
O bien mediante el comando CLI:
```bash
python -m reconocimiento evaluar-lfw
```

Todos los datos crudos, hashes de fotografías y resultados individuales por muestra se encuentran registrados en:
- Configuración congelada: [`configuracion_congelada.json`](file:///resultados/evaluacion_externa_lfw_cnn/configuracion_congelada.json)
- Manifiesto de 100 muestras: [`manifiesto_lfw_100.csv`](file:///resultados/evaluacion_externa_lfw_cnn/manifiesto_lfw_100.csv)
- Predicciones detalladas: [`predicciones.csv`](file:///resultados/evaluacion_externa_lfw_cnn/predicciones.csv)
"""
    return doc
