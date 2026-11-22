"""Entrena los cinco clasificadores con las sesiones 01–03 y valida con la 04."""

import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
import sklearn
import torch
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

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
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.fabrica import crear_detector
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion,
)

MANIFIESTO = RAIZ / "data/propio/manifiesto_local.csv"
RAIZ_IMAGENES = RAIZ / "data/propio/imagenes"
SESIONES = ("sesion_01", "sesion_02", "sesion_03", "sesion_04")
SEMILLA = 42
HILOS = 4
RESOLUCION = (64, 64)
SIGMA = 0.5
CUADRICULA = (4, 4)
BINS = 16
SOLAPAMIENTO_MINIMO = 0.5
REVISIONES_VALIDAS = {"aprobada", "corregida"}


def guardar_json(ruta, datos):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cargar_registros(solo_base: bool = True):
    with MANIFIESTO.open(encoding="utf-8-sig", newline="") as archivo:
        todas_filas = list(csv.DictReader(archivo))

    sesiones_base = {"sesion_01", "sesion_02", "sesion_03", "sesion_04"}
    sesiones_en_manifiesto = {f["sesion"] for f in todas_filas}
    if not sesiones_base.issubset(sesiones_en_manifiesto):
        raise ValueError("El manifiesto debe contener las cuatro sesiones de desarrollo.")

    if solo_base:
        filas = [f for f in todas_filas if f["sesion"] in SESIONES]
    else:
        filas = [
            f for f in todas_filas
            if f["particion"] in {"entrenamiento", "validacion"} and f["sesion"] != "sesion_05"
        ]

    if len({fila["identificador"] for fila in filas}) != len(filas):
        raise ValueError("El manifiesto contiene identificadores de desarrollo duplicados.")
    for fila in filas:
        if fila["particion"] != obtener_particion_esperada_propio(fila["sesion"]):
            raise ValueError(f"Partición incorrecta para {fila['identificador']}.")
    filas_canonicas = json.dumps(filas, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    registros = [
        RegistroCaptura(**{**fila, "consentimiento": fila["consentimiento"].lower() in {"true", "1", "si", "yes"}})
        for fila in filas
    ]
    return registros, hashlib.sha256(filas_canonicas.encode("utf-8")).hexdigest()


def preparar_captura(registro, detector, hashes_por_particion):
    decision = registro.decision_humana
    entrenamiento = registro.particion == "entrenamiento"
    caja = registro.caja_aprobada if decision in REVISIONES_VALIDAS else None
    referencia = caja if caja and min(caja[2:]) > 0 else None
    muestra = {
        "id": registro.identificador, "persona": registro.persona, "sesion": registro.sesion,
        "particion": registro.particion, "pose": registro.pose, "decision_humana": decision,
        "referencia": referencia, "origen_referencia": registro.origen_caja if referencia else None,
        "intento": False, "estado_deteccion": None, "caja_yunet": None,
        "iou": None, "iomin": None, "sha256_imagen": None,
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

    if referencia and entrenamiento:
        x, y, ancho, alto = referencia
        h_imagen, w_imagen = imagen.shape[:2]
        if not (0 <= x < x + ancho <= w_imagen and 0 <= y < y + alto <= h_imagen):
            muestra["estado"] = "caja_revisada_fuera_de_imagen"
            return muestra, None
        recorte = detector.recortar_con_margen_seguro(imagen, referencia)
    else:
        deteccion = detector.detectar(imagen)
        muestra["estado_deteccion"] = deteccion.estado.value
        muestra["caja_yunet"] = deteccion.recuadro
        if deteccion.estado != EstadoDeteccion.UN_ROSTRO:
            muestra["estado"] = deteccion.estado.value.lower()
            return muestra, None
        if referencia and deteccion.recuadro:
            iou, iomin = detector.calcular_iou_y_iomin(deteccion.recuadro, referencia)
            muestra.update(iou=iou, iomin=iomin)
            if max(iou, iomin) < SOLAPAMIENTO_MINIMO:
                muestra["estado"] = "localizacion_incorrecta"
                return muestra, None
        recorte = deteccion.rostro_recortado

    if recorte is None or recorte.size == 0:
        muestra["estado"] = "recorte_vacio"
        return muestra, None
    gris = a_escala_de_grises(recorte)
    normalizada = normalizar_resolucion(gris, RESOLUCION)
    normalizada = normalizar_intensidades(normalizada, "float32_0_1")
    muestra["estado"] = "procesada"
    return muestra, suavizado_gaussiano(normalizada, sigma=SIGMA)


def resumen_muestras(muestras):
    intentos = [m for m in muestras if m["intento"]]
    procesadas = sum(m["estado"] == "procesada" for m in intentos)
    return {
        "registros": len(muestras), "intentos": len(intentos), "procesadas": procesadas,
        "fallos_procesamiento": len(intentos) - procesadas,
        "estados": dict(Counter(m["estado"] for m in muestras)),
        "por_sesion": dict(Counter(m["sesion"] for m in muestras)),
    }


def metricas_validacion(muestras):
    resumen = resumen_muestras(muestras)
    if any(m["intento"] for m in muestras):
        aciertos = sum(m["correcta"] is True for m in muestras)
        resumen.update(
            aciertos=aciertos,
            errores_clasificacion=resumen["procesadas"] - aciertos,
            exactitud_condicional_pct=porcentaje(aciertos, resumen["procesadas"]),
            exactitud_extremo_a_extremo_pct=porcentaje(aciertos, resumen["intentos"]),
        )
    return resumen


def porcentaje(aciertos, total):
    return round(100 * aciertos / total, 2) if total else None


def matriz_datos(particion, muestras, imagenes, persona_to_id):
    procesadas = [m for m in muestras if m["particion"] == particion and m["estado"] == "procesada"]
    ids = [m["id"] for m in procesadas]
    return (
        np.stack([imagenes[id_muestra] for id_muestra in ids]),
        np.array([persona_to_id[m["persona"]] for m in procesadas]),
        ids,
    )


def entrenar_modelo(nombre, X_train, y_train, X_val, y_val):
    if nombre == "knn":
        modelo = ClasificadorKNN(k=min(5, len(y_train)), metrica="euclidean", ponderacion="uniform")
        modelo.entrenar(X_train, y_train)
        return modelo, {"k": modelo.k, "metrica": modelo.metrica, "ponderacion": modelo.ponderacion}, None
    if nombre == "arbol":
        modelo, historial, elegida = ClasificadorArbol.seleccionar_y_entrenar(
            X_train, y_train, X_val, y_val, semilla=SEMILLA
        )
        return modelo, {**elegida, "criterio": modelo.criterio, "n_hojas": modelo.n_hojas_}, historial
    if nombre == "svm":
        modelo, historial, elegida = ClasificadorSVM.seleccionar_y_entrenar(
            X_train, y_train, X_val, y_val, semilla=SEMILLA
        )
        return modelo, {
            **elegida, "escalador": "StandardScaler ajustado con entrenamiento",
            "n_vectores_soporte": modelo.n_vectores_soporte_,
        }, historial
    n_clases = len(np.unique(y_train))
    if nombre == "ann":
        modelo = ClasificadorANN(n_entrada=X_train.shape[1], n_clases=n_clases, semilla=SEMILLA)
    else:
        modelo = ClasificadorCNN(n_clases=n_clases, semilla=SEMILLA)
    modelo.entrenar_con_validacion(X_train, y_train, X_val, y_val)
    return modelo, {
        "epocas": modelo.epocas, "mejor_epoca": modelo.mejor_epoca_, "optimizador": "Adam",
        "perdida": "CrossEntropyLoss", "lr": modelo.lr, "dropout": modelo.dropout,
        "weight_decay": modelo.weight_decay, "tamano_lote": modelo.tamano_lote,
        "parametros_entrenables": modelo.red_wrapper.contar_parametros(),
    }, modelo.historia_entrenamiento


def evaluar_modelo(modelo, X_val, y_val, ids_val, validacion, personas):
    probabilidades = modelo.predecir_probabilidades(X_val)
    clases = np.asarray(modelo.clases_, dtype=int)
    predicciones_web = clases[np.argmax(probabilidades, axis=1)]
    predicciones_nativas = modelo.predecir(X_val)
    indice_por_id = {id_muestra: i for i, id_muestra in enumerate(ids_val)}
    clase_por_persona = {persona: i for i, persona in enumerate(personas)}
    predicciones = []
    for muestra in validacion:
        indice = indice_por_id.get(muestra["id"])
        registro = {
            **muestra, "clase_real": clase_por_persona.get(muestra["persona"]),
            "clase_predicha": None, "persona_predicha": None, "clase_predicha_nativa": None,
            "probabilidades": None, "confianza": None, "correcta": None,
        }
        if indice is not None:
            clase = int(predicciones_web[indice])
            registro.update(
                clase_predicha=clase, persona_predicha=personas[clase],
                clase_predicha_nativa=int(predicciones_nativas[indice]),
                probabilidades=probabilidades[indice].tolist(),
                confianza=float(probabilidades[indice].max()),
                correcta=bool(clase == y_val[indice]),
            )
        predicciones.append(registro)
    return {
        **metricas_validacion(predicciones),
        "exactitud_condicional_prediccion_nativa_pct": porcentaje(
            int(np.sum(predicciones_nativas == y_val)), len(y_val)
        ),
        "etiquetas_matriz_confusion": personas,
        "matriz_confusion_condicional": confusion_matrix(y_val, predicciones_web, labels=clases).tolist(),
        "por_persona": {
            persona: metricas_validacion([m for m in predicciones if m["persona"] == persona])
            for persona in personas
        },
        "por_pose": {
            pose: metricas_validacion([m for m in predicciones if m["pose"] == pose])
            for pose in sorted({m["pose"] for m in predicciones})
        },
        "predicciones": predicciones,
    }


def resumir_clasificacion_entrenamiento(modelo, X_train, y_train):
    probabilidades = modelo.predecir_probabilidades(X_train)
    clases = np.asarray(modelo.clases_, dtype=int)
    predicciones = clases[np.argmax(probabilidades, axis=1)]
    aciertos = int(np.sum(predicciones == y_train))
    return {
        "muestras": len(y_train),
        "aciertos": aciertos,
        "exactitud_pct": porcentaje(aciertos, len(y_train)),
    }


def desarrollar(solo_base: bool = False):
    torch.set_num_threads(HILOS)
    cv2.setNumThreads(HILOS)
    registros, hash_filas = cargar_registros(solo_base=solo_base)
    detector = crear_detector()
    muestras = []
    imagenes = {}
    hashes_por_particion = {}
    for registro in registros:
        muestra, imagen = preparar_captura(registro, detector, hashes_por_particion)
        muestras.append(muestra)
        if imagen is not None:
            imagenes[registro.identificador] = imagen

    entrenamiento = [m for m in muestras if m["particion"] == "entrenamiento"]
    validacion = [m for m in muestras if m["particion"] == "validacion"]
    personas = sorted({m["persona"] for m in entrenamiento if m["estado"] == "procesada"})
    personas_val = {m["persona"] for m in validacion if m["intento"]}
    if len(personas) < 2 or personas_val != set(personas):
        raise ValueError("Entrenamiento y validación deben cubrir las mismas identidades autorizadas.")
    if not any(m["estado"] == "procesada" for m in validacion):
        raise ValueError("Validación no tiene detecciones utilizables para elegir parámetros.")
    persona_to_id = {persona: i for i, persona in enumerate(personas)}
    mapa = {
        "persona_to_id": persona_to_id,
        "id_to_persona": {i: persona for persona, i in persona_to_id.items()},
        "personas_autorizadas": personas,
    }
    imagenes_train, y_train, _ = matriz_datos("entrenamiento", muestras, imagenes, persona_to_id)
    imagenes_val, y_val, ids_val = matriz_datos("validacion", muestras, imagenes, persona_to_id)
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS, rango_intensidad=(0.0, 1.0))
    X_train, _ = extractor.extraer_lote(imagenes_train)
    X_val, _ = extractor.extraer_lote(imagenes_val)
    ious = [m["iou"] for m in validacion if m["iou"] is not None]
    informe = {
        "manifiesto": MANIFIESTO.relative_to(RAIZ).as_posix(),
        "sha256_filas_manifiesto_01_a_04": hash_filas,
        "canonizacion_hash": "JSON UTF-8, claves ordenadas, filas en orden CSV y sin espacios entre separadores.",
        "semilla": SEMILLA,
        "entorno": {
            "plataforma": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "opencv": cv2.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__,
            "hilos": HILOS,
        },
        "detector": {
            "tipo": "YuNet", "sha256_onnx": calcular_hash_archivo(detector.ruta_modelo),
            "score_threshold": detector.score_threshold, "nms_threshold": detector.nms_threshold,
            "margen_expansion": detector.margen_expansion,
            "criterio_localizacion": "Con referencia revisada se exige IoU >= 0.5 o IoMin >= 0.5.",
        },
        "preprocesamiento": {"resolucion": RESOLUCION, "intensidad": "float32_0_1", "sigma": SIGMA},
        "caracteristicas": {"cuadricula": CUADRICULA, "bins": BINS, "dimension": extractor.dimension_total},
        "clases": mapa,
        "entrenamiento": resumen_muestras(entrenamiento),
        "validacion": {
            **resumen_muestras(validacion),
            "referencias_revisadas": sum(m["referencia"] is not None for m in validacion),
            "cajas_comparables": len(ious),
            "iou_medio": mean(ious) if ious else None,
            "iou_mediano": median(ious) if ious else None,
            "iou_menor_a_0_5": sum(iou < SOLAPAMIENTO_MINIMO for iou in ious),
            "criterio_prediccion": "Argmax de las probabilidades, como en la inferencia web. Etiquetas en orden de clase.",
        },
        "modelos": {},
    }
    salida = RAIZ / "resultados" / f"desarrollo_yunet_{datetime.now():%Y%m%d_%H%M%S_%f}"
    salida.mkdir(parents=True, exist_ok=False)
    guardar_json(salida / "muestras.json", muestras)
    print(f"Entrenamiento: {len(y_train)} procesadas. Validación: {len(y_val)}/{informe['validacion']['intentos']}.", flush=True)

    for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
        print(f"Entrenando {nombre.upper()}...", flush=True)
        inicio = perf_counter()
        entrada_train, entrada_val = (imagenes_train, imagenes_val) if nombre == "cnn" else (X_train, X_val)
        modelo, configuracion, historial = entrenar_modelo(nombre, entrada_train, y_train, entrada_val, y_val)
        configuracion.update(semilla=SEMILLA, representacion="espacial 64x64" if nombre == "cnn" else "histogramas regionales")
        nombre_archivo = "modelo.pt" if nombre in {"ann", "cnn"} else "modelo.joblib"
        dir_modelo = salida / nombre
        modelo.guardar(dir_modelo / nombre_archivo)
        guardar_json(dir_modelo / "clases_mapeo.json", mapa)
        guardar_json(dir_modelo / "configuracion.json", configuracion)
        if historial is not None:
            guardar_json(dir_modelo / "historial_ajuste.json", historial)
        metricas = evaluar_modelo(modelo, entrada_val, y_val, ids_val, validacion, personas)
        guardar_json(dir_modelo / "validacion.json", metricas)
        informe["modelos"][nombre] = {
            "artefacto": f"{nombre}/{nombre_archivo}", "configuracion": configuracion,
            "entrenamiento": resumir_clasificacion_entrenamiento(modelo, entrada_train, y_train),
            "validacion": {clave: valor for clave, valor in metricas.items() if clave != "predicciones"},
            "tiempo_segundos": round(perf_counter() - inicio, 3),
        }

    guardar_json(salida / "informe.json", informe)
    return salida


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrena los cinco clasificadores de desarrollo.")
    parser.add_argument("--solo-base", action="store_true", help="Usa únicamente las sesiones 01 a 03 para entrenamiento.")
    args = parser.parse_args()
    with threadpool_limits(limits=HILOS):
        resultado = desarrollar(solo_base=args.solo_base)
    print(f"Informe de validación: {resultado / 'informe.json'}")
