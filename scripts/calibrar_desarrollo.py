"""Calibra los umbrales de desarrollo con LFW de calibración y la sesión 04 exportada."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.entrenar_desarrollo import BINS, CUADRICULA, HILOS, RESOLUCION, SIGMA

from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.regla_acceso import evaluar_decision_acceso
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.fabrica import crear_detector
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises, normalizar_intensidades, normalizar_resolucion,
)


MODELOS = {
    "ann": (ClasificadorANN, "modelo.pt"),
    "arbol": (ClasificadorArbol, "modelo.joblib"),
    "cnn": (ClasificadorCNN, "modelo.pt"),
    "knn": (ClasificadorKNN, "modelo.joblib"),
    "svm": (ClasificadorSVM, "modelo.joblib"),
}
CALIBRACION_LFW = RAIZ / "data" / "desconocidos_lfw" / "calibracion"
RESULTADOS = RAIZ / "resultados"
SALIDA = "calibracion_desarrollo.json"


@dataclass(frozen=True)
class Intento:
    estado: str
    persona_real: Optional[str] = None
    persona_predicha: Optional[str] = None
    puntaje: Optional[float] = None

    def abre(self, umbral, autorizados):
        return self.puntaje is not None and evaluar_decision_acceso(
            self.persona_predicha, self.puntaje, umbral, autorizados
        ).es_autorizado


def sha256(contenido):
    return hashlib.sha256(contenido).hexdigest()


def cargar_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def preparar_negativos(detector):
    with (CALIBRACION_LFW / "manifiesto.csv").open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if len(filas) != 100 or any(
        len({fila[campo] for fila in filas}) != 100
        for campo in ("identidad_normalizada", "ruta_imagen", "sha256")
    ):
        raise ValueError("El manifiesto debe contener 100 negativos distintos de calibración.")
    rutas = [(RAIZ / fila["ruta_imagen"]).resolve() for fila in filas]
    if any(ruta.parent != CALIBRACION_LFW.resolve() for ruta in rutas):
        raise ValueError("El manifiesto contiene imágenes fuera de la carpeta de calibración LFW.")

    estados = []
    imagenes = []
    for fila, ruta in zip(filas, rutas):
        contenido = ruta.read_bytes()
        if sha256(contenido) != fila["sha256"]:
            raise ValueError(f"Contenido de calibración modificado: {ruta}")
        imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if imagen is None:
            raise ValueError(f"No se pudo decodificar {ruta}")
        resultado = detector.detectar(imagen, es_pre_recortada=False)
        estados.append(resultado.estado.value)
        if resultado.estado == EstadoDeteccion.UN_ROSTRO:
            recorte = resultado.rostro_recortado
            if recorte is None or recorte.size == 0:
                raise ValueError(f"YuNet presentó un recorte vacío para {ruta}")
            gris = a_escala_de_grises(recorte) if recorte.ndim == 3 else recorte
            normalizada = normalizar_resolucion(gris, RESOLUCION)
            intensidades = normalizar_intensidades(normalizada, "float32_0_1")
            imagenes.append(suavizado_gaussiano(intensidades, sigma=SIGMA))
    if not imagenes:
        raise ValueError("YuNet no halló ningún rostro único entre los negativos de calibración.")
    return filas, estados, np.asarray(imagenes, dtype=np.float32)


def elegir_clase(puntajes, clases):
    puntajes = np.asarray(puntajes, dtype=float)
    if puntajes.shape != (len(clases),) or not np.all(np.isfinite(puntajes)):
        raise ValueError("Vector de puntajes inválido.")
    if np.any((puntajes < 0) | (puntajes > 1)):
        raise ValueError("Los puntajes de clase deben estar entre cero y uno.")
    indice = int(np.argmax(puntajes))
    return clases[indice], float(puntajes[indice])


def cargar_validacion(documento, clases):
    filas = documento["predicciones"]
    if any(fila["sesion"] != "sesion_04" or fila["particion"] != "validacion" for fila in filas):
        raise ValueError("La exportación debe contener exclusivamente la validación de la sesión 04.")
    if len({fila["id"] for fila in filas}) != len(filas):
        raise ValueError("La validación incluye intentos duplicados.")
    intentos = []
    for fila in filas:
        if not fila["intento"]:
            continue
        estado, persona = fila["estado"], fila["persona"]
        if persona not in clases:
            raise ValueError(f"Identidad sin clase en validación: {persona}")
        if estado == "procesada":
            predicha, puntaje = elegir_clase(fila["probabilidades"], clases)
            intentos.append(Intento(estado, persona, predicha, puntaje))
        else:
            intentos.append(Intento(estado, persona))
    if not intentos or len(intentos) != documento["intentos"]:
        raise ValueError("La cantidad de intentos no coincide con la exportación de validación.")
    return intentos


def medir(umbral, positivos, negativos, clases):
    aperturas_negativas = [m for m in negativos if m.abre(umbral, clases)]
    aperturas_positivas = [m for m in positivos if m.abre(umbral, clases)]
    rechazados = len(positivos) - len(aperturas_positivas)
    correctos = sum(m.persona_real == m.persona_predicha for m in aperturas_positivas)
    confusion = Counter((m.persona_real, m.persona_predicha) for m in aperturas_positivas)
    return {
        "umbral": umbral,
        "far_pct": 100 * len(aperturas_negativas) / len(negativos),
        "falsa_aceptacion": len(aperturas_negativas),
        "rechazo_correcto": len(negativos) - len(aperturas_negativas),
        "falsas_aceptaciones_por_clase": dict(Counter(m.persona_predicha for m in aperturas_negativas)),
        "frr_puerta_pct": 100 * rechazados / len(positivos),
        "rechazos_autorizados": rechazados,
        "aperturas_autorizados": len(aperturas_positivas),
        "identificaciones_correctas": correctos,
        "confusiones_entre_autorizados": len(aperturas_positivas) - correctos,
        "matriz_identificacion_entre_aperturas": [
            [confusion[real, predicha] for predicha in clases] for real in clases
        ],
    }


def elegir_umbral(curva, total_positivos):
    viables = [fila for fila in curva if 10 * fila["rechazos_autorizados"] <= 3 * total_positivos]
    return min(viables, key=lambda fila: (
        fila["falsa_aceptacion"], fila["rechazos_autorizados"],
        -fila["identificaciones_correctas"], -fila["umbral"],
    ), default=None)


def calibrar_modelo(directorio, filas, estados, imagenes, histogramas):
    tipo, nombre_archivo = MODELOS[directorio.name]
    ruta_modelo = directorio / nombre_archivo
    ruta_validacion = directorio / "validacion.json"
    mapeo = cargar_json(directorio / "clases_mapeo.json")
    clases = [mapeo["id_to_persona"][str(i)] for i in range(len(mapeo["id_to_persona"]))]
    if clases != mapeo["personas_autorizadas"] or len(clases) != len(set(clases)):
        raise ValueError(f"Mapeo de clases incoherente en {directorio}")
    positivos = cargar_validacion(cargar_json(ruta_validacion), clases)
    modelo = tipo.cargar(ruta_modelo)
    if list(modelo.clases_) != list(range(len(clases))):
        raise ValueError(f"Orden de clases incompatible con la web en {directorio}")

    entrada = imagenes if directorio.name == "cnn" else histogramas
    puntajes = modelo.predecir_probabilidades(entrada)
    if len(puntajes) != len(imagenes):
        raise ValueError(f"Cantidad de predicciones incorrecta en {directorio}")
    predicciones = iter(puntajes)
    negativos = []
    for estado in estados:
        if estado == EstadoDeteccion.UN_ROSTRO.value:
            persona, puntaje = elegir_clase(next(predicciones), clases)
            negativos.append(Intento(estado, persona_predicha=persona, puntaje=puntaje))
        else:
            negativos.append(Intento(estado))

    curva = [medir(paso / 100, positivos, negativos, clases) for paso in range(50, 101)]
    elegido = elegir_umbral(curva, len(positivos))
    return {
        "modelo": directorio.name,
        "alcance": "desarrollo",
        "habilita_produccion": False,
        "estado": "umbral_viable_en_desarrollo" if elegido is not None else "sin_umbral_viable",
        "puntajes": "Máximo de las salidas de clase. No son probabilidades calibradas de autorización.",
        "regla": "Abre con clase autorizada y puntaje >= umbral. FRR puerta <= 30%; minimizar FAR, luego FRR; maximizar identificaciones correctas, luego el umbral.",
        "negativos": {
            "fuente": (CALIBRACION_LFW / "manifiesto.csv").relative_to(RAIZ).as_posix(),
            "sha256_manifiesto": sha256((CALIBRACION_LFW / "manifiesto.csv").read_bytes()),
            "dominio": "LFW funneled prealineado; dominio distinto a webcam.",
            "intentos": len(negativos), "estados_deteccion": dict(Counter(estados)),
            "predicciones": [
                {
                    "identidad_lfw": fila["identidad_lfw"], "ruta_imagen": fila["ruta_imagen"],
                    "sha256": fila["sha256"], "estado_deteccion": intento.estado,
                    "persona_predicha": intento.persona_predicha, "puntaje": intento.puntaje,
                }
                for fila, intento in zip(filas, negativos)
            ],
        },
        "autorizados": {
            "fuente": ruta_validacion.relative_to(RAIZ).as_posix(),
            "sha256_exportacion": sha256(ruta_validacion.read_bytes()),
            "sesion": "sesion_04", "intentos": len(positivos),
            "estados": dict(Counter(m.estado for m in positivos)),
        },
        "sha256_modelo": sha256(ruta_modelo.read_bytes()),
        "clases_en_orden": clases,
        "umbral_elegido": elegido["umbral"] if elegido is not None else None,
        "resultado_umbral_elegido": elegido,
        "curva": curva,
    }


def validar_pipeline(detector, extractor, informe):
    parametros_detector = {
        "tipo": "YuNet", "sha256_onnx": sha256(detector.ruta_modelo.read_bytes()),
        "score_threshold": detector.score_threshold, "nms_threshold": detector.nms_threshold,
        "margen_expansion": detector.margen_expansion,
    }
    if any(informe["detector"].get(clave) != valor for clave, valor in parametros_detector.items()):
        raise ValueError("El ONNX o los parámetros de YuNet difieren del desarrollo.")
    if informe["preprocesamiento"] != {
        "resolucion": list(RESOLUCION), "intensidad": "float32_0_1", "sigma": SIGMA,
    }:
        raise ValueError("El preprocesamiento difiere del desarrollo.")
    if informe["caracteristicas"] != {
        "cuadricula": list(CUADRICULA), "bins": BINS, "dimension": extractor.dimension_total,
    }:
        raise ValueError("Las características regionales difieren del desarrollo.")


def calibrar(directorio, actualizar_trazabilidad=False):
    directorio = directorio.resolve()
    if directorio.parent != RESULTADOS.resolve() or not directorio.name.startswith("desarrollo_yunet_"):
        raise ValueError("Se requiere un directorio de desarrollo YuNet en resultados/.")
    informe = cargar_json(directorio / "informe.json")
    if set(informe["modelos"]) != set(MODELOS):
        raise ValueError("El entrenamiento de desarrollo debe finalizar con los cinco modelos.")
    carpetas = [directorio / nombre for nombre in MODELOS]
    salidas = [carpeta / SALIDA for carpeta in carpetas]
    if actualizar_trazabilidad and not all(ruta.is_file() for ruta in salidas):
        raise FileNotFoundError("La actualización requiere las cinco calibraciones existentes.")
    if not actualizar_trazabilidad and any(ruta.exists() for ruta in salidas):
        raise FileExistsError("Ya existe una calibración en este directorio de desarrollo.")

    detector = crear_detector()
    extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS, rango_intensidad=(0.0, 1.0))
    validar_pipeline(detector, extractor, informe)
    filas, estados, imagenes = preparar_negativos(detector)
    histogramas, _ = extractor.extraer_lote(imagenes)
    informes = [calibrar_modelo(carpeta, filas, estados, imagenes, histogramas) for carpeta in carpetas]
    if actualizar_trazabilidad:
        for ruta, nuevo in zip(salidas, informes):
            anterior = cargar_json(ruta)
            anterior["negativos"].setdefault("predicciones", nuevo["negativos"]["predicciones"])
            if anterior != nuevo:
                raise ValueError(f"La actualización cambiaría resultados existentes de {ruta}.")

    with TemporaryDirectory(prefix="calibracion_", dir=directorio) as temporal:
        temporales = [Path(temporal) / f"{carpeta.name}.json" for carpeta in carpetas]
        for ruta, resultado in zip(temporales, informes):
            ruta.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for temporal, destino in zip(temporales, salidas):
            if actualizar_trazabilidad:
                temporal.replace(destino)
            else:
                temporal.rename(destino)
    for carpeta, resultado in zip(carpetas, informes):
        elegido = resultado["resultado_umbral_elegido"]
        if elegido is None:
            print(f"{carpeta.name}: sin umbral viable")
        else:
            print(f"{carpeta.name}: umbral={elegido['umbral']:.2f}, FAR={elegido['far_pct']:.2f}%, "
                  f"FRR puerta={elegido['frr_puerta_pct']:.2f}%, confusiones={elegido['confusiones_entre_autorizados']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directorio", type=Path, help="Directorio de resultados de desarrollo YuNet")
    parser.add_argument("--actualizar-trazabilidad", action="store_true",
                        help="Añade el detalle de predicciones tras verificar que los resultados coinciden")
    argumentos = parser.parse_args()
    torch.set_num_threads(HILOS)
    cv2.setNumThreads(HILOS)
    with threadpool_limits(limits=HILOS):
        calibrar(argumentos.directorio, argumentos.actualizar_trazabilidad)


if __name__ == "__main__":
    main()
