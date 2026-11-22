"""Compara Haar y YuNet sobre las sesiones 01 a 04 del manifiesto local."""

import csv
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.adquisicion.propio import RegistroCaptura, calcular_hash_archivo
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet

MANIFIESTO = RAIZ / "data/propio/manifiesto_local.csv"
MODELO = RAIZ / "modelos/detector/face_detection_yunet_2023mar.onnx"
SALIDA = RAIZ / "resultados/verificacion_yunet/informe.json"
SESIONES = ("sesion_01", "sesion_02", "sesion_03", "sesion_04")
UMBRAL_IOU = 0.5
CALENTAMIENTO = 3


def cargar_capturas():
    with MANIFIESTO.open(encoding="utf-8-sig", newline="") as archivo:
        return [
            RegistroCaptura(
                **{**fila, "consentimiento": fila["consentimiento"].lower() in {"true", "1", "si", "yes"}}
            )
            for fila in csv.DictReader(archivo)
            if fila["sesion"] in SESIONES
        ]


def resumir_iou(muestras):
    valores = [muestra["iou"] for muestra in muestras if muestra["iou"] is not None]
    return {
        "con_referencia": sum(muestra["referencia"] is not None for muestra in muestras),
        "con_iou": len(valores),
        "iou_medio": mean(valores) if valores else None,
        "iou_mediano": median(valores) if valores else None,
        "iou_mayor_o_igual_al_umbral": sum(valor >= UMBRAL_IOU for valor in valores),
        "iou_menor_al_umbral": sum(valor < UMBRAL_IOU for valor in valores),
    }


def resumir(muestras):
    estados = Counter(muestra["estado"] for muestra in muestras)
    tiempos = [muestra["tiempo_ms"] for muestra in muestras]
    return {
        "total": len(muestras),
        "estados": {estado.value: estados[estado.value] for estado in EstadoDeteccion},
        **resumir_iou(muestras),
        "tiempo_ms": {"media": mean(tiempos), "mediana": median(tiempos), "total": sum(tiempos)},
    }


def verificar():
    capturas = cargar_capturas()
    faltantes = set(SESIONES) - {captura.sesion for captura in capturas}
    if faltantes:
        raise ValueError(f"Faltan sesiones en el manifiesto local: {sorted(faltantes)}")

    detectores = {"haar": DetectorFacial(), "yunet": DetectorYuNet(ruta_modelo=str(MODELO))}
    resultados = {nombre: [] for nombre in detectores}
    omitidas = []

    for captura in capturas:
        if not captura.consentimiento or captura.decision_humana == "excluida":
            omitidas.append({"id": captura.identificador, "motivo": "sin consentimiento o excluida"})
            continue
        imagen = cv2.imread(str(RAIZ / captura.ruta))
        if imagen is None:
            omitidas.append({"id": captura.identificador, "motivo": "imagen ilegible"})
            continue

        referencia = captura.caja_aprobada
        if captura.decision_humana not in {"aprobada", "corregida"} or (
            referencia is not None and min(referencia[2:]) <= 0
        ):
            referencia = None

        for nombre, detector in detectores.items():
            if not resultados[nombre]:
                for _ in range(CALENTAMIENTO):
                    detector.detectar(imagen)

            inicio = perf_counter()
            deteccion = detector.detectar(imagen)
            tiempo_ms = (perf_counter() - inicio) * 1000
            iou = None
            if referencia is not None and deteccion.recuadro is not None:
                iou, _ = detector.calcular_iou_y_iomin(deteccion.recuadro, referencia)

            resultados[nombre].append({
                "id": captura.identificador,
                "sesion": captura.sesion,
                "pose": captura.pose,
                "estado": deteccion.estado.value,
                "caja": deteccion.recuadro,
                "referencia": referencia,
                "origen_referencia": (captura.origen_caja or "sin especificar") if referencia else None,
                "iou": iou,
                "tiempo_ms": tiempo_ms,
            })

    if not resultados["yunet"]:
        raise ValueError("No se pudo evaluar ninguna imagen de las sesiones elegidas.")

    informe = {
        "manifiesto": MANIFIESTO.relative_to(RAIZ).as_posix(),
        "modelo_yunet_sha256": calcular_hash_archivo(MODELO),
        "sesiones": dict(sorted(Counter(captura.sesion for captura in capturas).items())),
        "imagenes_evaluadas": len(resultados["yunet"]),
        "omitidas": omitidas,
        "umbral_iou": UMBRAL_IOU,
        "referencia_iou": (
            "Cajas aprobadas o corregidas en revisión humana. Las cajas de origen vivo y "
            "recalculado provienen de detectores, por lo que la referencia es imperfecta. "
            "El IoU resume detecciones con una caja comparable."
        ),
        "medicion_tiempo": (
            f"Una llamada a detectar por imagen y detector, tras {CALENTAMIENTO} llamadas "
            "de calentamiento con la primera imagen. Excluye lectura e IoU."
        ),
        "entorno": {"plataforma": platform.platform(), "opencv": cv2.__version__, "hilos_opencv": cv2.getNumThreads()},
        "detectores": {},
    }
    for nombre, muestras in resultados.items():
        poses = sorted({muestra["pose"] for muestra in muestras})
        origenes = sorted({muestra["origen_referencia"] for muestra in muestras if muestra["referencia"] is not None})
        informe["detectores"][nombre] = {
            "total": resumir(muestras),
            "por_pose": {
                pose: resumir([muestra for muestra in muestras if muestra["pose"] == pose])
                for pose in poses
            },
            "iou_por_origen": {
                origen: resumir_iou([muestra for muestra in muestras if muestra["origen_referencia"] == origen])
                for origen in origenes
            },
            "muestras": muestras,
        }
    return informe


if __name__ == "__main__":
    informe = verificar()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{informe['imagenes_evaluadas']} imágenes evaluadas. Informe: {SALIDA}")
