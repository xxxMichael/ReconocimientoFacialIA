"""
Script de comparación exhaustiva entre detectores faciales:
1. DetectorFacial (Cascadas de Haar Multi-Vista: Frontal + Perfiles en ambas direcciones)
2. DetectorYuNet (Red Neuronal Liviana basada en YuNet / FaceDetectorYN)

Evalúa por pose:
- Localizaciones correctas (IoU >= 0.5 con caja de referencia)
- Pérdidas de rostro (SIN_ROSTRO)
- Falsos positivos en ropa, pelo o fondo (IoU < 0.5 o detección errónea)
Evalúa además:
- Escenas vacías (sin falsos positivos)
- Escenas con 2 personas (detección de VARIOS_ROSTROS)
- Latencia en milisegundos y FPS efectivos.
"""
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from reconocimiento.adquisicion.propio import ManifiestoDataset
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet


def evaluar_detectores(
    ruta_manifiesto: Path = Path("data/propio/manifiesto.csv"),
    umbral_iou: float = 0.5
) -> Dict[str, Any]:
    manifiesto = ManifiestoDataset(ruta_manifiesto)
    det_haar = DetectorFacial()
    det_yunet = DetectorYuNet()

    poses = ["frontal", "izquierda", "derecha", "arriba", "abajo"]

    resultados: Dict[str, Dict[str, Dict[str, int]]] = {
        "haar": {p: {"total": 0, "localiza": 0, "pierde": 0, "falso_positivo": 0} for p in poses},
        "yunet": {p: {"total": 0, "localiza": 0, "pierde": 0, "falso_positivo": 0} for p in poses}
    }

    muestras_evaluadas = 0

    for reg in manifiesto.registros:
        p_img = Path(reg.ruta)
        if not p_img.is_file():
            continue

        pose = reg.pose if reg.pose in poses else "frontal"
        img = cv2.imread(str(p_img))
        if img is None:
            continue

        muestras_evaluadas += 1
        resultados["haar"][pose]["total"] += 1
        resultados["yunet"][pose]["total"] += 1

        # Caja de referencia humana: aprobada, o en su defecto la mejor estimación de rostro
        ref_box = reg.caja_aprobada
        if ref_box is None:
            # Para fotos sin caja humana aún, YuNet provee referencia de comparación robusta
            res_ref = det_yunet.detectar(img)
            if res_ref.estado == EstadoDeteccion.UN_ROSTRO:
                ref_box = res_ref.recuadro

        # 1. Evaluar Haar
        res_h = det_haar.detectar(img)
        if res_h.estado == EstadoDeteccion.UN_ROSTRO and res_h.recuadro is not None:
            if ref_box is not None:
                iou, _ = det_haar.calcular_iou_y_iomin(res_h.recuadro, ref_box)
                if iou >= umbral_iou:
                    resultados["haar"][pose]["localiza"] += 1
                else:
                    # Caja sobre ropa, pelo o zona descentrada
                    resultados["haar"][pose]["falso_positivo"] += 1
            else:
                resultados["haar"][pose]["localiza"] += 1
        elif res_h.estado == EstadoDeteccion.SIN_ROSTRO:
            resultados["haar"][pose]["pierde"] += 1
        else:
            # VARIOS_ROSTROS en imagen de una sola persona es un falso positivo extra
            resultados["haar"][pose]["falso_positivo"] += 1

        # 2. Evaluar YuNet
        res_y = det_yunet.detectar(img)
        if res_y.estado == EstadoDeteccion.UN_ROSTRO and res_y.recuadro is not None:
            if ref_box is not None:
                iou, _ = det_yunet.calcular_iou_y_iomin(res_y.recuadro, ref_box)
                if iou >= umbral_iou:
                    resultados["yunet"][pose]["localiza"] += 1
                else:
                    resultados["yunet"][pose]["falso_positivo"] += 1
            else:
                resultados["yunet"][pose]["localiza"] += 1
        elif res_y.estado == EstadoDeteccion.SIN_ROSTRO:
            resultados["yunet"][pose]["pierde"] += 1
        else:
            resultados["yunet"][pose]["falso_positivo"] += 1

    # 3. Evaluar escenas vacías
    img_negra = np.zeros((480, 640, 3), dtype=np.uint8)
    img_blanca = np.full((480, 640, 3), 255, dtype=np.uint8)
    img_ruido = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    vacias_haar_fp = 0
    vacias_yunet_fp = 0
    for v in [img_negra, img_blanca, img_ruido]:
        if det_haar.detectar(v).estado != EstadoDeteccion.SIN_ROSTRO:
            vacias_haar_fp += 1
        if det_yunet.detectar(v).estado != EstadoDeteccion.SIN_ROSTRO:
            vacias_yunet_fp += 1

    # 4. Evaluar escena con dos personas (1 frontal + 1 girada)
    p_front = Path("data/propio/imagenes/persona_1/sesion_01/frontal/persona_1_sesion_01_frontal_f03.jpg")
    p_turn = Path("data/propio/imagenes/persona_1/sesion_02/izquierda/persona_1_sesion_02_izquierda_f06.jpg")
    dos_personas_ok_haar = False
    dos_personas_ok_yunet = False

    if p_front.is_file() and p_turn.is_file():
        img_f = cv2.imread(str(p_front))
        img_t = cv2.imread(str(p_turn))
        escena_dos = np.hstack([img_f, img_t])
        res_h2 = det_haar.detectar(escena_dos)
        res_y2 = det_yunet.detectar(escena_dos)
        dos_personas_ok_haar = (res_h2.estado == EstadoDeteccion.VARIOS_ROSTROS)
        dos_personas_ok_yunet = (res_y2.estado == EstadoDeteccion.VARIOS_ROSTROS)

    # 5. Medir FPS y latencia
    img_bench = cv2.imread(str(p_front)) if p_front.is_file() else np.zeros((480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        det_haar.detectar(img_bench)
        det_yunet.detectar(img_bench)

    N_ITERS = 20
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        det_haar.detectar(img_bench)
    lat_haar_ms = ((time.perf_counter() - t0) / N_ITERS) * 1000.0
    fps_haar = 1000.0 / lat_haar_ms if lat_haar_ms > 0 else 0

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        det_yunet.detectar(img_bench)
    lat_yunet_ms = ((time.perf_counter() - t0) / N_ITERS) * 1000.0
    fps_yunet = 1000.0 / lat_yunet_ms if lat_yunet_ms > 0 else 0

    return {
        "muestras_evaluadas": muestras_evaluadas,
        "desglose": resultados,
        "escenas_vacias": {
            "haar_falsos_positivos": vacias_haar_fp,
            "yunet_falsos_positivos": vacias_yunet_fp
        },
        "dos_personas": {
            "haar_detecta_varios": dos_personas_ok_haar,
            "yunet_detecta_varios": dos_personas_ok_yunet
        },
        "rendimiento": {
            "haar": {"latencia_ms": round(lat_haar_ms, 2), "fps": round(fps_haar, 1)},
            "yunet": {"latencia_ms": round(lat_yunet_ms, 2), "fps": round(fps_yunet, 1)}
        }
    }


def imprimir_informe_comparativo(datos: Dict[str, Any]) -> None:
    print("=" * 80)
    print("INFORME COMPARATIVO: DETECTOR HAAR MULTI-VISTA vs DETECTOR YUNET (DEEP LEARNING)")
    print("=" * 80)
    print(f"Total imágenes analizadas: {datos['muestras_evaluadas']}\n")

    print(f"{'Pose':<12} | {'Haar Localiza':<14} | {'Haar Pierde':<12} | {'Haar FP/Ropa':<13} | {'YuNet Localiza':<15} | {'YuNet Pierde':<13} | {'YuNet FP/Ropa':<13}")
    print("-" * 105)

    poses = ["frontal", "izquierda", "derecha", "arriba", "abajo"]
    totales = {"h_loc": 0, "h_pie": 0, "h_fp": 0, "y_loc": 0, "y_pie": 0, "y_fp": 0, "n": 0}

    for p in poses:
        h = datos["desglose"]["haar"][p]
        y = datos["desglose"]["yunet"][p]
        n = h["total"]
        totales["n"] += n
        totales["h_loc"] += h["localiza"]
        totales["h_pie"] += h["pierde"]
        totales["h_fp"] += h["falso_positivo"]
        totales["y_loc"] += y["localiza"]
        totales["y_pie"] += y["pierde"]
        totales["y_fp"] += y["falso_positivo"]

        h_loc_str = f"{h['localiza']}/{n}"
        h_pie_str = f"{h['pierde']}/{n}"
        h_fp_str = f"{h['falso_positivo']}/{n}"
        y_loc_str = f"{y['localiza']}/{n}"
        y_pie_str = f"{y['pierde']}/{n}"
        y_fp_str = f"{y['falso_positivo']}/{n}"

        print(f"{p:<12} | {h_loc_str:<14} | {h_pie_str:<12} | {h_fp_str:<13} | {y_loc_str:<15} | {y_pie_str:<13} | {y_fp_str:<13}")

    print("-" * 105)
    print(f"{'TOTAL':<12} | {totales['h_loc']:<14} | {totales['h_pie']:<12} | {totales['h_fp']:<13} | {totales['y_loc']:<15} | {totales['y_pie']:<13} | {totales['y_fp']:<13}")
    print()

    ev = datos["escenas_vacias"]
    print(f"Escenas vacías (fondo negro, blanco, ruido):")
    print(f"  - Haar falsos positivos:  {ev['haar_falsos_positivos']}/3 (SIN_ROSTRO garantizado: {ev['haar_falsos_positivos'] == 0})")
    print(f"  - YuNet falsos positivos: {ev['yunet_falsos_positivos']}/3 (SIN_ROSTRO garantizado: {ev['yunet_falsos_positivos'] == 0})")
    print()

    dp = datos["dos_personas"]
    print(f"Escena con dos personas (1 frontal + 1 girada):")
    print(f"  - Haar detecta VARIOS_ROSTROS:  {dp['haar_detecta_varios']}")
    print(f"  - YuNet detecta VARIOS_ROSTROS: {dp['yunet_detecta_varios']}")
    print()

    rend = datos["rendimiento"]
    print(f"Rendimiento computacional (640x480):")
    print(f"  - Haar Multi-Vista: {rend['haar']['latencia_ms']} ms/frame (~{rend['haar']['fps']} FPS)")
    print(f"  - YuNet DNN:        {rend['yunet']['latencia_ms']} ms/frame (~{rend['yunet']['fps']} FPS)")
    print("=" * 80)


if __name__ == "__main__":
    resultado = evaluar_detectores()
    imprimir_informe_comparativo(resultado)
