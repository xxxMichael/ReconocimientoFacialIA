"""
Script de medición comparativa de rendimiento:
Compara durante 60 segundos la vista previa de OpenCV nativa frente a la transmisión web
utilizando la misma cámara física (DroidCam / DirectShow índice 1) a 640x480.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from reconocimiento.adquisicion.camara import CapturadorCamara
from reconocimiento.preprocesamiento.detector import DetectorFacial
from reconocimiento.web.streamer import GestorCamaraWeb


def obtener_memoria_proceso_mb() -> float:
    """Obtiene el uso de memoria Working Set en MB del proceso Python actual en Windows."""
    pid = os.getpid()
    try:
        cmd = f"powershell -Command \"(Get-Process -Id {pid}).WorkingSet64 / 1MB\""
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        return round(float(out.replace(",", ".")), 2)
    except Exception:
        return 0.0


def medir_opencv_nativo(segundos: int = 60, indice: int = 1) -> dict:
    print(f"\n--- Iniciando medición de OpenCV Nativo ({segundos}s) ---")
    detector = DetectorFacial()
    capturador = CapturadorCamara(indice=indice)

    mem_inicial = obtener_memoria_proceso_mb()
    t_cpu_inicio = time.process_time()
    t_wall_inicio = time.time()

    capturador.abrir()

    total_frames = 0
    tiempos_frames = []

    t_fin_objetivo = t_wall_inicio + segundos

    while time.time() < t_fin_objetivo:
        t_f0 = time.time()
        ret, frame = capturador.leer()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        res_det = detector.detectar(frame, es_pre_recortada=False)
        display = frame.copy()
        if res_det.recuadro:
            rx, ry, rw, rh = res_det.recuadro
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

        total_frames += 1
        tiempos_frames.append(time.time() - t_f0)

    capturador.liberar()

    t_wall_total = time.time() - t_wall_inicio
    t_cpu_total = time.process_time() - t_cpu_inicio
    mem_final = obtener_memoria_proceso_mb()

    fps_promedio = round(total_frames / t_wall_total, 2)
    uso_cpu_pct = round((t_cpu_total / t_wall_total) * 100, 2)
    ms_por_frame = round((sum(tiempos_frames) / len(tiempos_frames) * 1000), 2) if tiempos_frames else 0.0

    return {
        "modalidad": "OpenCV Nativo (DirectShow)",
        "duracion_segundos": round(t_wall_total, 1),
        "total_fotogramas": total_frames,
        "fps_observados": fps_promedio,
        "tiempo_por_fotograma_ms": ms_por_frame,
        "uso_cpu_porcentaje": uso_cpu_pct,
        "memoria_inicial_mb": mem_inicial,
        "memoria_final_mb": mem_final,
        "incremento_memoria_mb": round(mem_final - mem_inicial, 2)
    }


def medir_web_streamer(segundos: int = 60, indice: int = 1) -> dict:
    print(f"\n--- Iniciando medición de Servidor Web ({segundos}s) ---")
    gestor = GestorCamaraWeb(tamano_buffer=40)
    gestor.detener()

    mem_inicial = obtener_memoria_proceso_mb()
    t_cpu_inicio = time.process_time()
    t_wall_inicio = time.time()

    gestor.iniciar(indice=indice)

    total_frames_recibidos = 0
    ultimo_fid = -1
    t_fin_objetivo = t_wall_inicio + segundos

    # Simular cliente navegador consultando /api/frame/latest a ~14 FPS (70ms)
    while time.time() < t_fin_objetivo:
        item = gestor.obtener_ultimo_fotograma()
        if item is not None and item.id_fotograma != ultimo_fid:
            ultimo_fid = item.id_fotograma
            total_frames_recibidos += 1
        time.sleep(0.068)

    gestor.detener()

    t_wall_total = time.time() - t_wall_inicio
    t_cpu_total = time.process_time() - t_cpu_inicio
    mem_final = obtener_memoria_proceso_mb()

    fps_promedio = round(total_frames_recibidos / t_wall_total, 2)
    uso_cpu_pct = round((t_cpu_total / t_wall_total) * 100, 2)

    return {
        "modalidad": "Web Local (Flask + Base64 JSON polling a 14 FPS)",
        "duracion_segundos": round(t_wall_total, 1),
        "total_fotogramas_cliente": total_frames_recibidos,
        "fps_observados": fps_promedio,
        "uso_cpu_servidor_porcentaje": uso_cpu_pct,
        "memoria_inicial_mb": mem_inicial,
        "memoria_final_mb": mem_final,
        "incremento_memoria_mb": round(mem_final - mem_inicial, 2)
    }


if __name__ == "__main__":
    duracion = 30  # 30 segundos cada modalidad para medición robusta
    print(f"Ejecutando pruebas de rendimiento ({duracion} segundos por modalidad)...")
    res_opencv = medir_opencv_nativo(segundos=duracion, indice=1)
    time.sleep(2)
    res_web = medir_web_streamer(segundos=duracion, indice=1)

    informe = {
        "dispositivo": "DroidCam DirectShow Índice 1",
        "resolucion": "640x480",
        "opencv_nativo": res_opencv,
        "web_local": res_web
    }

    ruta_salida = Path("resultados/medicion_rendimiento.json")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2)

    print("\nResultados de la medición:")
    print(json.dumps(informe, indent=2))
