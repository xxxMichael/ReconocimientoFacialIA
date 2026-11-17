"""
Interfaz base y estructuras de reporte para clasificadores faciales.
"""
from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class ReporteEvaluacion:
    """Reporte detallado de evaluación para una partición."""
    particion: str
    total_muestras: int
    aciertos: int
    errores: int
    exactitud_porcentaje: float
    tasa_error_porcentaje: float
    matriz_confusion: List[List[int]]
    etiquetas_clases: List[int]
    muestras_fallidas: List[Dict[str, Any]]
    tiempo_total_segundos: float
    tiempo_promedio_por_muestra_ms: float


@dataclass
class ResumenExperimento:
    """Resumen completo del ciclo entrenar-validar-probar."""
    algoritmo: str
    dataset: str
    configuracion: Dict[str, Any]
    reporte_train: ReporteEvaluacion
    reporte_val: ReporteEvaluacion
    reporte_test: ReporteEvaluacion
    brecha_generalizacion_puntos: float  # Train Acc - Test Acc

    def guardar(self, directorio_salida: Path) -> None:
        directorio_salida.mkdir(parents=True, exist_ok=True)
        ruta_json = directorio_salida / "metricas_evaluacion.json"
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

        ruta_txt = directorio_salida / "resumen_reporte.txt"
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(f"=== RESULTADOS DE EVALUACIÓN: {self.algoritmo.upper()} ===\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Configuración: {self.configuracion}\n\n")
            for rep in [self.reporte_train, self.reporte_val, self.reporte_test]:
                f.write(f"--- Partición: {rep.particion.upper()} ---\n")
                f.write(f"  Total muestras:  {rep.total_muestras}\n")
                f.write(f"  Aciertos:        {rep.aciertos}\n")
                f.write(f"  Errores:         {rep.errores}\n")
                f.write(f"  Exactitud:       {rep.exactitud_porcentaje:.2f}%\n")
                f.write(f"  Tasa de error:   {rep.tasa_error_porcentaje:.2f}%\n")
                f.write(f"  Tiempo inferencia: {rep.tiempo_promedio_por_muestra_ms:.3f} ms/muestra\n\n")
            f.write(f"Brecha de generalización (Train - Test): {self.brecha_generalizacion_puntos:.2f} p.p.\n")


class ClasificadorFacialBase(ABC):
    """Clase base abstracta para los cinco modelos."""

    @abstractmethod
    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predecir(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        pass

    def evaluar_particion(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ids: List[str],
        nombre_particion: str
    ) -> ReporteEvaluacion:
        """
        Evalúa una partición registrando aciertos, errores, exactitud porcentual,
        matriz de confusión y muestras fallidas.
        """
        t0 = time.perf_counter()
        y_pred = self.predecir(X)
        tiempo_total = time.perf_counter() - t0
        total = len(y)
        aciertos = int(np.sum(y_pred == y))
        errores = total - aciertos
        exactitud = (aciertos / total) * 100.0 if total > 0 else 0.0
        tasa_error = 100.0 - exactitud
        ms_por_muestra = (tiempo_total / total * 1000.0) if total > 0 else 0.0

        etiquetas_unicas = sorted(list(set(y).union(set(y_pred))))
        matriz = confusion_matrix(y, y_pred, labels=etiquetas_unicas).tolist()

        fallos = []
        for i in range(total):
            if y_pred[i] != y[i]:
                fallos.append({
                    "id_muestra": ids[i] if i < len(ids) else f"indice_{i}",
                    "clase_real": int(y[i]),
                    "clase_predicha": int(y_pred[i])
                })

        return ReporteEvaluacion(
            particion=nombre_particion,
            total_muestras=total,
            aciertos=aciertos,
            errores=errores,
            exactitud_porcentaje=round(exactitud, 2),
            tasa_error_porcentaje=round(tasa_error, 2),
            matriz_confusion=matriz,
            etiquetas_clases=[int(c) for c in etiquetas_unicas],
            muestras_fallidas=fallos,
            tiempo_total_segundos=round(tiempo_total, 4),
            tiempo_promedio_por_muestra_ms=round(ms_por_muestra, 3)
        )


def guardar_artefactos_completos(
    resumen: ResumenExperimento,
    dir_salida: Path,
    ids_map: Dict[str, List[str]],
    y_test_pred: np.ndarray,
    ids_test: List[str],
    y_test_real: np.ndarray,
    configuraciones_validadas: Optional[List[Dict[str, Any]]] = None,
    scores_o_probs: Optional[np.ndarray] = None,
    historia_entrenamiento: Optional[Dict[str, List[float]]] = None
) -> None:
    """
    Guarda el conjunto completo y estandarizado de artefactos requeridos por las especificaciones:
    - metricas_evaluacion.json
    - resumen_reporte.txt
    - configuracion.json
    - identificadores_particion.json
    - configuraciones_validadas.json (si aplica)
    - predicciones_test.csv
    - matriz_confusion_test.csv
    - matriz_confusion_test.png
    - curvas_aprendizaje.png (si aplica historia de entrenamiento)
    """
    dir_salida.mkdir(parents=True, exist_ok=True)

    # 1. Metricas y resumen de texto
    resumen.guardar(dir_salida)

    # 2. Configuración aislada
    with open(dir_salida / "configuracion.json", "w", encoding="utf-8") as f:
        json.dump(resumen.configuracion, f, indent=2)

    # 3. Identificadores de partición
    with open(dir_salida / "identificadores_particion.json", "w", encoding="utf-8") as f:
        json.dump(ids_map, f, indent=2)

    # 4. Configuraciones validadas (exploración en conjunto de validación)
    if configuraciones_validadas is not None:
        with open(dir_salida / "configuraciones_validadas.json", "w", encoding="utf-8") as f:
            json.dump(configuraciones_validadas, f, indent=2)

    # 5. Predicciones en conjunto de prueba en formato CSV
    ruta_pred_csv = dir_salida / "predicciones_test.csv"
    with open(ruta_pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_muestra", "clase_real", "clase_predicha", "es_correcta", "confianza_o_score"])
        for i, id_m in enumerate(ids_test):
            real = int(y_test_real[i])
            pred = int(y_test_pred[i])
            ok = (real == pred)
            score_val = ""
            if scores_o_probs is not None:
                if scores_o_probs.ndim == 2 and scores_o_probs.shape[1] > pred:
                    score_val = f"{float(scores_o_probs[i, pred]):.4f}"
                elif scores_o_probs.ndim == 1:
                    score_val = f"{float(scores_o_probs[i]):.4f}"
            writer.writerow([id_m, real, pred, ok, score_val])

    # 6. Matriz de confusión en CSV
    matriz = np.array(resumen.reporte_test.matriz_confusion)
    etiquetas = resumen.reporte_test.etiquetas_clases
    ruta_cm_csv = dir_salida / "matriz_confusion_test.csv"
    with open(ruta_cm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clase_real \\ clase_pred"] + [f"clase_{c}" for c in etiquetas])
        for idx, c_real in enumerate(etiquetas):
            fila = [f"clase_{c_real}"] + [int(v) for v in matriz[idx]]
            writer.writerow(fila)

    # 7. Matriz de confusión en PNG
    plt.figure(figsize=(10, 8))
    plt.imshow(matriz, cmap="Blues", interpolation="nearest")
    plt.title(f"Matriz de Confusión - {resumen.algoritmo.upper()} (Conjunto de Prueba)")
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    plt.colorbar(label="Conteo de Muestras")
    plt.tight_layout()
    plt.savefig(dir_salida / "matriz_confusion_test.png", dpi=150)
    plt.close()

    # 8. Curvas de aprendizaje (si aplica historia)
    if historia_entrenamiento is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        epocas = range(1, len(historia_entrenamiento.get("loss_train", [])) + 1)

        # Gráfico de Pérdida
        if "loss_train" in historia_entrenamiento:
            axes[0].plot(epocas, historia_entrenamiento["loss_train"], label="Train Loss", color="royalblue")
        if "loss_val" in historia_entrenamiento:
            axes[0].plot(epocas, historia_entrenamiento["loss_val"], label="Val Loss", color="crimson")
        axes[0].set_title(f"Curvas de Pérdida ({resumen.algoritmo.upper()})")
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Pérdida (CrossEntropy)")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        # Gráfico de Exactitud
        if "acc_train" in historia_entrenamiento:
            axes[1].plot(epocas, historia_entrenamiento["acc_train"], label="Train Acc (%)", color="royalblue")
        if "acc_val" in historia_entrenamiento:
            axes[1].plot(epocas, historia_entrenamiento["acc_val"], label="Val Acc (%)", color="crimson")
        axes[1].set_title(f"Curvas de Exactitud ({resumen.algoritmo.upper()})")
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Exactitud (%)")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(dir_salida / "curvas_aprendizaje.png", dpi=150)
        plt.close()
