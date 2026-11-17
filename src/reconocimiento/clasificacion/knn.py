"""
Implementación del clasificador K-Nearest Neighbors (KNN) para reconocimiento facial.
Basado estrictamente en las enseñanzas de clase:
- Métricas de distancia (kNN.md pág 6): Euclidiana, Manhattan, Minkowski, Coseno, Chebyshev.
- Estimación de probabilidades por proporción de vecinos (kNN.md pág 5).
- Desempate determinista por menor etiqueta (inventario de ejercicios AI/KNN/KNN1.py).
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from reconocimiento.clasificacion.base import (
    ClasificadorFacialBase,
    ReporteEvaluacion,
    ResumenExperimento
)


class ClasificadorKNN(ClasificadorFacialBase):
    """
    Clasificador KNN para reconocimiento facial.
    """

    def __init__(
        self,
        k: int = 5,
        metrica: str = "euclidean",
        p_minkowski: int = 2,
        ponderacion: str = "uniform",
        regla_desempate: str = "menor_etiqueta"
    ):
        self.k = k
        self.metrica = metrica.lower()
        self.p_minkowski = p_minkowski
        self.ponderacion = ponderacion
        self.regla_desempate = regla_desempate

        # Mapeo a parámetros de scikit-learn garantizando equivalencia matemática
        metric_sklearn = self.metrica
        if self.metrica in {"euclidiana", "euclidean"}:
            metric_sklearn = "euclidean"
        elif self.metrica in {"manhattan", "l1"}:
            metric_sklearn = "manhattan"
        elif self.metrica in {"chebyshev", "linf"}:
            metric_sklearn = "chebyshev"
        elif self.metrica in {"coseno", "cosine"}:
            metric_sklearn = "cosine"
        elif self.metrica in {"minkowski"}:
            metric_sklearn = "minkowski"

        self.modelo = KNeighborsClassifier(
            n_neighbors=self.k,
            weights=self.ponderacion,
            metric=metric_sklearn,
            p=self.p_minkowski,
            algorithm="auto"
        )

        self.X_entrenamiento: Optional[np.ndarray] = None
        self.y_entrenamiento: Optional[np.ndarray] = None
        self.clases_: Optional[np.ndarray] = None

    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Almacena las instancias de entrenamiento (algoritmo 'lazy' o basado en instancias).
        """
        self.X_entrenamiento = np.asarray(X_train, dtype=np.float32)
        self.y_entrenamiento = np.asarray(y_train, dtype=np.int32)
        self.modelo.fit(self.X_entrenamiento, self.y_entrenamiento)
        self.clases_ = self.modelo.classes_

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Retorna las identidades predichas por votación de los K vecinos."""
        X_arr = np.asarray(X, dtype=np.float32)
        return self.modelo.predict(X_arr)

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula P(y = c | X) = (num vecinos de clase c) / K  [kNN.md pág 5]
        """
        X_arr = np.asarray(X, dtype=np.float32)
        return self.modelo.predict_proba(X_arr)

    def guardar(self, ruta_archivo: Path) -> None:
        """Serializa el modelo KNN y los ejemplos de entrenamiento en disco."""
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "k": self.k,
            "metrica": self.metrica,
            "ponderacion": self.ponderacion,
            "p_minkowski": self.p_minkowski,
            "regla_desempate": self.regla_desempate,
            "modelo": self.modelo,
            "X_entrenamiento": self.X_entrenamiento,
            "y_entrenamiento": self.y_entrenamiento,
            "clases": self.clases_
        }, ruta_archivo)

    @classmethod
    def cargar(cls, ruta_archivo: Path) -> "ClasificadorKNN":
        """Restaura un modelo KNN entrenado con sus ejemplos exactos de entrenamiento."""
        ruta_archivo = Path(ruta_archivo)
        if not ruta_archivo.exists():
            raise FileNotFoundError(f"Archivo de modelo KNN no encontrado: {ruta_archivo}")
        datos = joblib.load(ruta_archivo)
        instancia = cls(
            k=datos.get("k", 5),
            metrica=datos.get("metrica", "euclidean"),
            p_minkowski=datos.get("p_minkowski", 2),
            ponderacion=datos.get("ponderacion", "uniform"),
            regla_desempate=datos.get("regla_desempate", "menor_etiqueta")
        )
        instancia.modelo = datos["modelo"]
        instancia.X_entrenamiento = datos.get("X_entrenamiento")
        instancia.y_entrenamiento = datos.get("y_entrenamiento")
        instancia.clases_ = datos.get("clases")
        return instancia

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

        # Matriz de confusión
        etiquetas_unicas = sorted(list(set(y).union(set(y_pred))))
        matriz = confusion_matrix(y, y_pred, labels=etiquetas_unicas).tolist()

        # Registro de muestras fallidas
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

    def ejecutar_experimento_completo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        ids_train: List[str],
        X_val: np.ndarray,
        y_val: np.ndarray,
        ids_val: List[str],
        X_test: np.ndarray,
        y_test: np.ndarray,
        ids_test: List[str],
        nombre_dataset: str = "olivetti"
    ) -> ResumenExperimento:
        """
        Ejecuta el ciclo completo:
        1. Entrenar con X_train
        2. Evaluar en Train, Val y Test
        3. Calcular brecha de generalización
        """
        self.entrenar(X_train, y_train)

        rep_train = self.evaluar_particion(X_train, y_train, ids_train, "entrenamiento")
        rep_val = self.evaluar_particion(X_val, y_val, ids_val, "validacion")
        rep_test = self.evaluar_particion(X_test, y_test, ids_test, "prueba")

        brecha = round(rep_train.exactitud_porcentaje - rep_test.exactitud_porcentaje, 2)

        config = {
            "k": self.k,
            "metrica": self.metrica,
            "ponderacion": self.ponderacion,
            "p_minkowski": self.p_minkowski,
            "regla_desempate": self.regla_desempate
        }

        return ResumenExperimento(
            algoritmo="knn",
            dataset=nombre_dataset,
            configuracion=config,
            reporte_train=rep_train,
            reporte_val=rep_val,
            reporte_test=rep_test,
            brecha_generalizacion_puntos=brecha
        )
