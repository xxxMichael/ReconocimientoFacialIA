"""
Implementación del clasificador de Árbol de Decisión para reconocimiento facial.
Basado estrictamente en las enseñanzas de clase (DT.md):
- Criterio de división: Ganancia de Información basada en Entropía de Shannon (DT.md pág 9-10):
  Entropy(S) = - sum(p_i * log2(p_i))
  Gain(S, A) = Entropy(S) - sum(|S_v|/|S| * Entropy(S_v))
- Exploración explícita de hiperparámetros en el conjunto de validación:
  profundidades máximas y muestras mínimas por hoja.
- Registro de profundidad efectiva y número de hojas finales.
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from reconocimiento.clasificacion.base import (
    ClasificadorFacialBase,
    ReporteEvaluacion,
    ResumenExperimento
)


class ClasificadorArbol(ClasificadorFacialBase):
    """
    Clasificador de Árbol de Decisión para reconocimiento facial
    utilizando el criterio de Entropía/Ganancia de Información de clase.
    """

    def __init__(
        self,
        criterio: str = "entropy",
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        class_weight: Optional[Any] = None,
        semilla: int = 42
    ):
        self.criterio = criterio
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.semilla = semilla

        self.modelo = DecisionTreeClassifier(
            criterion=self.criterio,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.semilla
        )
        self.clases_: Optional[np.ndarray] = None
        self.profundidad_efectiva_: Optional[int] = None
        self.n_hojas_: Optional[int] = None
        self.importancias_caracteristicas_: Optional[np.ndarray] = None

    def entrenar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Ajusta el árbol de decisión sobre el conjunto de entrenamiento."""
        X_arr = np.asarray(X_train, dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.int32)
        sw = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None
        self.modelo.fit(X_arr, y_arr, sample_weight=sw)
        self.clases_ = self.modelo.classes_
        self.profundidad_efectiva_ = int(self.modelo.get_depth())
        self.n_hojas_ = int(self.modelo.get_n_leaves())
        self.importancias_caracteristicas_ = self.modelo.feature_importances_

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predice las identidades para las muestras dadas."""
        X_arr = np.asarray(X, dtype=np.float32)
        return self.modelo.predict(X_arr)

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """Estima las probabilidades a posteriori en la hoja terminal."""
        X_arr = np.asarray(X, dtype=np.float32)
        return self.modelo.predict_proba(X_arr)

    def guardar(self, ruta_archivo: Path) -> None:
        """Serializa el modelo entrenado a disco."""
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "modelo": self.modelo,
            "criterio": self.criterio,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "semilla": self.semilla,
            "profundidad_efectiva": self.profundidad_efectiva_,
            "n_hojas": self.n_hojas_,
            "clases": self.clases_
        }, ruta_archivo)

    @classmethod
    def cargar(cls, ruta_archivo: Path) -> "ClasificadorArbol":
        """Carga un modelo serializado desde disco."""
        datos = joblib.load(ruta_archivo)
        instancia = cls(
            criterio=datos["criterio"],
            max_depth=datos["max_depth"],
            min_samples_leaf=datos["min_samples_leaf"],
            semilla=datos["semilla"]
        )
        instancia.modelo = datos["modelo"]
        instancia.profundidad_efectiva_ = datos["profundidad_efectiva"]
        instancia.n_hojas_ = datos["n_hojas"]
        instancia.clases_ = datos["clases"]
        return instancia

    @classmethod
    def seleccionar_y_entrenar(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        rejilla_max_depth: List[Optional[int]] = [5, 10, 15, 20, None],
        rejilla_min_samples_leaf: List[int] = [1, 2, 4],
        class_weight: Optional[Any] = None,
        sample_weight: Optional[np.ndarray] = None,
        metrica: str = "acc",
        semilla: int = 42
    ) -> Tuple["ClasificadorArbol", List[Dict[str, Any]], Dict[str, Any]]:
        """
        Explora las combinaciones de hiperparámetros sobre el conjunto de validación
        y entrena el modelo con la mejor configuración encontrada.
        """
        historial_validaciones: List[Dict[str, Any]] = []
        mejor_score_val = -1.0
        mejor_cfg = {"max_depth": None, "min_samples_leaf": 1}

        for md in rejilla_max_depth:
            for msl in rejilla_min_samples_leaf:
                arbol_candidato = cls(
                    criterio="entropy",
                    max_depth=md,
                    min_samples_leaf=msl,
                    class_weight=class_weight,
                    semilla=semilla
                )
                t0 = time.perf_counter()
                arbol_candidato.entrenar(X_train, y_train, sample_weight=sample_weight)
                tiempo_fit = time.perf_counter() - t0

                y_val_pred = arbol_candidato.predecir(X_val)
                acc_val = float(np.mean(y_val_pred == y_val) * 100.0)

                if metrica == "macro_recall":
                    recalls = []
                    for c in np.unique(y_val):
                        mask = (y_val == c)
                        if np.sum(mask) > 0:
                            recalls.append(float(np.mean(y_val_pred[mask] == c)) * 100.0)
                    score_val = float(np.mean(recalls)) if recalls else acc_val
                else:
                    score_val = acc_val

                registro = {
                    "max_depth": md,
                    "min_samples_leaf": msl,
                    "criterio": "entropy",
                    "profundidad_efectiva": arbol_candidato.profundidad_efectiva_,
                    "n_hojas": arbol_candidato.n_hojas_,
                    "exactitud_validacion": round(acc_val, 2),
                    "score_validacion": round(score_val, 2),
                    "tiempo_entrenamiento_s": round(tiempo_fit, 4)
                }
                historial_validaciones.append(registro)

                # Criterio de selección: mayor score en validación; en empate, menor profundidad
                if score_val > mejor_score_val:
                    mejor_score_val = score_val
                    mejor_cfg = {"max_depth": md, "min_samples_leaf": msl}

        # Entrenar modelo final seleccionado
        modelo_elegido = cls(
            criterio="entropy",
            max_depth=mejor_cfg["max_depth"],
            min_samples_leaf=mejor_cfg["min_samples_leaf"],
            class_weight=class_weight,
            semilla=semilla
        )
        modelo_elegido.entrenar(X_train, y_train, sample_weight=sample_weight)

        return modelo_elegido, historial_validaciones, mejor_cfg

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
        Ejecuta el ciclo completo de evaluación sobre las tres particiones.
        """
        rep_train = self.evaluar_particion(X_train, y_train, ids_train, "entrenamiento")
        rep_val = self.evaluar_particion(X_val, y_val, ids_val, "validacion")
        rep_test = self.evaluar_particion(X_test, y_test, ids_test, "prueba")

        brecha = round(rep_train.exactitud_porcentaje - rep_test.exactitud_porcentaje, 2)

        config = {
            "criterio": self.criterio,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "profundidad_efectiva": self.profundidad_efectiva_,
            "n_hojas": self.n_hojas_,
            "semilla": self.semilla
        }

        return ResumenExperimento(
            algoritmo="arbol",
            dataset=nombre_dataset,
            configuracion=config,
            reporte_train=rep_train,
            reporte_val=rep_val,
            reporte_test=rep_test,
            brecha_generalizacion_puntos=brecha
        )
