"""
Implementación del clasificador de Máquinas de Vectores de Soporte (SVM) para reconocimiento facial.
Basado estrictamente en las enseñanzas de clase (SVM_UTAv1.md):
- Maximización de margen geométrico 2 / ||W||.
- Kernels de clase:
  - Lineal: K(x, z) = x . z
  - Polinomial: K(x, z) = (x . z + c)^d (grados 2, 3)
  - RBF / Gaussiano: K(x, z) = exp(-gamma * ||x - z||^2)
- Escalado estándar ajustado EXCLUSIVAMENTE sobre entrenamiento (sin fuga de datos).
- Estrategia multiclase OvO (One-vs-One con 40*39/2 = 780 hiperplanos) y puntuaciones de decisión.
- Exploración sistemática de C, kernel y gamma en validación.
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from reconocimiento.clasificacion.base import (
    ClasificadorFacialBase,
    ReporteEvaluacion,
    ResumenExperimento
)


class ClasificadorSVM(ClasificadorFacialBase):
    """
    Clasificador SVM con pre-escalado fitted exclusivamente en entrenamiento
    y soporte de kernels lineal, rbf y polinomial.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Any = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        probabilidad: bool = True,
        semilla: int = 42
    ):
        self.C = float(C)
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.probabilidad = probabilidad
        self.semilla = semilla

        self.escalador = StandardScaler()
        self.modelo = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            probability=self.probabilidad,
            decision_function_shape="ovr",
            random_state=self.semilla
        )
        self.clases_: Optional[np.ndarray] = None
        self.n_vectores_soporte_: Optional[int] = None

    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Ajusta el escalador y el clasificador SVM exclusivamente con los datos de entrenamiento.
        """
        X_arr = np.asarray(X_train, dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.int32)

        # Ajuste estricto del escalador solo en train
        X_scaled = self.escalador.fit_transform(X_arr)
        self.modelo.fit(X_scaled, y_arr)
        self.clases_ = self.modelo.classes_
        self.n_vectores_soporte_ = int(np.sum(self.modelo.n_support_))

    def transformar(self, X: np.ndarray) -> np.ndarray:
        """Aplica el escalado aprendido en entrenamiento a nuevas muestras."""
        X_arr = np.asarray(X, dtype=np.float32)
        return self.escalador.transform(X_arr)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predice las clases tras escalar las muestras de entrada."""
        X_scaled = self.transformar(X)
        return self.modelo.predict(X_scaled)

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """Calcula probabilidades calibradas mediante sigmoid de Platt."""
        X_scaled = self.transformar(X)
        if self.probabilidad:
            return self.modelo.predict_proba(X_scaled)
        # Si probabilidad=False, retornar softmax de la función de decisión
        df = self.modelo.decision_function(X_scaled)
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        return exp_df / np.sum(exp_df, axis=1, keepdims=True)

    def puntuaciones_decision(self, X: np.ndarray) -> np.ndarray:
        """Retorna las distancias geométricas / márgenes respecto a los hiperplanos."""
        X_scaled = self.transformar(X)
        return self.modelo.decision_function(X_scaled)

    def guardar(self, ruta_archivo: Path) -> None:
        """Serializa a disco el pipeline de escalador y modelo SVM."""
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "escalador": self.escalador,
            "modelo": self.modelo,
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "probabilidad": self.probabilidad,
            "semilla": self.semilla,
            "n_vectores_soporte": self.n_vectores_soporte_,
            "clases": self.clases_
        }, ruta_archivo)

    @classmethod
    def cargar(cls, ruta_archivo: Path) -> "ClasificadorSVM":
        """Carga el modelo y escalador desde disco."""
        datos = joblib.load(ruta_archivo)
        instancia = cls(
            C=datos["C"],
            kernel=datos["kernel"],
            gamma=datos["gamma"],
            degree=datos["degree"],
            coef0=datos["coef0"],
            probabilidad=datos["probabilidad"],
            semilla=datos["semilla"]
        )
        instancia.escalador = datos["escalador"]
        instancia.modelo = datos["modelo"]
        instancia.n_vectores_soporte_ = datos["n_vectores_soporte"]
        instancia.clases_ = datos["clases"]
        return instancia

    @classmethod
    def seleccionar_y_entrenar(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        semilla: int = 42
    ) -> Tuple["ClasificadorSVM", List[Dict[str, Any]], Dict[str, Any]]:
        """
        Explora una rejilla explícita de configuraciones SVM según la teoría de clase:
        - Kernels: 'linear', 'rbf', 'poly'
        - C: 0.1, 1.0, 10.0, 100.0
        - gamma: 'scale', 'auto', 0.01, 0.1
        Selecciona la configuración con mayor exactitud en validación.
        """
        configuraciones_a_probar = [
            {"kernel": "linear", "C": 0.1, "gamma": "scale", "degree": 3, "coef0": 0.0},
            {"kernel": "linear", "C": 1.0, "gamma": "scale", "degree": 3, "coef0": 0.0},
            {"kernel": "linear", "C": 10.0, "gamma": "scale", "degree": 3, "coef0": 0.0},
            {"kernel": "rbf", "C": 0.1, "gamma": "scale", "degree": 3, "coef0": 0.0},
            {"kernel": "rbf", "C": 1.0, "gamma": "scale", "degree": 3, "coef0": 0.0},
            {"kernel": "rbf", "C": 10.0, "gamma": "scale", "degree": 3, "coef0": 0.0},
            {"kernel": "rbf", "C": 100.0, "gamma": "scale", "degree": 3, "coef0": 0.0},
            {"kernel": "rbf", "C": 10.0, "gamma": 0.01, "degree": 3, "coef0": 0.0},
            {"kernel": "rbf", "C": 10.0, "gamma": 0.1, "degree": 3, "coef0": 0.0},
            {"kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 2, "coef0": 1.0},
            {"kernel": "poly", "C": 10.0, "gamma": "scale", "degree": 2, "coef0": 1.0},
            {"kernel": "poly", "C": 1.0, "gamma": "scale", "degree": 3, "coef0": 1.0},
            {"kernel": "poly", "C": 10.0, "gamma": "scale", "degree": 3, "coef0": 1.0},
        ]

        historial_validaciones: List[Dict[str, Any]] = []
        mejor_exactitud_val = -1.0
        mejor_cfg = configuraciones_a_probar[1]

        for cfg in configuraciones_a_probar:
            candidato = cls(
                C=cfg["C"],
                kernel=cfg["kernel"],
                gamma=cfg["gamma"],
                degree=cfg["degree"],
                coef0=cfg["coef0"],
                probabilidad=False,  # Más rápido para búsqueda en validación
                semilla=semilla
            )
            t0 = time.perf_counter()
            candidato.entrenar(X_train, y_train)
            tiempo_fit = time.perf_counter() - t0

            y_val_pred = candidato.predecir(X_val)
            acc_val = float(np.mean(y_val_pred == y_val) * 100.0)

            registro = {
                "kernel": cfg["kernel"],
                "C": cfg["C"],
                "gamma": str(cfg["gamma"]),
                "degree": cfg["degree"],
                "coef0": cfg["coef0"],
                "n_vectores_soporte": candidato.n_vectores_soporte_,
                "exactitud_validacion": round(acc_val, 2),
                "tiempo_entrenamiento_s": round(tiempo_fit, 4)
            }
            historial_validaciones.append(registro)

            if acc_val > mejor_exactitud_val:
                mejor_exactitud_val = acc_val
                mejor_cfg = cfg

        # Entrenar modelo final seleccionado con calibración de probabilidad activa
        modelo_elegido = cls(
            C=mejor_cfg["C"],
            kernel=mejor_cfg["kernel"],
            gamma=mejor_cfg["gamma"],
            degree=mejor_cfg["degree"],
            coef0=mejor_cfg["coef0"],
            probabilidad=True,
            semilla=semilla
        )
        modelo_elegido.entrenar(X_train, y_train)

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
        Ejecuta la evaluación en las tres particiones y calcula métricas.
        """
        rep_train = self.evaluar_particion(X_train, y_train, ids_train, "entrenamiento")
        rep_val = self.evaluar_particion(X_val, y_val, ids_val, "validacion")
        rep_test = self.evaluar_particion(X_test, y_test, ids_test, "prueba")

        brecha = round(rep_train.exactitud_porcentaje - rep_test.exactitud_porcentaje, 2)

        config = {
            "kernel": self.kernel,
            "C": self.C,
            "gamma": str(self.gamma),
            "degree": self.degree,
            "coef0": self.coef0,
            "estrategia_multiclase": "One-vs-One (780 clasificadores)",
            "escalador": "StandardScaler (fitted solo en train)",
            "n_vectores_soporte": self.n_vectores_soporte_,
            "semilla": self.semilla
        }

        return ResumenExperimento(
            algoritmo="svm",
            dataset=nombre_dataset,
            configuracion=config,
            reporte_train=rep_train,
            reporte_val=rep_val,
            reporte_test=rep_test,
            brecha_generalizacion_puntos=brecha
        )
