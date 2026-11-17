"""
Módulo de clasificación facial con los cinco algoritmos del proyecto:
KNN, Árbol de Decisión, SVM, ANN Densa y CNN.
"""
from reconocimiento.clasificacion.base import (
    ClasificadorFacialBase,
    ReporteEvaluacion,
    ResumenExperimento,
    guardar_artefactos_completos
)
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.regla_acceso import (
    DecisionAcceso,
    ETIQUETAS_DESCONOCIDO_RESERVADAS,
    evaluar_decision_acceso,
)

__all__ = [
    "ClasificadorFacialBase",
    "ReporteEvaluacion",
    "ResumenExperimento",
    "guardar_artefactos_completos",
    "ClasificadorKNN",
    "ClasificadorArbol",
    "ClasificadorSVM",
    "ClasificadorANN",
    "ClasificadorCNN",
    "DecisionAcceso",
    "ETIQUETAS_DESCONOCIDO_RESERVADAS",
    "evaluar_decision_acceso",
]
