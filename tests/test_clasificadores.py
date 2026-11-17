"""
Pruebas unitarias para los clasificadores faciales (KNN, Árbol, SVM, ANN, CNN):
1. Verificación de identidad estricta de particiones frente a resultados/knn/identificadores_particion.json.
2. Comprobación de que no hay fuga de datos ni solapamiento entre particiones.
3. Prueba de recarga de modelos desde disco (modelo.joblib y modelo.pt) comprobando
   que las predicciones son 100% idénticas e invariantes a la serialización.
"""
import csv
import json
from pathlib import Path
import unittest
import numpy as np

from reconocimiento.adquisicion.olivetti import cargar_olivetti
from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.cnn import ClasificadorCNN


class TestClasificadoresFaciales(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = cargar_olivetti(semilla=42)
        cls.ruta_ref_ids = Path("resultados/knn/identificadores_particion.json")
        with open(cls.ruta_ref_ids, "r", encoding="utf-8") as f:
            cls.ids_referencia = json.load(f)

        # Preprocesar características para modelos tabulares
        X_test_prep = np.array([suavizado_gaussiano(img, sigma=0.5) for img in cls.dataset.X_test])
        extractor = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16, rango_intensidad=(0.0, 1.0))
        cls.X_test_feat, _ = extractor.extraer_lote(X_test_prep)
        cls.X_test_prep = X_test_prep
        cls.y_test = cls.dataset.y_test

    def test_01_particiones_estrictamente_alineadas_con_referencia(self):
        """Verifica que el generador reproduce exactamente los identificadores de KNN."""
        self.assertEqual(self.dataset.ids_train, self.ids_referencia["entrenamiento"])
        self.assertEqual(self.dataset.ids_val, self.ids_referencia["validacion"])
        self.assertEqual(self.dataset.ids_test, self.ids_referencia["prueba"])

    def test_02_particiones_disjuntas_y_cobertura_completa_clases(self):
        """Verifica que no hay solapamiento y que las 40 identidades existen en las 3 particiones."""
        set_tr = set(self.dataset.ids_train)
        set_val = set(self.dataset.ids_val)
        set_te = set(self.dataset.ids_test)

        self.assertEqual(len(set_tr.intersection(set_val)), 0, "Fuga entre train y val")
        self.assertEqual(len(set_tr.intersection(set_te)), 0, "Fuga entre train y test")
        self.assertEqual(len(set_val.intersection(set_te)), 0, "Fuga entre val y test")

        clases_tr = set(self.dataset.y_train)
        clases_val = set(self.dataset.y_val)
        clases_te = set(self.dataset.y_test)

        self.assertEqual(len(clases_tr), 40)
        self.assertEqual(len(clases_val), 40)
        self.assertEqual(len(clases_te), 40)

    def test_03_recarga_arbol_reproduce_predicciones_exactas(self):
        """Verifica que el modelo de Árbol recargado de disco predice exactamente igual."""
        ruta_arbol = Path("resultados/arbol_olivetti/modelo.joblib")
        if not ruta_arbol.exists():
            self.skipTest("Modelo arbol_olivetti no generado aún.")

        modelo = ClasificadorArbol.cargar(ruta_arbol)
        preds = modelo.predecir(self.X_test_feat)

        # Leer predicciones de prueba guardadas
        ruta_csv = Path("resultados/arbol_olivetti/predicciones_test.csv")
        with open(ruta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            preds_guardadas = [int(row["clase_predicha"]) for row in reader]

        self.assertEqual(list(preds), preds_guardadas)

    def test_04_recarga_svm_reproduce_predicciones_exactas(self):
        """Verifica que el modelo SVM recargado reproduce exactamente las predicciones."""
        ruta_svm = Path("resultados/svm_olivetti/modelo.joblib")
        if not ruta_svm.exists():
            self.skipTest("Modelo svm_olivetti no generado aún.")

        modelo = ClasificadorSVM.cargar(ruta_svm)
        preds = modelo.predecir(self.X_test_feat)

        ruta_csv = Path("resultados/svm_olivetti/predicciones_test.csv")
        with open(ruta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            preds_guardadas = [int(row["clase_predicha"]) for row in reader]

        self.assertEqual(list(preds), preds_guardadas)

    def test_05_recarga_ann_reproduce_predicciones_exactas(self):
        """Verifica que el modelo ANN recargado reproduce exactamente las predicciones."""
        ruta_ann = Path("resultados/ann_olivetti/modelo.pt")
        if not ruta_ann.exists():
            self.skipTest("Modelo ann_olivetti no generado aún.")

        modelo = ClasificadorANN.cargar(ruta_ann)
        preds = modelo.predecir(self.X_test_feat)

        ruta_csv = Path("resultados/ann_olivetti/predicciones_test.csv")
        with open(ruta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            preds_guardadas = [int(row["clase_predicha"]) for row in reader]

        self.assertEqual(list(preds), preds_guardadas)

    def test_06_recarga_cnn_reproduce_predicciones_exactas(self):
        """Verifica que el modelo CNN recargado reproduce exactamente las predicciones."""
        ruta_cnn = Path("resultados/cnn_olivetti/modelo.pt")
        if not ruta_cnn.exists():
            self.skipTest("Modelo cnn_olivetti no generado aún.")

        modelo = ClasificadorCNN.cargar(ruta_cnn)
        preds = modelo.predecir(self.X_test_prep)

        ruta_csv = Path("resultados/cnn_olivetti/predicciones_test.csv")
        with open(ruta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            preds_guardadas = [int(row["clase_predicha"]) for row in reader]

        self.assertEqual(list(preds), preds_guardadas)


if __name__ == "__main__":
    unittest.main()
