"""
Pruebas unitarias para el clasificador KNN.
"""
import unittest
import numpy as np
from reconocimiento.clasificacion.knn import ClasificadorKNN


class TestClasificadorKNN(unittest.TestCase):

    def test_knn_entrenamiento_y_prediccion(self):
        """Verifica que el KNN clasifique correctamente casos separables."""
        # Conjunto de juguete: 2 clases con 4 muestras
        X_train = np.array([
            [0.1, 0.1],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.8, 0.9]
        ], dtype=np.float32)
        y_train = np.array([0, 0, 1, 1], dtype=np.int32)

        knn = ClasificadorKNN(k=1, metrica="euclidean")
        knn.entrenar(X_train, y_train)

        X_test = np.array([
            [0.15, 0.12],
            [0.85, 0.88]
        ], dtype=np.float32)
        y_pred = knn.predecir(X_test)

        self.assertEqual(y_pred.tolist(), [0, 1])

    def test_probabilidades_a_posteriori(self):
        """Verifica que las probabilidades sumen 1.0 (kNN.md pág 5)."""
        X_train = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.1, 1.0]
        ], dtype=np.float32)
        y_train = np.array([0, 0, 0, 1, 1], dtype=np.int32)

        knn = ClasificadorKNN(k=3, metrica="euclidean")
        knn.entrenar(X_train, y_train)

        X_test = np.array([[0.05, 0.05]], dtype=np.float32)
        prob = knn.predecir_probabilidades(X_test)
        self.assertEqual(prob.shape, (1, 2))
        self.assertAlmostEqual(prob.sum(), 1.0, places=5)
        self.assertGreater(prob[0, 0], prob[0, 1])


if __name__ == "__main__":
    unittest.main()
