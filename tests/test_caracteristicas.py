"""
Pruebas unitarias para el extractor de histogramas por regiones.
"""
import unittest
import numpy as np
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales


class TestCaracteristicas(unittest.TestCase):

    def test_dimension_vector_y_nombres(self):
        """Verifica que la dimensión sea Gy * Gx * n_bins y los nombres sean coherentes."""
        extractor = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16)
        self.assertEqual(extractor.dimension_total, 256)
        self.assertEqual(len(extractor.nombres_caracteristicas), 256)
        self.assertEqual(extractor.nombres_caracteristicas[0], "reg_y0_x0_bin00")
        self.assertEqual(extractor.nombres_caracteristicas[-1], "reg_y3_x3_bin15")

    def test_extraccion_normalizada(self):
        """Verifica que los histogramas regionales sumen 1.0 por región."""
        extractor = ExtractorHistogramasRegionales(cuadricula=(2, 2), n_bins=8)
        img = np.random.rand(64, 64).astype(np.float32)
        rep = extractor.extraer(img)

        self.assertEqual(rep.dimension, 32)  # 2 * 2 * 8
        self.assertEqual(len(rep.vector_caracteristicas), 32)
        # Cada bloque de 8 bins debe sumar 1.0
        for i in range(4):
            bloque = rep.vector_caracteristicas[i * 8:(i + 1) * 8]
            self.assertAlmostEqual(bloque.sum(), 1.0, places=5)

    def test_extraccion_lote(self):
        """Verifica la extracción simultánea sobre un lote de imágenes."""
        extractor = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16)
        lote = np.random.rand(10, 64, 64).astype(np.float32)
        X_feat, nombres = extractor.extraer_lote(lote)
        self.assertEqual(X_feat.shape, (10, 256))
        self.assertEqual(len(nombres), 256)


if __name__ == "__main__":
    unittest.main()
