"""
Pruebas unitarias para el detector alternativo YuNet (Deep Learning) y su comparación.
"""
from pathlib import Path
import unittest
import cv2
import numpy as np

from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet


class TestDetectorYuNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = DetectorYuNet()
        cls.ruta_frontal = Path("data/propio/imagenes/persona_1/sesion_01/frontal/persona_1_sesion_01_frontal_f03.jpg")
        cls.ruta_perfil = Path("data/propio/imagenes/persona_1/sesion_02/izquierda/persona_1_sesion_02_izquierda_f06.jpg")

    def test_deteccion_frontal_yunet(self):
        """Verifica detección de rostro frontal con YuNet."""
        if not self.ruta_frontal.exists():
            self.skipTest("Imagen frontal no disponible.")
        img = cv2.imread(str(self.ruta_frontal))
        res = self.detector.detectar(img)
        self.assertEqual(res.estado, EstadoDeteccion.UN_ROSTRO)
        self.assertIsNotNone(res.recuadro)
        self.assertIsNotNone(res.rostro_recortado)
        self.assertEqual(res.rostro_recortado.shape[0], res.rostro_recortado.shape[1])

    def test_deteccion_giro_perfil_yunet(self):
        """Verifica detección de rostro girado/perfil con YuNet."""
        if not self.ruta_perfil.exists():
            self.skipTest("Imagen de perfil no disponible.")
        img = cv2.imread(str(self.ruta_perfil))
        res = self.detector.detectar(img)
        self.assertEqual(res.estado, EstadoDeteccion.UN_ROSTRO)
        self.assertIsNotNone(res.recuadro)
        x, y, w, h = res.recuadro
        self.assertGreater(w, 50)
        self.assertGreater(h, 50)

    def test_escenas_vacias_sin_falsos_positivos_yunet(self):
        """Verifica que imágenes sin rostros (fondo plano, ruido) devuelvan SIN_ROSTRO."""
        negra = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(self.detector.detectar(negra).estado, EstadoDeteccion.SIN_ROSTRO)

        blanca = np.full((480, 640, 3), 255, dtype=np.uint8)
        self.assertEqual(self.detector.detectar(blanca).estado, EstadoDeteccion.SIN_ROSTRO)

        ruido = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.assertEqual(self.detector.detectar(ruido).estado, EstadoDeteccion.SIN_ROSTRO)

    def test_dos_personas_devuelve_varios_rostros_yunet(self):
        """Verifica que una escena compuesta con dos sujetos devuelva VARIOS_ROSTROS."""
        if not self.ruta_frontal.exists() or not self.ruta_perfil.exists():
            self.skipTest("Imágenes de prueba no disponibles.")
        img_f = cv2.imread(str(self.ruta_frontal))
        img_p = cv2.imread(str(self.ruta_perfil))
        dos_personas = np.hstack([img_f, img_p])
        res = self.detector.detectar(dos_personas)
        self.assertEqual(res.estado, EstadoDeteccion.VARIOS_ROSTROS)
        self.assertGreaterEqual(res.total_rostros_detectados, 2)


if __name__ == "__main__":
    unittest.main()
