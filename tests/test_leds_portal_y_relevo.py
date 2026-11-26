"""
Pruebas del relevo UDP del Arduino y de la integración de LED en el portal de reconocimiento.
No requieren el Arduino conectado: se usa un puerto serie simulado.
"""
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.hardware.indicador_arduino import (  # noqa: E402
    ConexionRelevo,
    IndicadorLeds,
    ServidorRelevo,
)
from reconocimiento.portal_reconocimiento.servicio import ServicioPortalReconocimiento  # noqa: E402


class PuertoFalso:
    def __init__(self):
        self.enviado = []

    def write(self, datos):
        self.enviado.append(datos)

    def flush(self):
        pass


def esperar(condicion, limite=2.0):
    fin = time.time() + limite
    while time.time() < fin:
        if condicion():
            return True
        time.sleep(0.02)
    return condicion()


class TestRelevoArduino(unittest.TestCase):
    def setUp(self):
        self.arduino = PuertoFalso()
        self.servidor = ServidorRelevo(self.arduino, puerto=0)
        self.cliente = ConexionRelevo(puerto=self.servidor.puerto)

    def tearDown(self):
        self.cliente.close()
        self.servidor.detener()

    def test_el_cliente_detecta_al_servidor(self):
        self.assertTrue(self.cliente.verificar())

    def test_el_cliente_detecta_que_no_hay_servidor(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        puerto_libre = s.getsockname()[1]
        s.close()
        cliente = ConexionRelevo(puerto=puerto_libre)
        self.assertFalse(cliente.verificar())
        cliente.close()

    def test_las_ordenes_llegan_al_arduino(self):
        for orden in (b"1", b"2", b"3", b"0"):
            self.cliente.write(orden)
        self.assertTrue(esperar(lambda: len(self.arduino.enviado) == 4))
        self.assertEqual(self.arduino.enviado, [b"1", b"2", b"3", b"0"])

    def test_ordenes_invalidas_se_ignoran(self):
        self.cliente.write(b"9")
        self.cliente.write(b"abc")
        self.cliente.write(b"2")
        self.assertTrue(esperar(lambda: self.arduino.enviado == [b"2"]))

    def test_indicador_completo_a_traves_del_relevo(self):
        indicador = IndicadorLeds(self.cliente)
        indicador.actualizar("persona_3")
        self.assertTrue(esperar(lambda: b"3" in self.arduino.enviado))


class IndicadorFalso:
    def __init__(self):
        self.llamadas = []

    def actualizar(self, persona):
        self.llamadas.append(persona)

    def cerrar(self):
        pass


class TestPortalConLeds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.indicador = IndicadorFalso()
        cls.servicio = ServicioPortalReconocimiento(
            raiz_proyecto=RAIZ,
            ruta_registro_intentos=Path(cls.tmp.name) / "intentos.jsonl",
            indicador_leds=cls.indicador,
        )
        ruta = RAIZ / "data/propio/imagenes/persona_1/sesion_04/frontal/persona_1_sesion_04_frontal_f01.jpg"
        cls.foto = ruta.read_bytes() if ruta.exists() else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_persona_autorizada_enciende_su_led(self):
        if self.foto is None:
            self.skipTest("Imagen de prueba no disponible")
        self.indicador.llamadas.clear()
        res = self.servicio.procesar_fotograma_bytes(self.foto, "cnn_estandar_portal_061")
        self.assertEqual(res["decision_puerta"], "ABIERTA")
        self.assertEqual(self.indicador.llamadas, [res["persona_identificada"]])
        self.assertEqual(self.indicador.llamadas[0], "persona_1")

    def test_fotograma_sin_rostro_no_enciende_ningun_led(self):
        self.indicador.llamadas.clear()
        negro = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))[1].tobytes()
        res = self.servicio.procesar_fotograma_bytes(negro, "cnn_estandar_portal_061")
        self.assertEqual(res["decision_puerta"], "CERRADA")
        self.assertEqual(self.indicador.llamadas, [None])


if __name__ == "__main__":
    unittest.main()
