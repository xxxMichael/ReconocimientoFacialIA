"""
Pruebas del indicador de LED sobre Arduino, con un puerto serie simulado.
Verifican: LED distinto por persona, apagado con desconocido/puerta cerrada, retención breve
contra parpadeos, latido periódico, y tolerancia a fallos de comunicación.
"""
import unittest

from reconocimiento.hardware.indicador_arduino import IndicadorLeds


class PuertoFalso:
    def __init__(self, falla_en=None):
        self.enviado = []
        self.cerrado = False
        self.falla_en = falla_en

    def write(self, datos):
        if self.falla_en is not None and len(self.enviado) >= self.falla_en:
            raise OSError("cable desconectado")
        self.enviado.append(datos)

    def flush(self):
        pass

    def close(self):
        self.cerrado = True


class RelojFalso:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def avanzar(self, s):
        self.t += s


class TestIndicadorLeds(unittest.TestCase):
    def setUp(self):
        self.puerto = PuertoFalso()
        self.reloj = RelojFalso()
        self.ind = IndicadorLeds(self.puerto, mantener_s=1.0, reenvio_s=0.5, reloj=self.reloj)

    def test_cada_persona_enciende_su_led(self):
        for persona, orden in (("persona_1", b"1"), ("persona_2", b"2"), ("persona_3", b"3")):
            self.ind.actualizar(persona)
            self.assertEqual(self.puerto.enviado[-1], orden)
            self.reloj.avanzar(2.0)
            self.ind.actualizar(None)
            self.assertEqual(self.puerto.enviado[-1], b"0")

    def test_sin_persona_nunca_enciende(self):
        for _ in range(5):
            self.ind.actualizar(None)
            self.reloj.avanzar(0.3)
        self.assertTrue(all(o == b"0" for o in self.puerto.enviado))

    def test_persona_desconocida_o_invalida_no_enciende(self):
        self.ind.actualizar("DESCONOCIDO")
        self.ind.actualizar("persona_9")
        self.assertTrue(all(o == b"0" for o in self.puerto.enviado))

    def test_retiene_el_led_ante_un_fotograma_fallido(self):
        self.ind.actualizar("persona_3")
        self.reloj.avanzar(0.2)
        self.ind.actualizar(None)  # un solo fotograma sin reconocer
        self.assertEqual(self.ind._orden_actual, b"3")
        self.reloj.avanzar(0.2)
        self.ind.actualizar("persona_3")
        self.assertEqual(self.ind._orden_actual, b"3")

    def test_apaga_pasado_el_tiempo_de_retencion(self):
        self.ind.actualizar("persona_1")
        self.reloj.avanzar(1.5)
        self.ind.actualizar(None)
        self.assertEqual(self.puerto.enviado[-1], b"0")

    def test_cambio_de_persona_es_inmediato(self):
        self.ind.actualizar("persona_1")
        self.reloj.avanzar(0.1)
        self.ind.actualizar("persona_2")
        self.assertEqual(self.puerto.enviado[-1], b"2")

    def test_no_envia_en_cada_fotograma_pero_mantiene_latido(self):
        self.ind.actualizar("persona_3")
        n0 = len(self.puerto.enviado)
        for _ in range(3):  # fotogramas a ~14 FPS, dentro de los 0,5 s
            self.reloj.avanzar(0.07)
            self.ind.actualizar("persona_3")
        self.assertEqual(len(self.puerto.enviado), n0)
        self.reloj.avanzar(0.4)
        self.ind.actualizar("persona_3")  # supera el intervalo de latido
        self.assertEqual(len(self.puerto.enviado), n0 + 1)
        self.assertEqual(self.puerto.enviado[-1], b"3")

    def test_apagar_es_inmediato(self):
        self.ind.actualizar("persona_2")
        self.ind.apagar()
        self.assertEqual(self.puerto.enviado[-1], b"0")

    def test_cerrar_apaga_y_libera_el_puerto(self):
        self.ind.actualizar("persona_1")
        self.ind.cerrar()
        self.assertEqual(self.puerto.enviado[-1], b"0")
        self.assertTrue(self.puerto.cerrado)
        self.assertFalse(self.ind.activo)

    def test_fallo_de_comunicacion_no_propaga_excepcion(self):
        puerto = PuertoFalso(falla_en=1)
        ind = IndicadorLeds(puerto, reloj=self.reloj)
        ind.actualizar("persona_1")  # primer envío correcto
        self.reloj.avanzar(2.0)
        ind.actualizar("persona_2")  # falla: no debe lanzar excepción
        self.assertFalse(ind.activo)
        ind.actualizar("persona_3")  # ya desactivado: se ignora sin error
        ind.apagar()
        ind.cerrar()


if __name__ == "__main__":
    unittest.main()
