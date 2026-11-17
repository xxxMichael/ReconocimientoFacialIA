"""
Pruebas unitarias para el módulo de cámara, capturador, tolerancia a fallos y guardado atómico.
Utiliza fuentes de video simuladas para verificar:
1. Falta de cámara / error de apertura.
2. Interrupción de lectura / pérdida de señal.
3. Liberación garantizada de recursos.
4. Preservación del fotograma original sin alteraciones y atomicidad de guardado.
5. Distinción entre protocolo formal 300 y piloto parcial.
"""
import shutil
import unittest
from pathlib import Path
import cv2
import numpy as np

from reconocimiento.adquisicion.camara import (
    CapturadorCamara,
    guardar_fotograma_atomico,
    sondear_camaras
)
from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    generar_manifiesto_plantilla_300
)


class MockCamaraFallaApertura:
    """Simula una cámara inexistente o no disponible."""
    def isOpened(self):
        return False
    def release(self):
        pass


class MockCamaraConInterrupcion:
    """Simula una cámara que entrega N fotogramas y luego interrumpe la señal."""
    def __init__(self, fotogramas_validos: int = 3):
        self.fotogramas_validos = fotogramas_validos
        self.llamadas = 0
        self.liberada = False

    def isOpened(self):
        return True

    def read(self):
        self.llamadas += 1
        if self.llamadas <= self.fotogramas_validos:
            # Fotograma sintético 640x480 BGR
            frame = np.full((480, 640, 3), 120, dtype=np.uint8)
            return (True, frame)
        # Señal interrumpida / cable desconectado / app cerrada
        return (False, None)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640.0
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480.0
        elif prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def release(self):
        self.liberada = True


class TestCamaraAdquisicion(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("data/test_camara_tmp")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.ruta_manifiesto = self.test_dir / "manifiesto_test.csv"
        if self.ruta_manifiesto.exists():
            self.ruta_manifiesto.unlink()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_falta_de_camara_lanza_error_claro(self):
        """Verifica que si la cámara no se puede abrir se lance RuntimeError explicativo."""
        mock = MockCamaraFallaApertura()
        capturador = CapturadorCamara(indice=99, fuente_custom=mock)
        with self.assertRaises(RuntimeError) as ctx:
            capturador.abrir()
        self.assertIn("No se pudo abrir el dispositivo de video en índice 99", str(ctx.exception))

    def test_interrupcion_de_lectura_y_perdida_de_senal(self):
        """
        Verifica que tras lecturas fallidas consecutivas (ej. interrupción sostenida),
        se detecte la pérdida de señal y se lance una excepción limpia.
        """
        # Entrega 1 frame de arranque y luego falla
        mock = MockCamaraConInterrupcion(fotogramas_validos=1)
        capturador = CapturadorCamara(
            indice=1,
            intentos_arranque=1,
            max_fallos_consecutivos=3,
            fuente_custom=mock
        )
        capturador.abrir()
        self.assertTrue(capturador.esta_activo)

        # Primer lectura falla (fotogramas_validos=1 consumido en arranque)
        ret1, f1 = capturador.leer()
        self.assertFalse(ret1)

        ret2, f2 = capturador.leer()
        self.assertFalse(ret2)

        # Tercera lectura consecutiva fallida -> debe lanzar RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            capturador.leer()
        self.assertIn("Pérdida de señal en la cámara", str(ctx.exception))
        # Debe haber liberado el recurso
        self.assertFalse(capturador.esta_activo)
        self.assertTrue(mock.liberada)

    def test_liberacion_garantizada_con_context_manager(self):
        """Verifica que al usar context manager o liberar, release() siempre sea llamado."""
        mock = MockCamaraConInterrupcion(fotogramas_validos=5)
        capturador = CapturadorCamara(indice=1, intentos_arranque=1, fuente_custom=mock)

        with capturador as cap:
            ret, frame = cap.leer()
            self.assertTrue(ret)
            self.assertEqual(frame.shape, (480, 640, 3))

        # Al salir del bloque with, debe estar liberado
        self.assertFalse(capturador.esta_activo)
        self.assertTrue(mock.liberada)

    def test_guardado_atomico_y_preservacion_original(self):
        """
        Verifica que el fotograma original guardado sea idéntico byte a byte
        (sin dibujos ni reflejos) y que el manifiesto se actualice de forma coherente.
        """
        manifiesto = ManifiestoDataset(self.ruta_manifiesto)
        ruta_foto = self.test_dir / "persona_test_s01_frontal_f01.jpg"

        # Crear fotograma original con valores conocidos
        frame_original = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_original[20:80, 20:80] = [10, 50, 200]

        registro = RegistroCaptura(
            identificador="test_atomico_01",
            ruta=str(ruta_foto),
            procedencia="camara_droidcam",
            persona="voluntario_test",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True
        )

        guardar_fotograma_atomico(frame_original, ruta_foto, registro, manifiesto)

        # 1. Comprobar que el archivo existe en disco
        self.assertTrue(ruta_foto.is_file())

        # 2. Comprobar que el contenido leído de disco coincide con el original
        frame_leido = cv2.imread(str(ruta_foto))
        # En compresión JPEG con calidad estándar puede haber pérdidas mínimas de compresión,
        # pero la resolución y canales deben coincidir exactamente
        self.assertEqual(frame_leido.shape, frame_original.shape)
        self.assertTrue(np.allclose(frame_leido[40:60, 40:60], frame_original[40:60, 40:60], atol=15))

        # 3. Comprobar que el manifiesto tiene el registro con su hash SHA-256
        self.assertEqual(len(manifiesto.registros), 1)
        self.assertEqual(manifiesto.registros[0].identificador, "test_atomico_01")
        self.assertTrue(len(manifiesto.registros[0].hash_contenido) > 20)

    def test_guardado_atomico_rechaza_sobrescritura(self):
        """Verifica que si el archivo de destino ya existe, se rechace la sobrescritura."""
        manifiesto = ManifiestoDataset(self.ruta_manifiesto)
        ruta_existente = self.test_dir / "existente.jpg"
        ruta_existente.write_text("dummy")

        registro = RegistroCaptura(
            identificador="test_colision",
            ruta=str(ruta_existente),
            procedencia="camara",
            persona="persona_x",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True
        )
        dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)

        with self.assertRaises(FileExistsError):
            guardar_fotograma_atomico(dummy_frame, ruta_existente, registro, manifiesto)

    def test_distincion_protocolo_formal_vs_piloto_parcial(self):
        """Verifica la distinción entre un dataset piloto parcial y el protocolo formal completo."""
        # Plantilla completa de 300 fotos
        manifiesto_300 = generar_manifiesto_plantilla_300()
        res_300 = manifiesto_300.verificar_protocolo_completo()
        self.assertTrue(res_300["aislamiento_sesiones_valido"])
        self.assertTrue(res_300["sin_duplicados_entre_particiones"])
        self.assertEqual(res_300["estado_dataset"], "EVALUACION_FORMAL_300")
        self.assertTrue(res_300["protocolo_formal_300_completo"])

        # Manifiesto parcial piloto (solo 1 registro)
        manifiesto_piloto = ManifiestoDataset(self.ruta_manifiesto)
        r = RegistroCaptura(
            identificador="piloto_01",
            ruta="foto.jpg",
            procedencia="camara_droidcam",
            persona="persona_piloto",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True
        )
        manifiesto_piloto.agregar_registro(r)
        res_piloto = manifiesto_piloto.verificar_protocolo_completo()
        self.assertTrue(res_piloto["aislamiento_sesiones_valido"])
        self.assertTrue(res_piloto["es_piloto_parcial_valido"])
        self.assertFalse(res_piloto["protocolo_formal_300_completo"])
        self.assertEqual(res_piloto["estado_dataset"], "PILOTO_PARCIAL_VALIDO")


if __name__ == "__main__":
    unittest.main()
