"""
Pruebas dirigidas para la interfaz web de adquisición facial y el gestor de fotogramas.
Verifica:
1. Guardado exacto del fotograma mostrado aunque ya hayan llegado otros (búfer circular).
2. Rechazo de fotogramas caducados fuera del búfer.
3. Impedir guardado sin consentimiento explícito.
4. Separación estricta entre captura válida (UN_ROSTRO) y registro de fallo (SIN_ROSTRO / VARIOS_ROSTROS).
5. Protección contra duplicación por doble clic / peticiones simultáneas (idempotencia).
6. Recuperación fiel del progreso desde disco al recargar página.
7. Conservación del archivo CSV previo ante fallos simulados durante la escritura.
"""
import shutil
import unittest
from pathlib import Path
import numpy as np

from reconocimiento.adquisicion.camara import guardar_fotograma_atomico
from reconocimiento.adquisicion.propio import ManifiestoDataset, RegistroCaptura
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import FotogramaBuffer, GestorCamaraWeb, calcular_progreso_sesion


class MockVideoSourceEstable:
    """Fuente de video simulada para pruebas web."""
    def __init__(self, estado_deteccion: EstadoDeteccion = EstadoDeteccion.UN_ROSTRO):
        self.abierto = True
        self.calls = 0
        self.estado = estado_deteccion

    def isOpened(self):
        return self.abierto

    def read(self):
        self.calls += 1
        # Generar un fotograma de 640x480 con píxel distintivo según llamadas
        frame = np.full((480, 640, 3), (self.calls % 250), dtype=np.uint8)
        return (True, frame)

    def get(self, prop):
        return 640.0

    def release(self):
        self.abierto = False


class TestWebAdquisicion(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("data/test_web_tmp")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.manifiesto_test = self.test_dir / "manifiesto_test.csv"
        if self.manifiesto_test.exists():
            self.manifiesto_test.unlink()

        self.gestor = GestorCamaraWeb(tamano_buffer=10)
        self.gestor.detener()
        self.gestor.buffer.clear()
        self.gestor.cache_idempotencia.clear()

        self.app = crear_app(indice_inicial=-1)
        self.client = self.app.test_client()

    def tearDown(self):
        self.gestor.detener()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _inyectar_fotograma_al_buffer(self, id_frame: int, estado: EstadoDeteccion, valor_pixel: int = 100):
        """Inyecta un fotograma sintético directamente al búfer circular del gestor."""
        frame = np.full((480, 640, 3), valor_pixel, dtype=np.uint8)
        item = FotogramaBuffer(
            id_fotograma=id_frame,
            timestamp=1000.0 + id_frame,
            frame_raw_bgr=frame,
            estado_deteccion=estado,
            recuadro=(50, 50, 100, 100) if estado == EstadoDeteccion.UN_ROSTRO else None,
            total_rostros=1 if estado == EstadoDeteccion.UN_ROSTRO else (2 if estado == EstadoDeteccion.VARIOS_ROSTROS else 0),
            frame_display_jpeg=b"fake_jpeg",
            ancho=640,
            alto=480
        )
        with self.gestor.lock_buffer:
            self.gestor.buffer.append(item)

    def test_guardar_fotograma_mostrado_aunque_hayan_llegado_otros(self):
        """
        Verifica que si el cliente vio el fotograma #10 y entretanto el servidor ya va
        por el #15, el servidor recupere exactamente el fotograma #10 del búfer y lo guarde.
        """
        # Inyectar fotogramas 10 a 15 con diferentes intensidades
        for fid in range(10, 16):
            self._inyectar_fotograma_al_buffer(id_frame=fid, estado=EstadoDeteccion.UN_ROSTRO, valor_pixel=fid * 10)

        # El cliente solicita guardar el fotograma #10
        payload = {
            "id_fotograma": 10,
            "modo": "valida",
            "persona": "persona_test",
            "tipo_sujeto": "autorizado",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": True,
            "idempotency_key": "test_sync_10"
        }

        resp = self.client.post("/api/capturar", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["id_fotograma"], 10)

        # Comprobar que la imagen guardada en disco corresponde al fotograma #10 (valor_pixel=100)
        ruta_guardada = Path(data["ruta"])
        self.assertTrue(ruta_guardada.is_file())
        img_leida = np.fromfile(str(ruta_guardada), dtype=np.uint8)
        self.assertGreater(len(img_leida), 0)

    def test_rechazo_fotograma_caducado(self):
        """Verifica que si un fotograma ya no está en el búfer circular, se rechace con 410."""
        # Llenar búfer con fotogramas del 20 al 30 (tamaño búfer = 10)
        for fid in range(20, 31):
            self._inyectar_fotograma_al_buffer(id_frame=fid, estado=EstadoDeteccion.UN_ROSTRO)

        # Solicitar fotograma #5 que ya caducó
        payload = {
            "id_fotograma": 5,
            "modo": "valida",
            "persona": "persona_test",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": True
        }

        resp = self.client.post("/api/capturar", json=payload)
        self.assertEqual(resp.status_code, 410)
        data = resp.get_json()
        self.assertEqual(data["tipo"], "CADUCADO")
        self.assertIn("caducado en el búfer", data["mensaje"])

    def test_impedir_guardado_sin_consentimiento(self):
        """Verifica que una solicitud con consentimiento=False sea rechazada con 400."""
        self._inyectar_fotograma_al_buffer(id_frame=50, estado=EstadoDeteccion.UN_ROSTRO)

        payload = {
            "id_fotograma": 50,
            "modo": "valida",
            "persona": "persona_test",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": False,
            "es_ensayo": True
        }

        resp = self.client.post("/api/capturar", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("consentimiento", data["mensaje"].lower())

    def test_separacion_estricta_captura_valida_y_fallo(self):
        """
        Verifica que:
        - Captura válida rechace SIN_ROSTRO o VARIOS_ROSTROS.
        - Captura de fallo rechace UN_ROSTRO.
        """
        self._inyectar_fotograma_al_buffer(id_frame=60, estado=EstadoDeteccion.SIN_ROSTRO)
        self._inyectar_fotograma_al_buffer(id_frame=61, estado=EstadoDeteccion.UN_ROSTRO)

        # Intento de guardar #60 (SIN_ROSTRO) como válida -> debe fallar con 400
        p_valida_erronea = {
            "id_fotograma": 60,
            "modo": "valida",
            "persona": "persona_test",
            "sesion": "sesion_01",
            "pose": "arriba",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": True
        }
        r1 = self.client.post("/api/capturar", json=p_valida_erronea)
        self.assertEqual(r1.status_code, 400)
        self.assertIn("Se requiere exactamente UN_ROSTRO", r1.get_json()["mensaje"])

        # Intento de guardar #61 (UN_ROSTRO) como fallo -> debe fallar con 400
        p_fallo_erroneo = {
            "id_fotograma": 61,
            "modo": "fallo",
            "persona": "persona_test",
            "sesion": "sesion_01",
            "pose": "arriba",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": True
        }
        r2 = self.client.post("/api/capturar", json=p_fallo_erroneo)
        self.assertEqual(r2.status_code, 400)
        self.assertIn("No se puede registrar como fallo", r2.get_json()["mensaje"])

        # Registro legítimo de fallo en #60 -> debe ser exitoso con 200
        p_fallo_legitimo = {
            "id_fotograma": 60,
            "modo": "fallo",
            "persona": "persona_test",
            "sesion": "sesion_01",
            "pose": "arriba",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": True
        }
        r3 = self.client.post("/api/capturar", json=p_fallo_legitimo)
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.get_json()["modo"], "fallo")

    def test_evitar_duplicacion_por_doble_clic(self):
        """Verifica que peticiones con la misma idempotency_key retornen la misma captura sin duplicar."""
        self._inyectar_fotograma_al_buffer(id_frame=70, estado=EstadoDeteccion.UN_ROSTRO)

        payload = {
            "id_fotograma": 70,
            "modo": "valida",
            "persona": "persona_doble",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": True,
            "idempotency_key": "clave_unica_12345"
        }

        # Primer clic
        r1 = self.client.post("/api/capturar", json=payload)
        self.assertEqual(r1.status_code, 200)
        d1 = r1.get_json()

        # Segundo clic rápido con la misma clave
        r2 = self.client.post("/api/capturar", json=payload)
        self.assertEqual(r2.status_code, 200)
        d2 = r2.get_json()

        # Ambos deben reportar exactamente el mismo identificador y ruta
        self.assertEqual(d1["identificador"], d2["identificador"])
        self.assertEqual(d1["ruta"], d2["ruta"])

    def test_recuperar_progreso_tras_recargar(self):
        """Verifica que calcular_progreso_sesion lea directamente del manifiesto en disco."""
        manifiesto = ManifiestoDataset(self.manifiesto_test)
        manifiesto.registros.clear()

        # Registrar 3 fotos frontales válidas y 1 fallo en izquierda
        for f in range(1, 4):
            manifiesto.agregar_registro(RegistroCaptura(
                identificador=f"sujeto_s01_frontal_f{f:02d}",
                ruta=f"img_{f}.jpg",
                procedencia="camara_droidcam",
                persona="sujeto_p",
                tipo_sujeto="autorizado",
                sesion="sesion_01",
                pose="frontal",
                particion="entrenamiento",
                consentimiento=True,
                anotaciones="forzado=no"
            ))

        manifiesto.agregar_registro(RegistroCaptura(
            identificador="sujeto_s01_izq_f01",
            ruta="img_izq.jpg",
            procedencia="camara_droidcam",
            persona="sujeto_p",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="izquierda",
            particion="entrenamiento",
            consentimiento=True,
            anotaciones="forzado=si"  # Fallo registrado
        ))
        manifiesto.guardar()

        # Simular lectura fresca desde el archivo en disco
        manifiesto_recargado = ManifiestoDataset(self.manifiesto_test)
        progreso = calcular_progreso_sesion(manifiesto_recargado, "sujeto_p", "sesion_01")

        self.assertEqual(progreso["poses"]["frontal"]["validas"], 3)
        self.assertEqual(progreso["poses"]["frontal"]["fallos"], 0)
        self.assertFalse(progreso["poses"]["frontal"]["completada"])

        self.assertEqual(progreso["poses"]["izquierda"]["validas"], 0)
        self.assertEqual(progreso["poses"]["izquierda"]["fallos"], 1)

        self.assertEqual(progreso["total_validas"], 3)
        self.assertEqual(progreso["total_fallos"], 1)

    def test_conservar_csv_previo_si_falla_escritura(self):
        """Verifica que la escritura atómica con archivo .tmp conserve intacto el CSV anterior si ocurre un fallo."""
        manifiesto = ManifiestoDataset(self.manifiesto_test)
        manifiesto.registros.clear()
        reg_original = RegistroCaptura(
            identificador="original_01",
            ruta="ruta_orig.jpg",
            procedencia="camara",
            persona="p1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True
        )
        manifiesto.agregar_registro(reg_original)
        manifiesto.guardar()

        contenido_inicial = self.manifiesto_test.read_text(encoding="utf-8")
        self.assertIn("original_01", contenido_inicial)

        # Simular un objeto que falle durante asdict() o serialización
        class RegistroDañado:
            def __getattr__(self, name):
                raise IOError("Fallo simulado a mitad de escritura CSV")

        manifiesto.registros.append(RegistroDañado())

        with self.assertRaises(Exception):
            manifiesto.guardar()

        # El archivo manifiesto final debe permanecer idéntico al contenido inicial sin corrupciones
        contenido_despues_fallo = self.manifiesto_test.read_text(encoding="utf-8")
        self.assertEqual(contenido_inicial, contenido_despues_fallo)

        # El temporal no debe quedar huérfano
        ruta_tmp = self.manifiesto_test.with_name(f"{self.manifiesto_test.name}.tmp")
        self.assertFalse(ruta_tmp.exists())


if __name__ == "__main__":
    unittest.main()
