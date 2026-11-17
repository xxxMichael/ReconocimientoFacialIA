"""
Pruebas unitarias para el sistema de revisión de capturas, fotograma congelado,
y filtrado estricto para entrenamiento con cajas aprobadas/corregidas.
"""
from pathlib import Path
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from reconocimiento.adquisicion.propio import ManifiestoDataset, RegistroCaptura
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
from reconocimiento.web.app import crear_app


class TestRevisionCapturas(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.ruta_manifiesto = self.test_dir / "manifiesto_test.csv"
        self.manifiesto = ManifiestoDataset(self.ruta_manifiesto)

        # Crear imagen sintética con rostro simulado
        self.dir_imgs = self.test_dir / "imagenes" / "persona_1" / "sesion_01" / "frontal"
        self.dir_imgs.mkdir(parents=True, exist_ok=True)
        self.ruta_img = self.dir_imgs / "p1_s01_f01.jpg"

        # Imagen 640x480 con rectángulo dibujado
        img = np.full((480, 640, 3), 180, dtype=np.uint8)
        # Añadir estructura para simular rostro
        cv2.rectangle(img, (200, 100), (400, 300), (80, 80, 80), -1)
        cv2.imwrite(str(self.ruta_img), img)

        # Crear registros de prueba con distintos estados de revisión
        self.reg1 = RegistroCaptura(
            identificador="p1_s01_frontal_01",
            ruta=str(self.ruta_img),
            procedencia="camara_web",
            persona="persona_1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            consentimiento=True,
            anotaciones="captura_web;modo=valida;frame_id=101;estado=UN_ROSTRO"
        )
        self.manifiesto.agregar_registro(self.reg1)
        self.manifiesto.guardar()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_registro_metadatos_revision_iniciales(self):
        """Verifica que un registro antiguo sin decisión explícita inicie como 'pendiente'."""
        self.assertEqual(self.reg1.decision_humana, "pendiente")
        self.assertEqual(self.reg1.estado_historico_detector, "UN_ROSTRO")
        self.assertIsNone(self.reg1.caja_aprobada)

    def test_actualizar_decision_humana_y_caja(self):
        """Verifica aprobación y corrección conservando el estado histórico."""
        ok = self.manifiesto.actualizar_revision_captura(
            "p1_s01_frontal_01",
            decision="corregida",
            caja=(190, 95, 215, 215),
            origen_caja="manual"
        )
        self.assertTrue(ok)

        # Recargar desde disco
        m_recargado = ManifiestoDataset(self.ruta_manifiesto)
        reg = m_recargado.buscar_registro("p1_s01_frontal_01")
        self.assertIsNotNone(reg)
        self.assertEqual(reg.decision_humana, "corregida")
        self.assertEqual(reg.caja_aprobada, (190, 95, 215, 215))
        self.assertEqual(reg.origen_caja, "manual")
        # El estado histórico original permanece inalterado
        self.assertEqual(reg.estado_historico_detector, "UN_ROSTRO")

    def test_decision_excluir_e_inutilizable(self):
        """Verifica que una toma marcada como 'excluida' quede registrada."""
        ok = self.manifiesto.actualizar_revision_captura(
            "p1_s01_frontal_01",
            decision="excluida",
            caja=None
        )
        self.assertTrue(ok)
        reg = self.manifiesto.buscar_registro("p1_s01_frontal_01")
        self.assertEqual(reg.decision_humana, "excluida")

    def test_recalculada_para_fotos_antiguas_sin_caja(self):
        """Para fotos antiguas sin caja registrada, rotúlala 'recalculada'."""
        app = crear_app(indice_inicial=-1)
        client = app.test_client()

        # Inyectar manifiesto temporal en app
        resp = client.get("/api/revision/capturas?es_ensayo=false")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("capturas", data)


class TestFotogramaCongeladoYPipeline(unittest.TestCase):

    def test_endpoint_congelar_fotograma(self):
        """Verifica que /api/captura/congelar congele y genere imagen y recorte."""
        app = crear_app(indice_inicial=-1)
        client = app.test_client()

        gestor = app.config["GESTOR_CAMARA"]
        # Inyectar un fotograma simulado en el búfer
        dummy_frame = np.full((480, 640, 3), 150, dtype=np.uint8)
        # Añadir un recuadro
        recuadro = (150, 100, 200, 200)
        from reconocimiento.web.streamer import FotogramaBuffer
        _, jpeg_raw = cv2.imencode(".jpg", dummy_frame)
        fb = FotogramaBuffer(
            id_fotograma=999,
            timestamp=123.456,
            frame_raw_bgr=dummy_frame,
            frame_display_jpeg=jpeg_raw.tobytes(),
            estado_deteccion=EstadoDeteccion.UN_ROSTRO,
            recuadro=recuadro,
            total_rostros=1,
            ancho=640,
            alto=480
        )
        with gestor.lock_buffer:
            gestor.buffer.append(fb)

        resp = client.post("/api/captura/congelar", json={"id_fotograma": 999})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["id_fotograma"], 999)
        self.assertEqual(data["recuadro"], [150, 100, 200, 200])
        self.assertTrue(len(data["image_base64"]) > 50)
        self.assertTrue(len(data["recorte_base64"]) > 50)

    def test_decision_revision_api(self):
        """Verifica la API /api/revision/decidir."""
        app = crear_app(indice_inicial=-1)
        client = app.test_client()

        # Usar una captura real existente en data/propio/manifiesto.csv
        resp = client.post("/api/revision/decidir", json={
            "identificador": "persona_1_sesion_01_frontal_03",
            "decision": "aprobada",
            "caja": [238, 115, 216, 216],
            "es_ensayo": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["decision"], "aprobada")
        self.assertEqual(data["caja"], [238, 115, 216, 216])


if __name__ == "__main__":
    unittest.main()
