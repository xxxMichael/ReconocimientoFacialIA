from pathlib import Path
import sys
import unittest
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import FotogramaBuffer
from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion


class TestInspectorPipelineWeb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg_cand = RAIZ / "configuracion_candidato_alineado.json"
        cls.app = crear_app(indice_inicial=-1, ruta_config_candidato=cfg_cand if cfg_cand.exists() else None)
        cls.client = cls.app.test_client()

    def test_01_ui_contiene_seccion_preprocesamiento_y_rasgos(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")
        self.assertIn("tab-nav-pipeline", html)
        self.assertIn("pipeline-layout", html)
        self.assertIn("img-pipe-deteccion", html)
        self.assertIn("img-pipe-recorte-usado", html)
        self.assertIn("img-pipe-cnn-b1", html)
        self.assertIn("img-pipe-cnn-b3", html)

    def test_02_listado_dataset_excluye_prueba_reservada(self):
        resp = self.client.get("/api/pipeline/dataset_imagenes")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        muestras = data["muestras"]
        self.assertGreater(len(muestras), 0)
        for m in muestras:
            self.assertNotIn("prueba_reservada", m.get("etiqueta", "").lower())

    def test_03_inspeccionar_dataset_retorna_cuatro_fases_y_rasgos_cnn(self):
        resp_list = self.client.get("/api/pipeline/dataset_imagenes")
        muestras = resp_list.get_json()["muestras"]
        ruta_muestra = muestras[0]["ruta"]

        resp = self.client.post("/api/pipeline/inspeccionar_dataset", json={"ruta": ruta_muestra})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

        f1 = data["fase_1_deteccion"]
        self.assertTrue(len(f1["imagen_deteccion_b64"]) > 100)

        if f1["estado"] == "UN_ROSTRO":
            f2 = data["fase_2_preprocesamiento"]
            self.assertTrue(f2["usa_alineacion"])
            self.assertTrue(len(f2["recorte_usado_b64"]) > 50)
            self.assertTrue(len(f2["gris_64_b64"]) > 50)
            self.assertTrue(len(f2["gaussiano_64_b64"]) > 50)
            self.assertTrue(len(f2["diff_gaussiano_b64"]) > 50)

            f3 = data["fase_3_rasgos"]
            self.assertEqual(f3["tipo_rasgos"], "cnn")
            self.assertTrue(len(f3["cnn_bloque1_b64"]) > 50)
            self.assertTrue(len(f3["cnn_bloque2_b64"]) > 50)
            self.assertTrue(len(f3["cnn_bloque3_b64"]) > 50)
            self.assertEqual(f3["cuadricula_4x4_b64"], "")

            f4 = data["fase_4_clasificacion"]
            self.assertIn(f4["decision_puerta"], ("ABIERTA", "CERRADA"))
            self.assertIsNotNone(f4["probabilidades"])

    def test_04_inspeccionar_camara_maneja_sin_fotograma_y_con_fotograma(self):
        gestor = self.app.config["GESTOR_CAMARA"]
        # Sin fotograma
        with gestor.lock_buffer:
            gestor.buffer.clear()
        resp_vacio = self.client.post("/api/pipeline/inspeccionar_camara")
        self.assertEqual(resp_vacio.status_code, 400)

        # Inyectar un fotograma sintético en el buffer para probar el flujo cámara
        frame_fake = np.zeros((240, 320, 3), dtype=np.uint8)
        with gestor.lock_buffer:
            gestor.buffer.append(FotogramaBuffer(
                id_fotograma=1, timestamp=0.0, frame_raw_bgr=frame_fake,
                estado_deteccion=EstadoDeteccion.SIN_ROSTRO, recuadro=None,
                total_rostros=0, frame_display_jpeg=b"", ancho=320, alto=240
            ))
        resp_ok = self.client.post("/api/pipeline/inspeccionar_camara")
        self.assertEqual(resp_ok.status_code, 200)
        self.assertEqual(resp_ok.get_json()["status"], "ok")

    def test_05_revision_y_correccion_fotos_desc(self):
        ruta_rev = RAIZ / "data" / "universidad_desc" / "revision_universidad.json"
        contenido_previo = ruta_rev.read_text(encoding="utf-8") if ruta_rev.exists() else None

        try:
            resp_list = self.client.get("/api/revision/capturas?entorno=fotos_desc")
            self.assertEqual(resp_list.status_code, 200)
            data_list = resp_list.get_json()
            self.assertEqual(data_list["status"], "ok")
            self.assertEqual(data_list["entorno"], "fotos_desc")
            self.assertGreaterEqual(data_list["total"], 33)

            primer_id = data_list["capturas"][0]["identificador"]
            resp_det = self.client.get(f"/api/revision/detalle/{primer_id}?entorno=fotos_desc")
            self.assertEqual(resp_det.status_code, 200)
            self.assertTrue(len(resp_det.get_json()["imagen_base64"]) > 100)

            resp_dec = self.client.post("/api/revision/decidir", json={
                "identificador": primer_id,
                "decision": "corregida",
                "caja": [10, 10, 100, 100],
                "entorno": "fotos_desc",
            })
            self.assertEqual(resp_dec.status_code, 200)
            self.assertEqual(resp_dec.get_json()["decision"], "corregida")
        finally:
            if contenido_previo is not None:
                ruta_rev.write_text(contenido_previo, encoding="utf-8")
            elif ruta_rev.exists():
                ruta_rev.unlink()


if __name__ == "__main__":
    unittest.main()
