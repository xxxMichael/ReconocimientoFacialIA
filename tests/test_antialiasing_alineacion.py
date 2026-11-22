import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.preprocesamiento.alineador import alinear_rostro, preprocesar_rostro_alineado
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from reconocimiento.web import crear_app
from reconocimiento.web.inspector_pipeline import inspeccionar_fotograma_pipeline


class TestAntialiasingAlineacion(unittest.TestCase):
    def test_01_reduce_aliasing_en_patron_alta_frecuencia(self):
        H, W = 512, 512
        x = np.arange(W)
        patron = (((x // 2) % 2) * 255).astype(np.uint8)
        img_hf = cv2.cvtColor(np.tile(patron, (H, 1)), cv2.COLOR_GRAY2BGR)
        pts_hf = np.array([
            [172.0, 185.0], [338.0, 200.0], [250.0, 285.0], [182.0, 376.0], [312.0, 388.0]
        ], dtype=np.float32)

        rec_est, m_est = alinear_rostro(img_hf, pts_hf, resolucion=(64, 64), metodo_remuestreo="estandar")
        rec_aa, m_aa = alinear_rostro(img_hf, pts_hf, resolucion=(64, 64), metodo_remuestreo="supersampling_area")

        self.assertEqual(m_est.factor_sobremuestreo, 1)
        self.assertGreater(m_aa.factor_sobremuestreo, 1)

        std_est = float(np.std(rec_est[10:54, 10:54]))
        std_aa = float(np.std(rec_aa[10:54, 10:54]))
        self.assertLess(std_aa, std_est * 0.1)

    def test_02_no_sobremuestrea_cuando_se_amplia(self):
        img_pq = np.zeros((48, 48, 3), dtype=np.uint8)
        pts_pq = np.array([
            [16.0, 18.0], [32.0, 18.0], [24.0, 26.0], [18.0, 34.0], [30.0, 34.0]
        ], dtype=np.float32)
        _, m_aa = alinear_rostro(img_pq, pts_pq, resolucion=(64, 64), metodo_remuestreo="supersampling_area")
        self.assertGreaterEqual(m_aa.escala, 0.9)
        self.assertEqual(m_aa.factor_sobremuestreo, 1)

    def test_03_equivalencia_numerica_entrenamiento_vs_web_varias_escalas_y_rotaciones(self):
        ruta_img = RAIZ / "data/propio/imagenes/persona_1/sesion_04/frontal/persona_1_sesion_04_frontal_f01.jpg"
        ruta_cfg_aa = RAIZ / "configuracion_candidato_antialiasing_052.json"
        if not ruta_img.exists() or not ruta_cfg_aa.exists():
            self.skipTest("Imagen o configuración antialiasing no disponible")

        img_base = cv2.imread(str(ruta_img))
        pipe_aa = PipelineCandidatoCongelado(ruta_cfg_aa, raiz_proyecto=RAIZ)
        self.assertEqual(pipe_aa.metodo_remuestreo, "supersampling_area")

        # Cubrir múltiples factores de reducción (1.5x, 1.0x, 0.5x) y orientaciones (0°, 90°, 180°)
        variantes = [
            ("original", img_base),
            ("escala_1_5x", cv2.resize(img_base, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)),
            ("escala_0_5x", cv2.resize(img_base, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)),
            ("rot_90", cv2.rotate(img_base, cv2.ROTATE_90_CLOCKWISE)),
            ("rot_180", cv2.rotate(img_base, cv2.ROTATE_180)),
        ]

        for etiqueta, frame in variantes:
            with self.subTest(variante=etiqueta):
                res_det = pipe_aa.detector.detectar(frame, es_pre_recortada=False)
                self.assertIsNotNone(res_det.recorte_alineado, f"No detectó rostro en {etiqueta}")
                self.assertIsNotNone(res_det.puntos_faciales)

                # Tensor producido por la ruta de inferencia web
                tensor_web = pipe_aa.preprocesar_recorte(res_det.recorte_alineado)

                # Tensor producido por la función de entrenamiento sobre el mismo fotograma
                tensor_train, meta_train = preprocesar_rostro_alineado(
                    frame,
                    res_det.puntos_faciales,
                    res_det.recuadro,
                    resolucion=(64, 64),
                    sigma=pipe_aa.sigma_gaussiano,
                    metodo_remuestreo="supersampling_area",
                )

                self.assertTrue(meta_train.alineado)
                self.assertEqual(tensor_web.shape, (64, 64))
                self.assertEqual(tensor_train.shape, (64, 64))
                self.assertEqual(tensor_web.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(tensor_web)))
                self.assertGreaterEqual(float(np.min(tensor_web)), 0.0)
                self.assertLessEqual(float(np.max(tensor_web)), 1.0)
                # Equivalencia numérica exacta (no solo dimensiones)
                self.assertTrue(
                    np.allclose(tensor_web, tensor_train, atol=1e-6),
                    f"Diferencia numérica entrenamiento vs web en {etiqueta}: max_diff={np.max(np.abs(tensor_web - tensor_train))}",
                )

    def test_04_equivalencia_estandar_y_fallback_sin_landmarks(self):
        ruta_img = RAIZ / "data/propio/imagenes/persona_1/sesion_04/frontal/persona_1_sesion_04_frontal_f01.jpg"
        ruta_cfg_std = RAIZ / "configuracion_candidato_estandar_portal_061.json"
        if not ruta_img.exists() or not ruta_cfg_std.exists():
            self.skipTest("Imagen o configuración estándar no disponible")

        img_bgr = cv2.imread(str(ruta_img))
        pipe_std = PipelineCandidatoCongelado(ruta_cfg_std, raiz_proyecto=RAIZ)
        self.assertEqual(pipe_std.metodo_remuestreo, "estandar")

        res_det = pipe_std.detector.detectar(img_bgr, es_pre_recortada=False)
        self.assertIsNotNone(res_det.recorte_alineado)
        tensor_web_std = pipe_std.preprocesar_recorte(res_det.recorte_alineado)
        tensor_train_std, _ = preprocesar_rostro_alineado(
            img_bgr,
            res_det.puntos_faciales,
            res_det.recuadro,
            resolucion=(64, 64),
            sigma=pipe_std.sigma_gaussiano,
            metodo_remuestreo="estandar",
        )
        self.assertTrue(np.allclose(tensor_web_std, tensor_train_std, atol=1e-6))

        # Fallback sin landmarks debe ser idéntico entre estandar y supersampling_area
        rec_box_est, m_box_est = preprocesar_rostro_alineado(
            img_bgr, None, res_det.recuadro, resolucion=(64, 64), sigma=0.5, metodo_remuestreo="estandar"
        )
        rec_box_aa, m_box_aa = preprocesar_rostro_alineado(
            img_bgr, None, res_det.recuadro, resolucion=(64, 64), sigma=0.5, metodo_remuestreo="supersampling_area"
        )
        self.assertFalse(m_box_est.alineado)
        self.assertFalse(m_box_aa.alineado)
        self.assertTrue(np.allclose(rec_box_est, rec_box_aa, atol=1e-6))

    def test_05_compatibilidad_configuracion_anterior_e_inspector(self):
        cfg_ant = RAIZ / "configuracion_candidato_alineado.json"
        ruta_cfg_aa = RAIZ / "configuracion_candidato_antialiasing_052.json"
        ruta_img = RAIZ / "data/propio/imagenes/persona_1/sesion_04/frontal/persona_1_sesion_04_frontal_f01.jpg"
        if not cfg_ant.exists() or not ruta_cfg_aa.exists() or not ruta_img.exists():
            self.skipTest("Archivos de prueba no encontrados")

        pipe_ant = PipelineCandidatoCongelado(cfg_ant, raiz_proyecto=RAIZ)
        self.assertEqual(pipe_ant.metodo_remuestreo, "estandar")
        self.assertEqual(pipe_ant.detector.metodo_remuestreo, "estandar")

        # Verificar que el inspector visual muestra las estadísticas del tensor real que recibe el clasificador
        img_bgr = cv2.imread(str(ruta_img))
        pipe_aa = PipelineCandidatoCongelado(ruta_cfg_aa, raiz_proyecto=RAIZ)

        class _GestorStub:
            def __init__(self, pipeline):
                self.pipeline_candidato = pipeline
                self.detector = pipeline.detector
                self.modelo = pipeline.clasificador
                self.modelo_habilitado = True
                self.nombre_modelo_activo = pipeline.nombre

        res_insp = inspeccionar_fotograma_pipeline(img_bgr, _GestorStub(pipe_aa), origen_etiqueta="test")
        self.assertEqual(res_insp["status"], "ok")
        stats_insp = res_insp["fase_2_preprocesamiento"]["estadisticas_tensor"]

        res_det = pipe_aa.detector.detectar(img_bgr, es_pre_recortada=False)
        tensor_real = pipe_aa.preprocesar_recorte(res_det.recorte_alineado)
        self.assertAlmostEqual(stats_insp["media"], round(float(np.mean(tensor_real)), 4), places=4)
        self.assertAlmostEqual(stats_insp["desv_std"], round(float(np.std(tensor_real)), 4), places=4)
        self.assertAlmostEqual(stats_insp["min"], round(float(np.min(tensor_real)), 4), places=4)
        self.assertAlmostEqual(stats_insp["max"], round(float(np.max(tensor_real)), 4), places=4)

    def test_06_web_cambia_entre_antialiasing_052_y_estandar_061(self):
        ruta_cfg_aa = RAIZ / "configuracion_candidato_antialiasing_052.json"
        ruta_cfg_std = RAIZ / "configuracion_candidato_estandar_portal_061.json"
        if not ruta_cfg_aa.exists() or not ruta_cfg_std.exists():
            self.skipTest("Configuraciones congeladas no encontradas")

        app = crear_app(indice_inicial=-1, ruta_config_candidato=ruta_cfg_aa)
        client = app.test_client()
        gestor = app.config["GESTOR_CAMARA"]

        self.assertIsNotNone(gestor.pipeline_candidato)
        self.assertEqual(gestor.pipeline_candidato.metodo_remuestreo, "supersampling_area")
        self.assertAlmostEqual(gestor.umbral_confianza, 0.52, places=2)

        resp_std = client.post("/api/reconocimiento/configurar", json={"modelo": "cnn_estandar_portal_061"})
        self.assertEqual(resp_std.status_code, 200)
        data_std = resp_std.get_json()
        self.assertEqual(data_std["status"], "ok")
        self.assertEqual(gestor.pipeline_candidato.metodo_remuestreo, "estandar")
        self.assertEqual(gestor.detector.metodo_remuestreo, "estandar")
        self.assertAlmostEqual(gestor.umbral_confianza, 0.61, places=2)
        self.assertEqual(
            data_std["sha256"],
            "24d3a1ed14cd8750863ff5bf464185c22eaef02bf4313044b64747929bd636f9",
        )

        resp_aa = client.post("/api/reconocimiento/configurar", json={"modelo": "cnn_antialiasing_052"})
        self.assertEqual(resp_aa.status_code, 200)
        data_aa = resp_aa.get_json()
        self.assertEqual(data_aa["status"], "ok")
        self.assertEqual(gestor.pipeline_candidato.metodo_remuestreo, "supersampling_area")
        self.assertEqual(gestor.detector.metodo_remuestreo, "supersampling_area")
        self.assertAlmostEqual(gestor.umbral_confianza, 0.52, places=2)
        self.assertEqual(
            data_aa["sha256"],
            "581b805b649be08577e8157fe26fc30d8d8bfe90a3b4819379f42051bc613f7f",
        )


if __name__ == "__main__":
    unittest.main()
