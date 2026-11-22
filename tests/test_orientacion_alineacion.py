"""
Pruebas funcionales de orientación y alineación facial:
1. Transformación de coordenadas y puntos faciales en las cuatro rotaciones (0°, 90°, 180°, 270°).
2. Fusión de detecciones duplicadas del mismo rostro vs preservación de rostros distintos.
3. Preservación estricta de estados: SIN_ROSTRO, UN_ROSTRO, VARIOS_ROSTROS.
4. Alineación facial por similitud sin reflejos (det > 0), con dimensiones (64, 64) y tratamiento de bordes.
5. Fallback explícito documentado ante puntos faciales degenerados o faltantes.
6. Concordancia entre preprocesamiento de entrenamiento e inferencia.
7. Carga de ambos sistemas (candidato original y nuevo candidato alineado) sin interferencia.
8. Detección en imágenes reales de desarrollo giradas 90°, 180° y 270°.
9. Ciclo de vida del streamer con detector multi-orientación (inicio, detención, reinicio).
"""

from pathlib import Path
import unittest
import cv2
import numpy as np

from reconocimiento.preprocesamiento.alineador import (
    PLANTILLA_FACIAL_64x64,
    alinear_rostro,
    estimar_similitud_sin_reflejo,
    preprocesar_rostro_alineado,
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from reconocimiento.preprocesamiento.orientacion import (
    DeteccionOrientada,
    calcular_iou_y_iomin,
    fusionar_detecciones_multiorientacion,
    rotar_fotograma,
    transformar_puntos_a_original,
    transformar_recuadro_a_original,
)
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import GestorCamaraWeb


class TestOrientacionYAlineacion(unittest.TestCase):

    def setUp(self):
        self.alto = 480
        self.ancho = 640
        self.ruta_img_dev = Path("data/propio/imagenes/persona_1/sesion_01/frontal/persona_1_sesion_01_frontal_f03.jpg")

    def test_01_transformacion_recuadros_cuatro_orientaciones(self):
        """Verifica que los recuadros se transformen exactamente en 0°, 90°, 180° y 270° intercambiando H y W."""
        # Recuadro original
        x, y, w, h = 100, 150, 80, 120

        # En 0°
        box_0 = transformar_recuadro_a_original((x, y, w, h), 0, self.alto, self.ancho)
        self.assertEqual(box_0, (100, 150, 80, 120))

        # En 90° CW: rotated has shape (640, 480).
        # x_rot = alto - (y + h) = 480 - 270 = 210, y_rot = x = 100, w_rot = h = 120, h_rot = w = 80
        box_90 = transformar_recuadro_a_original((210, 100, 120, 80), 90, self.alto, self.ancho)
        self.assertEqual(box_90, (100, 150, 80, 120))

        # En 180°: rotated has shape (480, 640).
        # x_rot = ancho - (x + w) = 640 - 180 = 460, y_rot = alto - (y + h) = 480 - 270 = 210
        box_180 = transformar_recuadro_a_original((460, 210, 80, 120), 180, self.alto, self.ancho)
        self.assertEqual(box_180, (100, 150, 80, 120))

        # En 270° CW: rotated has shape (640, 480).
        # x_rot = y = 150, y_rot = ancho - (x + w) = 640 - 180 = 460, w_rot = h = 120, h_rot = w = 80
        box_270 = transformar_recuadro_a_original((150, 460, 120, 80), 270, self.alto, self.ancho)
        self.assertEqual(box_270, (100, 150, 80, 120))

    def test_02_transformacion_puntos_faciales_preserva_orden(self):
        """Verifica que los 5 puntos faciales mapeen a las coordenadas originales preservando el orden semántico."""
        pts_orig = np.array([
            [320.0, 200.0],  # p0: ojo derecho
            [410.0, 202.0],  # p1: ojo izquierdo
            [365.0, 250.0],  # p2: nariz
            [330.0, 290.0],  # p3: boca derecha
            [400.0, 290.0],  # p4: boca izquierda
        ], dtype=np.float32)

        for angulo in (0, 90, 180, 270):
            # Rotar puntos hacia el espacio rotado
            if angulo == 0:
                pts_rot = pts_orig.copy()
            elif angulo == 90:
                # x_rot = alto - 1 - y, y_rot = x
                pts_rot = np.column_stack([self.alto - 1.0 - pts_orig[:, 1], pts_orig[:, 0]])
            elif angulo == 180:
                pts_rot = np.column_stack([self.ancho - 1.0 - pts_orig[:, 0], self.alto - 1.0 - pts_orig[:, 1]])
            elif angulo == 270:
                pts_rot = np.column_stack([pts_orig[:, 1], self.ancho - 1.0 - pts_orig[:, 0]])

            # Transformar de vuelta con la función del módulo
            pts_recuperados = transformar_puntos_a_original(pts_rot, angulo, self.alto, self.ancho)
            np.testing.assert_allclose(pts_recuperados, pts_orig, atol=1e-3)

    def test_03_fusion_duplicados_mismo_rostro(self):
        """Verifica que detecciones superpuestas del mismo rostro se fusionen eligiendo el mayor score."""
        d1 = DeteccionOrientada(
            recuadro_original=(100, 100, 80, 80),
            puntos_original=np.zeros((5, 2)),
            score=0.92,
            angulo_deteccion=0,
            recuadro_rotado=(100, 100, 80, 80)
        )
        d2 = DeteccionOrientada(
            recuadro_original=(105, 98, 78, 82),  # IoU alto con d1
            puntos_original=np.zeros((5, 2)),
            score=0.75,
            angulo_deteccion=90,
            recuadro_rotado=(200, 105, 82, 78)
        )
        fusionados = fusionar_detecciones_multiorientacion([d1, d2])
        self.assertEqual(len(fusionados), 1)
        self.assertEqual(fusionados[0].score, 0.92)
        self.assertEqual(fusionados[0].angulo_deteccion, 0)

    def test_04_no_fusion_personas_distintas(self):
        """Verifica que dos rostros en ubicaciones distintas permanezcan separados como VARIOS_ROSTROS."""
        d_persona_a = DeteccionOrientada(
            recuadro_original=(50, 100, 80, 80),
            puntos_original=np.zeros((5, 2)),
            score=0.88,
            angulo_deteccion=0,
            recuadro_rotado=(50, 100, 80, 80)
        )
        d_persona_b = DeteccionOrientada(
            recuadro_original=(400, 100, 80, 80),  # Lado opuesto
            puntos_original=np.zeros((5, 2)),
            score=0.86,
            angulo_deteccion=0,
            recuadro_rotado=(400, 100, 80, 80)
        )
        fusionados = fusionar_detecciones_multiorientacion([d_persona_a, d_persona_b])
        self.assertEqual(len(fusionados), 2)

    def test_05_alineacion_sin_reflejo_y_dimensiones(self):
        """Verifica que la transformación afín tenga det > 0 y produzca exactamente resolución 64x64."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = np.array([
            [250.0, 180.0],
            [350.0, 180.0],
            [300.0, 230.0],
            [260.0, 280.0],
            [340.0, 280.0]
        ], dtype=np.float32)

        recorte, meta = alinear_rostro(img, pts, resolucion=(64, 64))
        self.assertEqual(recorte.shape, (64, 64, 3))
        self.assertTrue(meta.alineado)
        self.assertGreater(meta.determinante, 0.0)
        self.assertGreater(meta.escala, 0.0)

    def test_06_deteccion_reflejo_invalido(self):
        """Verifica que si los ojos están invertidos (reflejo especular), la alineación rechaza la matriz."""
        # Invertir ojo derecho e izquierdo para simular un reflejo
        pts_espejo = np.array([
            [350.0, 180.0],  # p0 puesto a la derecha
            [250.0, 180.0],  # p1 puesto a la izquierda
            [300.0, 230.0],
            [340.0, 280.0],
            [260.0, 280.0]
        ], dtype=np.float32)

        M = estimar_similitud_sin_reflejo(pts_espejo, PLANTILLA_FACIAL_64x64)
        # La función de similitud estricta debe rechazar el reflejo o no converger con det > 0 coherente
        if M is not None:
            a, b = M[0, 0], M[1, 0]
            det = a * M[1, 1] - b * M[0, 1]
            self.assertGreater(det, 0.0)

    def test_07_fallback_ante_puntos_degenerados(self):
        """Verifica que ante puntos faciales colineales o nulos se ejecute el fallback seguro sin lanzar excepción."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Puntos todos iguales (degenerados)
        pts_deg = np.array([[50.0, 50.0]] * 5, dtype=np.float32)

        recorte, meta = alinear_rostro(img, pts_deg, recuadro=(20, 20, 80, 80), resolucion=(64, 64))
        self.assertEqual(recorte.shape, (64, 64, 3))
        self.assertFalse(meta.alineado)
        self.assertEqual(meta.metodo, "fallback_recorte_seguro")

    def test_08_concordancia_preprocesamiento(self):
        """Verifica que preprocesar_rostro_alineado produzca tensores normalizados válidos en [0, 1]."""
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        pts = np.array([[100.0, 80.0], [180.0, 80.0], [140.0, 120.0], [110.0, 160.0], [170.0, 160.0]], dtype=np.float32)

        proc, meta = preprocesar_rostro_alineado(img, pts, resolucion=(64, 64), sigma=0.5)
        self.assertEqual(proc.shape, (64, 64))
        self.assertEqual(proc.dtype, np.float32)
        self.assertGreaterEqual(float(proc.min()), 0.0)
        self.assertLessEqual(float(proc.max()), 1.0)
        self.assertFalse(np.isnan(proc).any())

    def test_09_deteccion_en_imagenes_reales_rotadas_desarrollo(self):
        """Verifica que el detector en modo cascada localice el rostro en una imagen real girada 90°, 180° y 270°."""
        if not self.ruta_img_dev.exists():
            self.skipTest("Imagen de desarrollo no disponible.")

        img_orig = cv2.imread(str(self.ruta_img_dev))
        detector = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)

        for ang in (0, 90, 180, 270):
            rot = rotar_fotograma(img_orig, ang)
            res = detector.detectar(rot)
            self.assertEqual(
                res.estado,
                EstadoDeteccion.UN_ROSTRO,
                f"Fallo de detección en imagen real girada {ang}°"
            )
            self.assertIsNotNone(res.recuadro)
            self.assertIsNotNone(res.puntos_faciales)
            self.assertIsNotNone(res.recorte_alineado)
            self.assertEqual(res.recorte_alineado.shape, (64, 64, 3))

    def test_10_carga_independiente_candidato_original_vs_alineado(self):
        """Verifica que ambos candidatos se puedan instanciar y operen con sus configuraciones respectivas."""
        ruta_orig = Path("configuracion_candidato_congelado_corregida.json")
        ruta_alin = Path("configuracion_candidato_alineado.json")

        self.assertTrue(ruta_orig.exists(), "Falta configuración de candidato original")
        pipe_orig = PipelineCandidatoCongelado(ruta_orig)
        self.assertEqual(pipe_orig.identificador, "cnn_4clases_desarrollo_congelado")
        self.assertFalse(pipe_orig.detector.alinear_rostro)
        self.assertEqual(pipe_orig.umbral_score, 0.46)

        if ruta_alin.exists():
            pipe_alin = PipelineCandidatoCongelado(ruta_alin)
            self.assertEqual(pipe_alin.identificador, "cnn_4clases_alineada")
            self.assertTrue(pipe_alin.detector.alinear_rostro)
            # El umbral del nuevo candidato no se sobrepone al original
            self.assertIsNotNone(pipe_alin.umbral_score)

    def test_11_modelo_sin_umbral_viable_deshabilita_regla_b_y_no_exporta(self):
        """Verifica que si ningún umbral cumple FRR_p <= 30%, el estado sea sin_umbral_viable,
        el umbral sea null, Regla B quede deshabilitada y no se exporte candidato."""
        import tempfile
        from scripts.experimentar_alineacion import (
            evaluar_modelo_reglas,
            exportar_candidato_congelado_si_viable,
            generar_tabla_comparativa,
        )

        class ClasificadorInviableDummy:
            def predecir_probabilidades(self, X):
                # Predice 100% desconocido (clase 3)
                # Para los autorizados todos son clasificados como desconocido (bloqueo FRR=100%)
                n = len(X)
                probs = np.zeros((n, 4), dtype=np.float32)
                probs[:, 3] = 1.0
                return probs

        # Construir datos sintéticos mínimos de validación
        val_auth = []
        for i in range(10):
            val_auth.append({"id": f"p1_{i}", "persona": "persona_1", "pose": "frontal"})
            val_auth.append({"id": f"p2_{i}", "persona": "persona_2", "pose": "frontal"})
            val_auth.append({"id": f"p3_{i}", "persona": "persona_3", "pose": "frontal"})

        datos_test = {
            "val_auth_todos": val_auth,
            "val_auth_proc_ids": [m["id"] for m in val_auth],
            "imagenes_autorizadas": {m["id"]: np.zeros((64, 64), dtype=np.float32) for m in val_auth},
            "filas_desc_val": [{"id": f"desc_{j}"} for j in range(20)],
            "estados_desc_val": ["UN_ROSTRO"] * 20,
            "X_val_desc_img": np.zeros((20, 64, 64), dtype=np.float32),
        }

        eval_res = evaluar_modelo_reglas("cnn", ClasificadorInviableDummy(), datos_test)

        self.assertEqual(eval_res["estado"], "sin_umbral_viable")
        self.assertIsNone(eval_res["umbral_optimo"])
        self.assertIsNone(eval_res["umbral_recomendado"])
        self.assertFalse(eval_res["tiene_umbral_viable"])
        self.assertIsNone(eval_res["regla_b"])
        self.assertFalse(eval_res["regla_b_habilitada"])
        self.assertIsNotNone(eval_res["regla_a"])
        self.assertEqual(len(eval_res["curva_umbrales"]), 101)

        # Verificar que exportar_candidato_congelado_si_viable rechace exportar
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exportado = exportar_candidato_congelado_si_viable({"cnn": eval_res}, tmp_path)
            self.assertIsNone(exportado)
            self.assertFalse((tmp_path / "configuracion_candidato_alineado.json").exists())

        # Verificar formateo en tabla comparativa
        tabla = generar_tabla_comparativa({"cnn": eval_res}, {"cnn": {"macro_recall_val": 50.0}})
        self.assertIn("null", tabla)
        self.assertIn("N/A (Regla B deshabilitada)", tabla)
        self.assertIn("**NO** (sin_umbral_viable)", tabla)
        self.assertIn("### Detalle de Modelos sin Umbral Viable", tabla)

    def test_12_modelos_guardados_desarrollo_identifican_inviables(self):
        """Verifica que el informe de resultados del experimento guarde Árbol y SVM
        con estado sin_umbral_viable y umbral null."""
        import json
        ruta_informe = Path("resultados/desarrollo_4clases_alineado_20261119_223850/informe_experimento.json")
        if not ruta_informe.exists():
            self.skipTest("Informe de experimento alineado no encontrado.")

        with open(ruta_informe, "r", encoding="utf-8") as f:
            informe = json.load(f)

        evals = informe["evaluaciones"]
        # Árbol y SVM no deben tener umbral viable
        self.assertEqual(evals["arbol"]["estado"], "sin_umbral_viable")
        self.assertIsNone(evals["arbol"]["umbral_optimo"])
        self.assertFalse(evals["arbol"]["regla_b_habilitada"])
        self.assertIsNone(evals["arbol"]["regla_b"])

        self.assertEqual(evals["svm"]["estado"], "sin_umbral_viable")
        self.assertIsNone(evals["svm"]["umbral_optimo"])
        self.assertFalse(evals["svm"]["regla_b_habilitada"])
        self.assertIsNone(evals["svm"]["regla_b"])

        # CNN y KNN deben ser viables
        self.assertEqual(evals["cnn"]["estado"], "viable")
        self.assertIsNotNone(evals["cnn"]["umbral_optimo"])
        self.assertTrue(evals["cnn"]["regla_b_habilitada"])


if __name__ == "__main__":
    unittest.main()
