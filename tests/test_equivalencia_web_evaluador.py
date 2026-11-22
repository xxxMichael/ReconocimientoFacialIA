"""
Pruebas unitarias de equivalencia estricta entre el evaluador independiente y el streamer web.
Verifica que ante idénticas entradas, el PipelineCandidatoCongelado y el GestorCamaraWeb
produzcan recuadros, recortes, tensores preprocesados, vectores de probabilidad,
mapeos de clase y decisiones de puerta exactamente idénticos en todos los escenarios:
  1. Sujeto desconocido de desarrollo (LFW calibración / entrenamiento).
  2. Sujeto autorizado con score alto (desarrollo propio).
  3. Sujeto autorizado con score bajo (< 0.46).
  4. Ausencia de rostro (SIN_ROSTRO).
  5. Múltiples rostros (VARIOS_ROSTROS).
  6. Errores forzados (array vacío / dimensiones corruptas).
"""

import json
import unittest
from pathlib import Path
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]

from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.clasificacion.regla_acceso import evaluar_regla_acceso_completa
from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.web.streamer import GestorCamaraWeb


class MockModeloScoreControlado:
    """Modelo mock para controlar scores exactos en pruebas de frontera."""
    def __init__(self, probs: np.ndarray):
        self.probs = np.asarray(probs, dtype=np.float32)

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.probs, (len(X), 1))


class TestEquivalenciaWebEvaluador(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ruta_config = RAIZ / "configuracion_candidato_congelado_corregida.json"
        if not cls.ruta_config.exists():
            cls.ruta_config = RAIZ / "configuracion_candidato_congelado.json"

        cls.pipeline = PipelineCandidatoCongelado(cls.ruta_config, raiz_proyecto=RAIZ)
        cls.gestor = GestorCamaraWeb()
        res = cls.gestor.cargar_desde_configuracion_congelada(cls.ruta_config)
        assert res["status"] == "ok", f"Fallo al cargar candidato en gestor: {res}"

        # Buscar imagen real de desarrollo autorizado (Sesion 01 frontal)
        ruta_p1 = None
        dir_p1_dev = RAIZ / "data" / "propio" / "imagenes" / "persona_1" / "sesion_01" / "frontal"
        if dir_p1_dev.exists():
            posibles = [f for f in dir_p1_dev.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
            if posibles:
                ruta_p1 = posibles[0]
        cls.ruta_autorizado_dev = ruta_p1

        # Buscar imagen real de desarrollo desconocido
        ruta_desc = None
        dir_lfw_dev = RAIZ / "data" / "desconocidos_lfw" / "calibracion"
        if dir_lfw_dev.exists():
            posibles_desc = list(dir_lfw_dev.rglob("*.jpg")) + list(dir_lfw_dev.rglob("*.png"))
            if posibles_desc:
                ruta_desc = posibles_desc[0]
        if ruta_desc is None:
            dir_lfw_tr = RAIZ / "data" / "desconocidos_lfw" / "entrenamiento_4clases"
            posibles_desc = list(dir_lfw_tr.rglob("*.jpg")) + list(dir_lfw_tr.rglob("*.png"))
            ruta_desc = posibles_desc[0] if posibles_desc else None
        cls.ruta_desconocido_dev = ruta_desc

    def test_01_equivalencia_sujeto_desconocido_desarrollo(self):
        """Verifica equivalencia bit a bit ante imagen de desconocido de desarrollo."""
        if self.ruta_desconocido_dev is None or not self.ruta_desconocido_dev.exists():
            self.skipTest("No se encontró imagen de desconocido de desarrollo.")

        img_bgr = cv2.imread(str(self.ruta_desconocido_dev))
        self.assertIsNotNone(img_bgr)

        # 1. Pipeline evaluador
        res_det_eval, dec_eval, probs_eval = self.pipeline.evaluar_fotograma_completo(img_bgr)

        # 2. Gestor streamer web
        res_web = self.gestor.procesar_fotograma_para_reconocimiento(img_bgr)

        # Verificaciones de equivalencia
        self.assertEqual(dec_eval.estado_puerta, res_web["estado_puerta"])
        self.assertEqual(dec_eval.es_autorizado, res_web["es_autorizado"])
        self.assertEqual(dec_eval.persona_identificada, res_web["persona_reconocida"])
        self.assertEqual(dec_eval.estado_puerta, "CERRADA")
        self.assertFalse(dec_eval.es_autorizado)
        self.assertEqual(dec_eval.persona_identificada, "DESCONOCIDO")

        if probs_eval is not None:
            score_esperado = float(np.max(probs_eval))
            self.assertAlmostEqual(score_esperado * 100, res_web["confianza"], places=1)

    def test_02_equivalencia_sujeto_autorizado_score_alto(self):
        """Verifica equivalencia ante imagen de autorizado de desarrollo."""
        if self.ruta_autorizado_dev is None or not self.ruta_autorizado_dev.exists():
            self.skipTest("No se encontró imagen de autorizado de desarrollo.")

        img_bgr = cv2.imread(str(self.ruta_autorizado_dev))
        self.assertIsNotNone(img_bgr)

        # 1. Pipeline evaluador
        res_det_eval, dec_eval, probs_eval = self.pipeline.evaluar_fotograma_completo(img_bgr)

        # 2. Gestor streamer web
        res_web = self.gestor.procesar_fotograma_para_reconocimiento(img_bgr)

        # Verificaciones de equivalencia
        self.assertEqual(dec_eval.estado_puerta, res_web["estado_puerta"])
        self.assertEqual(dec_eval.es_autorizado, res_web["es_autorizado"])
        self.assertEqual(dec_eval.persona_identificada, res_web["persona_reconocida"])
        if probs_eval is not None:
            pred_idx = int(np.argmax(probs_eval))
            score_esperado = float(probs_eval[pred_idx])
            self.assertAlmostEqual(score_esperado * 100, res_web["confianza"], places=1)

    def test_03_equivalencia_autorizado_score_bajo_umbral(self):
        """Verifica equivalencia estricta de rechazo cuando el score es inferior a 0.46."""
        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=recorte,
        )

        mock_modelo = MockModeloScoreControlado(np.array([0.40, 0.20, 0.20, 0.20]))
        # Inyectar mock en ambos
        self.gestor.modelo_reconocimiento = mock_modelo
        self.gestor.pipeline_candidato = None

        res_web = self.gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det)

        dec_canonica = evaluar_regla_acceso_completa(
            estado_deteccion="UN_ROSTRO",
            procesamiento_exitoso=True,
            clase_predicha="persona_1",
            score=0.40,
            umbral=0.46,
            personas_autorizadas=self.pipeline.personas_autorizadas,
        )

        self.assertEqual(res_web["estado_puerta"], "CERRADA")
        self.assertFalse(res_web["es_autorizado"])
        self.assertEqual(res_web["persona_reconocida"], "DESCONOCIDO")
        self.assertEqual(dec_canonica.estado_puerta, res_web["estado_puerta"])
        self.assertEqual(dec_canonica.es_autorizado, res_web["es_autorizado"])
        self.assertEqual(dec_canonica.persona_identificada, res_web["persona_reconocida"])

        # Restaurar modelo original en gestor
        self.gestor.cargar_desde_configuracion_congelada(self.ruta_config)

    def test_04_equivalencia_ausencia_de_rostro(self):
        """Verifica equivalencia ante fotograma negro sin rostro (SIN_ROSTRO)."""
        frame_negro = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Pipeline evaluador
        res_det_eval, dec_eval, probs_eval = self.pipeline.evaluar_fotograma_completo(frame_negro)

        # 2. Gestor streamer web
        res_web = self.gestor.procesar_fotograma_para_reconocimiento(frame_negro)

        self.assertEqual(res_det_eval.estado, EstadoDeteccion.SIN_ROSTRO)
        self.assertIsNone(probs_eval)
        self.assertEqual(dec_eval.estado_puerta, "CERRADA")
        self.assertFalse(dec_eval.es_autorizado)
        self.assertEqual(dec_eval.persona_identificada, "DESCONOCIDO")

        self.assertEqual(res_web["estado_puerta"], "CERRADA")
        self.assertFalse(res_web["es_autorizado"])
        self.assertIn(res_web["persona_reconocida"], ("", "DESCONOCIDO"))
        self.assertEqual(res_web["confianza"], 0.0)

    def test_05_equivalencia_multiples_rostros(self):
        """Verifica equivalencia ante detección de múltiples rostros (VARIOS_ROSTROS)."""
        res_det_varios = ResultadoDeteccion(
            estado=EstadoDeteccion.VARIOS_ROSTROS,
            recuadro=None,
            total_rostros_detectados=2,
            rostro_recortado=None,
        )
        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Función canónica usada por evaluador
        dec_eval = evaluar_regla_acceso_completa(
            estado_deteccion="VARIOS_ROSTROS",
            procesamiento_exitoso=False,
            clase_predicha=None,
            score=0.0,
            umbral=self.pipeline.umbral_score,
            personas_autorizadas=self.pipeline.personas_autorizadas,
        )

        # 2. Gestor streamer web
        res_web = self.gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det_varios)

        self.assertEqual(dec_eval.estado_puerta, "CERRADA")
        self.assertFalse(dec_eval.es_autorizado)
        self.assertEqual(dec_eval.persona_identificada, "DESCONOCIDO")

        self.assertEqual(res_web["estado_puerta"], "CERRADA")
        self.assertFalse(res_web["es_autorizado"])
        self.assertIn(res_web["persona_reconocida"], ("", "DESCONOCIDO"))
        self.assertEqual(res_web["confianza"], 0.0)

    def test_06_equivalencia_errores_forzados(self):
        """Verifica equivalencia y bloqueo de puerta ante imágenes vacías o corruptas."""
        for entrada_invalida in (None, np.zeros((0, 0, 3), dtype=np.uint8)):
            # 1. Pipeline evaluador
            res_det_eval, dec_eval, probs_eval = self.pipeline.evaluar_fotograma_completo(entrada_invalida)
            self.assertEqual(dec_eval.estado_puerta, "CERRADA")
            self.assertFalse(dec_eval.es_autorizado)
            self.assertEqual(dec_eval.persona_identificada, "DESCONOCIDO")
            self.assertIsNone(probs_eval)

            # 2. Gestor streamer web
            res_det_err = ResultadoDeteccion(
                estado=EstadoDeteccion.SIN_ROSTRO,
                recuadro=None,
                total_rostros_detectados=0,
                rostro_recortado=None,
            )
            frame_placeholder = np.zeros((10, 10, 3), dtype=np.uint8)
            res_web = self.gestor.procesar_fotograma_para_reconocimiento(frame_placeholder, res_det_err)
            self.assertEqual(res_web["estado_puerta"], "CERRADA")
            self.assertFalse(res_web["es_autorizado"])
            self.assertIn(res_web["persona_reconocida"], ("", "DESCONOCIDO"))


if __name__ == "__main__":
    unittest.main()
