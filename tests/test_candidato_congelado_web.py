"""
Pruebas de integración para la carga del candidato congelado de 4 clases
en la interfaz web y verificación de concordancia estricta entre la evaluación
y el servidor web en vivo.
Incluye validación estricta contra inyección de valores numéricos inválidos (NaN, Inf),
verificación de integridad de vectores de probabilidad y persistencia/revocación
atómica de estado en transiciones entre autorizados, desconocidos y fallos de detección.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]

from reconocimiento.clasificacion.pipeline_candidato import (
    PipelineCandidatoCongelado,
    ErrorConfiguracionCandidato,
)
from reconocimiento.clasificacion.regla_acceso import (
    es_valor_probabilidad_valido,
    evaluar_decision_acceso,
    evaluar_regla_acceso_completa,
    validar_vector_probabilidades,
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import GestorCamaraWeb


class MockModelo4Clases:
    """Modelo simulado de 4 clases: [p1, p2, p3, desconocido]."""
    def __init__(self, probabilidades: np.ndarray):
        self.probabilidades = probabilidades

    def predecir_probabilidades(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.probabilidades])


class TestCandidatoCongeladoWeb(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ruta_config = RAIZ / "configuracion_candidato_congelado_corregida.json"
        if not cls.ruta_config.exists():
            cls.ruta_config = RAIZ / "configuracion_candidato_congelado.json"

    def setUp(self):
        gestor = GestorCamaraWeb()
        gestor.detener()

    def tearDown(self):
        gestor = GestorCamaraWeb()
        gestor.detener()

    def test_01_carga_candidato_desde_configuracion_congelada(self):
        """Verifica que crear_app con ruta_config_candidato cargue exactamente el candidato 4 clases."""
        app = crear_app(ruta_config_candidato=self.ruta_config, indice_inicial=-1)
        gestor = app.config["GESTOR_CAMARA"]

        self.assertTrue(gestor.es_cnn)
        self.assertEqual(gestor.umbral_confianza, 0.46)
        self.assertIn("Candidato Congelado", gestor.nombre_modelo_activo)
        self.assertEqual(gestor.personas_autorizadas, ["persona_1", "persona_2", "persona_3"])

        # Verificar mapeo de 4 clases
        id_to_p = gestor.mapeo_clases.get("id_to_persona", gestor.mapeo_clases)
        self.assertEqual(id_to_p["0"], "persona_1")
        self.assertEqual(id_to_p["1"], "persona_2")
        self.assertEqual(id_to_p["2"], "persona_3")
        self.assertEqual(id_to_p["3"], "desconocido")

    def test_02_preservacion_modelo_predeterminado_sin_config(self):
        """Verifica que crear_app() sin parámetros no active el candidato silenciosamente."""
        app = crear_app(indice_inicial=-1)
        gestor = app.config["GESTOR_CAMARA"]
        self.assertNotIn("Candidato Congelado", gestor.nombre_modelo_activo)

    def test_03_denegacion_estricta_clase_desconocido_en_web(self):
        """Verifica que la clase 'desconocido' nunca abra la puerta en la web, incluso con score alto."""
        app = crear_app(ruta_config_candidato=self.ruta_config, indice_inicial=-1)
        gestor = app.config["GESTOR_CAMARA"]

        # Inyectar predicción de clase desconocido (índice 3) con 99% de confianza
        gestor.modelo_reconocimiento = MockModelo4Clases(np.array([0.005, 0.003, 0.002, 0.990]))
        gestor.pipeline_candidato = None

        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=recorte,
        )

        ultimo = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det)

        self.assertEqual(ultimo["estado_puerta"], "CERRADA")
        self.assertFalse(ultimo["es_autorizado"])
        self.assertEqual(ultimo["persona_reconocida"], "DESCONOCIDO")

        # Verificar respuesta vía API Flask
        client = app.test_client()
        resp = client.get("/api/reconocimiento/estado")
        self.assertEqual(resp.status_code, 200)
        datos_api = resp.get_json()
        self.assertEqual(datos_api["estado_puerta"], "CERRADA")
        self.assertFalse(datos_api["es_autorizado"])
        self.assertEqual(datos_api["persona_reconocida"], "DESCONOCIDO")

        # Concordancia con función canónica
        dec_canonica = evaluar_regla_acceso_completa(
            estado_deteccion="UN_ROSTRO",
            procesamiento_exitoso=True,
            clase_predicha="desconocido",
            score=0.99,
            umbral=0.46,
            personas_autorizadas=["persona_1", "persona_2", "persona_3"],
        )
        self.assertEqual(dec_canonica.estado_puerta, ultimo["estado_puerta"])
        self.assertEqual(dec_canonica.es_autorizado, ultimo["es_autorizado"])
        self.assertEqual(dec_canonica.persona_identificada, ultimo["persona_reconocida"])

    def test_04_concordancia_autorizado_score_suficiente(self):
        """Verifica apertura cuando un autorizado supera el umbral de 0.46."""
        gestor = GestorCamaraWeb()
        gestor.cargar_desde_configuracion_congelada(self.ruta_config)

        gestor.modelo_reconocimiento = MockModelo4Clases(np.array([0.85, 0.05, 0.05, 0.05]))
        gestor.pipeline_candidato = None

        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=recorte,
        )

        ultimo = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det)

        self.assertEqual(ultimo["estado_puerta"], "ABIERTA")
        self.assertTrue(ultimo["es_autorizado"])
        self.assertEqual(ultimo["persona_reconocida"], "persona_1")

        dec_canonica = evaluar_regla_acceso_completa(
            estado_deteccion="UN_ROSTRO",
            procesamiento_exitoso=True,
            clase_predicha="persona_1",
            score=0.85,
            umbral=0.46,
            personas_autorizadas=["persona_1", "persona_2", "persona_3"],
        )
        self.assertEqual(dec_canonica.estado_puerta, "ABIERTA")
        self.assertEqual(dec_canonica.persona_identificada, "persona_1")

    def test_05_concordancia_autorizado_score_insuficiente(self):
        """Verifica bloqueo cuando un autorizado tiene score < 0.46."""
        gestor = GestorCamaraWeb()
        gestor.cargar_desde_configuracion_congelada(self.ruta_config)

        gestor.modelo_reconocimiento = MockModelo4Clases(np.array([0.20, 0.40, 0.20, 0.20]))
        gestor.pipeline_candidato = None

        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=recorte,
        )

        ultimo = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det)

        self.assertEqual(ultimo["estado_puerta"], "CERRADA")
        self.assertFalse(ultimo["es_autorizado"])
        self.assertEqual(ultimo["persona_reconocida"], "DESCONOCIDO")

        dec_canonica = evaluar_regla_acceso_completa(
            estado_deteccion="UN_ROSTRO",
            procesamiento_exitoso=True,
            clase_predicha="persona_2",
            score=0.40,
            umbral=0.46,
            personas_autorizadas=["persona_1", "persona_2", "persona_3"],
        )
        self.assertEqual(dec_canonica.estado_puerta, "CERRADA")
        self.assertFalse(dec_canonica.es_autorizado)
        self.assertEqual(dec_canonica.persona_identificada, "DESCONOCIDO")

    def test_06_revocacion_inmediata_de_estado_autorizado_secuencial(self):
        """
        Verifica que una predicción de autorizado previa NUNCA permanezca activa
        tras un fotograma con desconocido, fallo de detección o error.
        """
        gestor = GestorCamaraWeb()
        gestor.cargar_desde_configuracion_congelada(self.ruta_config)
        gestor.pipeline_candidato = None

        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det_rostro = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=recorte,
        )

        # 1. Fotograma autorizado -> ABIERTA
        gestor.modelo_reconocimiento = MockModelo4Clases(np.array([0.90, 0.03, 0.03, 0.04]))
        res1 = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det_rostro)
        self.assertEqual(res1["estado_puerta"], "ABIERTA")
        self.assertEqual(res1["persona_reconocida"], "persona_1")

        # 2. Fotograma siguiente con desconocido -> CERRADA y DESCONOCIDO inmediato
        gestor.modelo_reconocimiento = MockModelo4Clases(np.array([0.02, 0.02, 0.02, 0.94]))
        res2 = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det_rostro)
        self.assertEqual(res2["estado_puerta"], "CERRADA")
        self.assertEqual(res2["persona_reconocida"], "DESCONOCIDO")
        self.assertFalse(res2["es_autorizado"])

        # 3. Fotograma autorizado -> ABIERTA
        gestor.modelo_reconocimiento = MockModelo4Clases(np.array([0.05, 0.85, 0.05, 0.05]))
        res3 = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det_rostro)
        self.assertEqual(res3["estado_puerta"], "ABIERTA")
        self.assertEqual(res3["persona_reconocida"], "persona_2")

        # 4. Fotograma siguiente con SIN_ROSTRO -> CERRADA y revocación de persona_2
        res_det_sin = ResultadoDeteccion(
            estado=EstadoDeteccion.SIN_ROSTRO,
            recuadro=None,
            total_rostros_detectados=0,
            rostro_recortado=None,
        )
        res4 = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det_sin)
        self.assertEqual(res4["estado_puerta"], "CERRADA")
        self.assertNotEqual(res4["persona_reconocida"], "persona_2")
        self.assertFalse(res4["es_autorizado"])

        # 5. Fotograma autorizado -> ABIERTA
        gestor.modelo_reconocimiento = MockModelo4Clases(np.array([0.05, 0.05, 0.85, 0.05]))
        res5 = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det_rostro)
        self.assertEqual(res5["estado_puerta"], "ABIERTA")
        self.assertEqual(res5["persona_reconocida"], "persona_3")

        # 6. Fotograma siguiente con VARIOS_ROSTROS -> CERRADA y revocación de persona_3
        res_det_varios = ResultadoDeteccion(
            estado=EstadoDeteccion.VARIOS_ROSTROS,
            recuadro=None,
            total_rostros_detectados=2,
            rostro_recortado=None,
        )
        res6 = gestor.procesar_fotograma_para_reconocimiento(frame_raw, res_det_varios)
        self.assertEqual(res6["estado_puerta"], "CERRADA")
        self.assertNotEqual(res6["persona_reconocida"], "persona_3")
        self.assertFalse(res6["es_autorizado"])

    def test_07_rechazo_valores_numericos_invalidos_nan_inf(self):
        """Verifica que NaN, Inf, None y valores fuera de rango NUNCA abran la puerta."""
        autorizados = ["persona_1", "persona_2", "persona_3"]

        # 1. Scores inválidos con umbral válido
        for val_invalido in (float("nan"), float("inf"), float("-inf"), -0.1, 1.05, None, "0.9"):
            dec = evaluar_decision_acceso("persona_1", val_invalido, 0.46, autorizados)
            self.assertEqual(dec.estado_puerta, "CERRADA")
            self.assertFalse(dec.es_autorizado)
            self.assertEqual(dec.persona_identificada, "DESCONOCIDO")
            self.assertIn("SCORE_INVALIDO", dec.motivo_cierre)

            dec_comp = evaluar_regla_acceso_completa("UN_ROSTRO", True, "persona_1", val_invalido, 0.46, autorizados)
            self.assertEqual(dec_comp.estado_puerta, "CERRADA")
            self.assertFalse(dec_comp.es_autorizado)
            self.assertEqual(dec_comp.persona_identificada, "DESCONOCIDO")
            self.assertIn("SCORE_INVALIDO", dec_comp.motivo_cierre)

        # 2. Umbrales inválidos con score válido
        for umb_invalido in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01, None, "0.46"):
            dec = evaluar_decision_acceso("persona_1", 0.95, umb_invalido, autorizados)
            self.assertEqual(dec.estado_puerta, "CERRADA")
            self.assertFalse(dec.es_autorizado)
            self.assertEqual(dec.persona_identificada, "DESCONOCIDO")
            self.assertIn("UMBRAL_INVALIDO", dec.motivo_cierre)

            dec_comp = evaluar_regla_acceso_completa("UN_ROSTRO", True, "persona_1", 0.95, umb_invalido, autorizados)
            self.assertEqual(dec_comp.estado_puerta, "CERRADA")
            self.assertFalse(dec_comp.es_autorizado)
            self.assertEqual(dec_comp.persona_identificada, "DESCONOCIDO")
            self.assertIn("UMBRAL_INVALIDO", dec_comp.motivo_cierre)

    def test_08_validacion_vector_probabilidades(self):
        """Verifica las invariantes matemáticas del vector de probabilidades del clasificador."""
        # Válido 4 clases
        val, mot = validar_vector_probabilidades([0.7, 0.1, 0.1, 0.1], n_clases_esperadas=4)
        self.assertTrue(val)
        self.assertEqual(mot, "")

        # Longitud incorrecta
        val, mot = validar_vector_probabilidades([0.8, 0.1, 0.1], n_clases_esperadas=4)
        self.assertFalse(val)
        self.assertIn("Dimensión", mot)

        # Contiene NaN
        val, mot = validar_vector_probabilidades([0.5, float("nan"), 0.2, 0.3], n_clases_esperadas=4)
        self.assertFalse(val)
        self.assertIn("NaN", mot)

        # Contiene infinitos
        val, mot = validar_vector_probabilidades([0.5, float("inf"), 0.2, 0.3], n_clases_esperadas=4)
        self.assertFalse(val)

        # Fuera de rango [0, 1]
        val, mot = validar_vector_probabilidades([-0.1, 0.9, 0.1, 0.1], n_clases_esperadas=4)
        self.assertFalse(val)
        self.assertIn("fuera del rango", mot)

        # Suma inconsistente con 1.0 (fuera de tolerancia 1e-3)
        val, mot = validar_vector_probabilidades([0.3, 0.3, 0.3, 0.3], n_clases_esperadas=4)
        self.assertFalse(val)
        self.assertIn("La suma de probabilidades", mot)

    def test_09_rechazo_configuracion_invalida_impide_activacion(self):
        """Verifica que una configuración congelada anómala aborte y no active el candidato."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write(json.dumps({"candidato": {"nombre": "Invalido"}}))
            tf_path = Path(tf.name)

        gestor = GestorCamaraWeb()
        res = gestor.cargar_desde_configuracion_congelada(tf_path)
        self.assertEqual(res["status"], "error")
        self.assertEqual(gestor.nombre_modelo_activo, "ninguno")
        self.assertEqual(gestor.ultimo_resultado_reconocimiento["estado_puerta"], "CERRADA")
        self.assertFalse(gestor.ultimo_resultado_reconocimiento["es_autorizado"])

        # Con Pipeline directo debe alzar ErrorConfiguracionCandidato
        with self.assertRaises(ErrorConfiguracionCandidato):
            PipelineCandidatoCongelado(tf_path)

        tf_path.unlink()


if __name__ == "__main__":
    unittest.main()
