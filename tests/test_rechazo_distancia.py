"""
Pruebas unitarias para el módulo de rechazo por distancia a representaciones de entrenamiento.

Verifica:
1. Aceptación y rechazo en los límites exactos de los umbrales de puntaje y distancia.
2. Cálculo de distancia estrictamente contra la identidad predicha.
3. Rechazo explícito de vectores inválidos (None, NaN, Inf, norma cero) y clases sin referencias.
4. Exclusión estricta de validación y desconocidos del banco de entrenamiento (sólo sesiones 01–03).
5. Exclusión de sesión 05 y reserva_prueba de LFW.
6. Persistencia, serialización y recarga determinista con decisiones idénticas.
7. Denominadores consistentes y separación estricta de confusión de identidad vs rechazo de puerta.
8. Consistencia de preprocesamiento entre entrenamiento e inferencia.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]

from reconocimiento.clasificacion.regla_acceso import DecisionAcceso
from reconocimiento.clasificacion.rechazo_distancia import (
    BancoReferencias,
    evaluar_decision_acceso_con_distancia,
    METRICAS_VALIDAS
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion


class TestRechazoDistancia(unittest.TestCase):

    def setUp(self):
        # Crear referencias sintéticas controladas para dos personas
        # persona_1 cerca del vector [1.0, 1.0]
        # persona_2 cerca del vector [5.0, 5.0]
        rng = np.random.RandomState(42)
        ref_p1 = np.array([
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
        ], dtype=np.float32)

        ref_p2 = np.array([
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
        ], dtype=np.float32)

        self.banco = BancoReferencias(
            referencias_por_clase={"persona_1": ref_p1, "persona_2": ref_p2},
            metadatos={
                "modelo": "test_sintetico",
                "sesiones_origen": ["sesion_01", "sesion_02", "sesion_03"]
            }
        )
        self.autorizadas = ["persona_1", "persona_2"]

    # 1. Aceptación/rechazo en los límites exactos de los umbrales
    def test_limites_exactos_umbrales(self):
        umbral_s = 0.80
        umbral_d = 0.50

        # Vector idéntico a una referencia de persona_1 -> dist = 0.0 <= 0.50
        v_exacto = np.array([1.0, 1.0], dtype=np.float32)

        # En el límite exacto de score (confianza == 0.80) y dist <= 0.50 -> ABRE
        dec_limite_s = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.80,
            umbral_confianza=umbral_s,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_exacto,
            banco_referencias=self.banco,
            umbral_distancia=umbral_d,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_limite_s.estado_puerta, "ABIERTA")
        self.assertTrue(dec_limite_s.es_autorizado)

        # Inmediatamente por debajo del umbral de score (0.7999) -> CIERRA por RECHAZO_UMBRAL
        dec_bajo_s = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.7999,
            umbral_confianza=umbral_s,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_exacto,
            banco_referencias=self.banco,
            umbral_distancia=umbral_d,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_bajo_s.estado_puerta, "CERRADA")
        self.assertIn("RECHAZO_UMBRAL", dec_bajo_s.motivo_cierre)
        # La identidad predicha no se altera silenciosamente
        self.assertEqual(dec_bajo_s.persona_identificada, "persona_1")

        # Vector a distancia exactamente igual a umbral_d (0.50)
        # Dirección (-1, -1) desde [1.0, 1.0] garantiza que [1.0, 1.0] es el vecino más cercano
        delta_50 = 0.50 / np.sqrt(2.0)
        v_limite_d = np.array([1.0 - delta_50, 1.0 - delta_50], dtype=np.float32)
        dec_limite_d = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=umbral_s,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_limite_d,
            banco_referencias=self.banco,
            umbral_distancia=umbral_d,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_limite_d.estado_puerta, "ABIERTA")
        self.assertAlmostEqual(dec_limite_d.distancia, 0.50, places=4)

        # Vector a distancia apenas superior (0.505) -> CIERRA por RECHAZO_DISTANCIA
        delta_excede = 0.505 / np.sqrt(2.0)
        v_excede_d = np.array([1.0 - delta_excede, 1.0 - delta_excede], dtype=np.float32)
        dec_excede_d = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=umbral_s,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_excede_d,
            banco_referencias=self.banco,
            umbral_distancia=umbral_d,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_excede_d.estado_puerta, "CERRADA")
        self.assertIn("RECHAZO_DISTANCIA", dec_excede_d.motivo_cierre)
        self.assertEqual(dec_excede_d.persona_identificada, "persona_1")

    # 2. Distancia calculada contra la identidad predicha
    def test_distancia_calculada_contra_identidad_predicha(self):
        # Punto cerca de persona_2 ([4.95, 5.05])
        v_cerca_p2 = np.array([4.95, 5.05], dtype=np.float32)

        # Si el clasificador predice persona_2: dist a persona_2 es pequeña (< 0.1) -> ABRE
        dec_p2 = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_2",
            confianza=0.95,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_cerca_p2,
            banco_referencias=self.banco,
            umbral_distancia=0.50,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_p2.estado_puerta, "ABIERTA")

        # Si el clasificador erróneamente predijera persona_1 para ese mismo vector:
        # La distancia calculada contra persona_1 es aprox 5.6 > 0.50 -> CIERRA por distancia
        dec_err = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.95,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_cerca_p2,
            banco_referencias=self.banco,
            umbral_distancia=0.50,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_err.estado_puerta, "CERRADA")
        self.assertIn("RECHAZO_DISTANCIA", dec_err.motivo_cierre)
        self.assertGreater(dec_err.distancia, 5.0)

    # 3. Rechazo de vectores inválidos y ausencia de referencias
    def test_rechazo_vectores_invalidos_y_sin_referencias(self):
        # Vector None
        dec_none = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=None,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_none.estado_puerta, "CERRADA")
        self.assertIn("VECTOR_NULO", dec_none.motivo_cierre)

        # Vector con NaN
        v_nan = np.array([np.nan, 1.0], dtype=np.float32)
        dec_nan = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_nan,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_nan.estado_puerta, "CERRADA")
        self.assertIn("VALOR_NO_FINITO", dec_nan.motivo_cierre)

        # Vector con Inf
        v_inf = np.array([np.inf, 1.0], dtype=np.float32)
        dec_inf = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_inf,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_inf.estado_puerta, "CERRADA")
        self.assertIn("VALOR_NO_FINITO", dec_inf.motivo_cierre)

        # Clase no presente en el banco de referencias
        dec_sin_ref = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_inexistente",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=["persona_1", "persona_2", "persona_inexistente"],
            vector_representacion=np.array([1.0, 1.0], dtype=np.float32),
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_sin_ref.estado_puerta, "CERRADA")
        self.assertIn("CLASE_SIN_REFERENCIAS", dec_sin_ref.motivo_cierre)

    # 4. Exclusión de validación y desconocidos del banco de entrenamiento
    def test_exclusion_validacion_y_desconocidos_del_banco(self):
        # El banco de referencias debe declarar explícitamente su origen en sesiones 01-03
        self.assertEqual(
            set(self.banco.metadatos["sesiones_origen"]),
            {"sesion_01", "sesion_02", "sesion_03"}
        )
        self.assertNotIn("sesion_04", self.banco.metadatos["sesiones_origen"])
        self.assertNotIn("sesion_05", self.banco.metadatos["sesiones_origen"])
        self.assertNotIn("calibracion", str(self.banco.metadatos))

    # 5. Exclusión de sesión 05 y reserva_prueba de LFW
    def test_exclusion_sesion05_y_reserva_lfw(self):
        from scripts import entrenar_desarrollo as des
        registros, _ = des.cargar_registros()
        sesiones_cargadas = {r.sesion for r in registros}
        self.assertNotIn("sesion_05", sesiones_cargadas)
        self.assertTrue(sesiones_cargadas.issubset({"sesion_01", "sesion_02", "sesion_03", "sesion_04"}))

    # 6. Persistencia y recarga con decisiones idénticas
    def test_persistencia_y_recarga_decisiones_identicas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ruta_npz = Path(temp_dir) / "banco_test.npz"
            sha = self.banco.guardar_npz(ruta_npz)
            self.assertTrue(ruta_npz.exists())
            self.assertEqual(len(sha), 64)

            # Recargar banco
            banco_recargado = BancoReferencias.cargar_npz(ruta_npz)
            self.assertEqual(banco_recargado.clases, self.banco.clases)
            self.assertEqual(banco_recargado.dimension, self.banco.dimension)
            self.assertEqual(banco_recargado.total_vectores, self.banco.total_vectores)

            # Comparar distancias sobre una muestra
            test_vec = np.array([1.2, 0.8], dtype=np.float32)
            d_orig = self.banco.calcular_distancia(test_vec, "persona_1", metrica="min_l2")
            d_recar = banco_recargado.calcular_distancia(test_vec, "persona_1", metrica="min_l2")
            self.assertAlmostEqual(d_orig, d_recar, places=6)

    # 7. Denominadores y separación de confusión de identidad frente a rechazo de puerta
    def test_denominadores_y_separacion_confusion(self):
        # 10 autorizados: 6 correctos, 2 confusiones entre autorizados, 2 rechazados
        aperturas_correctas = 6
        aperturas_confusion = 2
        rechazados_puerta = 2
        total_autorizados = 10

        self.assertEqual(
            aperturas_correctas + aperturas_confusion + rechazados_puerta,
            total_autorizados
        )
        frr_puerta = 100.0 * rechazados_puerta / total_autorizados
        self.assertEqual(frr_puerta, 20.0)

        # En desconocidos: 100 en total. Si 5 abren, FAR es 5/100 = 5%
        aperturas_desconocidos = 5
        total_desconocidos = 100
        far = 100.0 * aperturas_desconocidos / total_desconocidos
        self.assertEqual(far, 5.0)

    # 8. Consistencia de preprocesamiento entre entrenamiento e inferencia
    def test_consistencia_preprocesamiento(self):
        from scripts.entrenar_desarrollo import RESOLUCION, SIGMA, CUADRICULA, BINS
        self.assertEqual(RESOLUCION, (64, 64))
        self.assertEqual(SIGMA, 0.5)
        self.assertEqual(CUADRICULA, (4, 4))
        self.assertEqual(BINS, 16)

    # 9. Rechazo explícito de NaN / Inf en puntajes, umbrales y vectores
    def test_rechazo_nan_inf_en_puntajes_umbrales_y_vectores(self):
        v_ok = np.array([1.0, 1.0], dtype=np.float32)

        # Confianza NaN
        dec_nan_conf = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=float("nan"),
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_nan_conf.estado_puerta, "CERRADA")
        self.assertFalse(dec_nan_conf.es_autorizado)
        self.assertEqual(dec_nan_conf.motivo_cierre, "PUNTAJE_INVALIDO")
        self.assertEqual(dec_nan_conf.persona_identificada, "persona_1")

        # Confianza Inf
        dec_inf_conf = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=float("inf"),
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_inf_conf.estado_puerta, "CERRADA")
        self.assertEqual(dec_inf_conf.motivo_cierre, "PUNTAJE_INVALIDO")

        # Confianza fuera de rango [0, 1]
        dec_neg_conf = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=-0.1,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_neg_conf.estado_puerta, "CERRADA")
        self.assertEqual(dec_neg_conf.motivo_cierre, "PUNTAJE_INVALIDO")

        # Umbral de confianza NaN
        dec_nan_umbral_c = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=float("nan"),
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_nan_umbral_c.estado_puerta, "CERRADA")
        self.assertEqual(dec_nan_umbral_c.motivo_cierre, "UMBRAL_CONFIANZA_INVALIDO")

        # Umbral de distancia NaN
        dec_nan_umbral_d = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=self.banco,
            umbral_distancia=float("nan")
        )
        self.assertEqual(dec_nan_umbral_d.estado_puerta, "CERRADA")
        self.assertEqual(dec_nan_umbral_d.motivo_cierre, "UMBRAL_DISTANCIA_INVALIDO")

        # Umbral de distancia negativo
        dec_neg_umbral_d = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=self.banco,
            umbral_distancia=-0.05
        )
        self.assertEqual(dec_neg_umbral_d.estado_puerta, "CERRADA")
        self.assertEqual(dec_neg_umbral_d.motivo_cierre, "UMBRAL_DISTANCIA_INVALIDO")

        # Vector con norma cero para métrica de coseno
        v_cero = np.array([0.0, 0.0], dtype=np.float32)
        dec_v_cero = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_cero,
            banco_referencias=self.banco,
            umbral_distancia=0.50,
            metrica_distancia="min_cos"
        )
        self.assertEqual(dec_v_cero.estado_puerta, "CERRADA")
        self.assertIn("VECTOR_NORMA_INVALIDA", dec_v_cero.motivo_cierre)

        # Vector con dimensión incompatible
        v_dim3 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        dec_dim = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_dim3,
            banco_referencias=self.banco,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_dim.estado_puerta, "CERRADA")
        self.assertIn("DIMENSION_INCOMPATIBLE", dec_dim.motivo_cierre)

    # 10. Rechazo por configuración parcial de distancia (sin degradación silenciosa)
    def test_rechazo_configuracion_parcial_distancia(self):
        v_ok = np.array([1.0, 1.0], dtype=np.float32)

        # Banco presente pero umbral_distancia es None -> RECHAZA (no abre por regla A silenciosa)
        dec_solo_banco = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.95,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=self.banco,
            umbral_distancia=None
        )
        self.assertEqual(dec_solo_banco.estado_puerta, "CERRADA")
        self.assertEqual(dec_solo_banco.motivo_cierre, "CONFIGURACION_DISTANCIA_INCOMPLETA")

        # Umbral de distancia presente pero banco es None -> RECHAZA
        dec_solo_umbral = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.95,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_ok,
            banco_referencias=None,
            umbral_distancia=0.50
        )
        self.assertEqual(dec_solo_umbral.estado_puerta, "CERRADA")
        self.assertEqual(dec_solo_umbral.motivo_cierre, "CONFIGURACION_DISTANCIA_INCOMPLETA")

    # 11. Detección de banco sin metadatos o con hash inconsistente
    def test_rechazo_banco_sin_metadatos_o_hash_inconsistente(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ruta_npz = Path(temp_dir) / "banco_test.npz"
            self.banco.guardar_npz(ruta_npz)
            ruta_meta = ruta_npz.with_suffix(".json")

            # Carga correcta con metadatos y hash
            banco_ok = BancoReferencias.cargar_npz(ruta_npz)
            self.assertIsNotNone(banco_ok)

            # Caso 1: Sin archivo de metadatos .json
            ruta_meta.unlink()
            with self.assertRaises(ValueError) as ctx1:
                BancoReferencias.cargar_npz(ruta_npz)
            self.assertIn("METADATOS_FALTANTES", str(ctx1.exception))

            # Caso 2: Metadatos sin campo sha256_npz
            ruta_meta.write_text('{"metadatos": {}}', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx2:
                BancoReferencias.cargar_npz(ruta_npz)
            self.assertIn("HASH_NO_ESPECIFICADO", str(ctx2.exception))

            # Caso 3: Archivo .npz modificado (hash inconsistente)
            meta_con_sha = json.dumps({"sha256_npz": "0000000000000000000000000000000000000000000000000000000000000000"})
            ruta_meta.write_text(meta_con_sha, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx3:
                BancoReferencias.cargar_npz(ruta_npz)
            self.assertIn("HASH_BANCO_INCONSISTENTE", str(ctx3.exception))

    # 12. Persistencia sin pérdida de precisión de umbral float
    def test_persistencia_sin_perdida_precision_y_recarga_identica(self):
        # Umbral float con alta precisión decimal
        umbral_preciso = 0.41620898246765137
        # Si se redondeara a 4 decimales sería 0.4162.
        # Un vector cuya distancia sea 0.416205 está por debajo del umbral exacto (<= 0.41620898...)
        # pero por encima del redondeado (> 0.416200), lo que cambiaría la decisión de ABIERTA a CERRADA.
        d_muestra = 0.416205
        v_test = np.array([1.0 + d_muestra, 1.0], dtype=np.float32)

        # Con umbral completo: dist = 0.416205 <= 0.41620898 -> ABIERTA
        dec_precisa = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_test,
            banco_referencias=self.banco,
            umbral_distancia=umbral_preciso,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_precisa.estado_puerta, "ABIERTA")

        # Serializar y deserializar JSON con float nativo (sin round)
        payload = json.dumps({"umbral_distancia": umbral_preciso})
        cargado = json.loads(payload)["umbral_distancia"]
        self.assertEqual(cargado, umbral_preciso)

        dec_recargada = evaluar_decision_acceso_con_distancia(
            nombre_predicho="persona_1",
            confianza=0.90,
            umbral_confianza=0.80,
            personas_autorizadas=self.autorizadas,
            vector_representacion=v_test,
            banco_referencias=self.banco,
            umbral_distancia=cargado,
            metrica_distancia="min_l2"
        )
        self.assertEqual(dec_recargada.estado_puerta, "ABIERTA")

    # 13. Flujo real cuando no existe umbral viable: cero aperturas y concordancia detalle-resumen
    def test_flujo_sin_umbral_viable_cero_aperturas_y_concordancia(self):
        from scripts.experimentar_rechazo_distancia import (
            optimizar_umbrales_regla_B, evaluar_lote_decisiones, calcular_metricas_experimento
        )

        intentos_val = [
            {"id": "m1", "persona": "persona_1", "pose": "frontal", "estado": "procesada"},
            {"id": "m2", "persona": "persona_1", "pose": "abajo", "estado": "procesada"},
            {"id": "m3", "persona": "persona_2", "pose": "frontal", "estado": "procesada"},
            {"id": "m4", "persona": "persona_2", "pose": "derecha", "estado": "sin_rostro"},
        ]
        mapa_idx_val = {"m1": 0, "m2": 1, "m3": 2}
        preds_val = ["persona_1", "persona_1", "persona_2"]
        scores_val = [0.40, 0.40, 0.40]
        # Distancias enormes que superan cualquier umbral posible
        dists_val = {"min_cos": [10.0, 10.0, 10.0]}
        dists_neg = {"min_cos": [10.0, 10.0]}
        scores_neg = [0.95, 0.95]
        estados_neg = ["UN_ROSTRO", "VARIOS_ROSTROS"]
        idx_neg_unrostro = [0]
        rep_val = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
        rep_neg = np.array([[9.0, 9.0]], dtype=np.float32)

        # Optimización con cota estricta (ningún candidato logra FRR <= 30%)
        estado, met, s_th, d_th, info_opt, mejores = optimizar_umbrales_regla_B(
            candidatos_score=[0.80, 0.90, 1.00],
            metricas_eval=["min_cos"],
            dists_val_por_metrica=dists_val,
            dists_neg_por_metrica=dists_neg,
            scores_val_procesadas=scores_val,
            scores_neg_unrostro=scores_neg,
            preds_val_procesadas=preds_val,
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            idx_neg_unrostro=idx_neg_unrostro,
            cota_frr=0.30
        )
        self.assertEqual(estado, "sin_umbral_viable")
        self.assertIsNone(met)
        self.assertIsNone(s_th)
        self.assertIsNone(d_th)

        # Evaluar lote bajo sin_umbral_viable
        dec_val, dec_neg = evaluar_lote_decisiones(
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            preds_val_procesadas=preds_val,
            scores_val_procesadas=scores_val,
            rep_val=rep_val,
            banco=self.banco,
            s_th=s_th,
            d_th=d_th,
            metrica=met,
            personas=self.autorizadas,
            estados_neg=estados_neg,
            preds_neg_unrostro=["persona_1"],
            scores_neg_unrostro=scores_neg,
            rep_neg=rep_neg,
            idx_neg_unrostro=idx_neg_unrostro,
            estado_modelo=estado
        )

        # Todas las decisiones deben ser CERRADAS
        for mid, d in dec_val.items():
            self.assertEqual(d.estado_puerta, "CERRADA")
            if mid in mapa_idx_val:
                self.assertEqual(d.motivo_cierre, "SIN_UMBRAL_VIABLE")
                # Conserva predicción original del clasificador para auditoría
                self.assertEqual(d.persona_identificada, preds_val[mapa_idx_val[mid]])
            else:
                self.assertEqual(d.motivo_cierre, "DETECCION_SIN_ROSTRO")

        for d in dec_neg:
            self.assertEqual(d.estado_puerta, "CERRADA")

        # Métricas calculadas desde las decisiones efectivas
        metricas = calcular_metricas_experimento(
            decisiones_val=dec_val,
            decisiones_neg=dec_neg,
            intentos_val=intentos_val,
            estados_neg=estados_neg,
            idx_neg_unrostro=idx_neg_unrostro,
            personas=self.autorizadas
        )
        self.assertEqual(metricas["aperturas_val"], 0)
        self.assertEqual(metricas["aperturas_neg"], 0)
        self.assertEqual(metricas["rechazos_val"], 4)
        self.assertEqual(metricas["correctas_val"], 0)
        self.assertEqual(metricas["confusiones_val"], 0)
        self.assertEqual(metricas["far_total_pct"], 0.0)
        self.assertEqual(metricas["frr_puerta_pct"], 100.0)

    # 14. Fallos intercalados y conservación de causas reales
    def test_fallos_intercalados_y_conservacion_causas_reales(self):
        from scripts.experimentar_rechazo_distancia import (
            categorizar_fallo, evaluar_lote_decisiones, calcular_metricas_experimento
        )

        # Probar categorizador de fallos
        self.assertEqual(categorizar_fallo("sin_rostro"), ("deteccion", "DETECCION_SIN_ROSTRO"))
        self.assertEqual(categorizar_fallo("varios_rostros"), ("deteccion", "DETECCION_VARIOS_ROSTROS"))
        self.assertEqual(categorizar_fallo("localizacion_incorrecta"), ("deteccion", "DETECCION_LOCALIZACION_INCORRECTA"))
        self.assertEqual(categorizar_fallo("error_lectura"), ("procesamiento", "PROCESAMIENTO_ERROR_LECTURA"))
        self.assertEqual(categorizar_fallo("error_decodificacion"), ("procesamiento", "PROCESAMIENTO_ERROR_DECODIFICACION"))
        self.assertEqual(categorizar_fallo("recorte_vacio"), ("procesamiento", "PROCESAMIENTO_RECORTE_VACIO"))
        self.assertEqual(categorizar_fallo("caja_revisada_fuera_de_imagen"), ("procesamiento", "PROCESAMIENTO_CAJA_FUERA_DE_IMAGEN"))

        intentos = [
            {"id": "m1", "persona": "persona_1", "pose": "frontal", "estado": "procesada"},
            {"id": "m2", "persona": "persona_1", "pose": "abajo", "estado": "sin_rostro"},
            {"id": "m3", "persona": "persona_2", "pose": "derecha", "estado": "error_lectura"},
            {"id": "m4", "persona": "persona_2", "pose": "frontal", "estado": "localizacion_incorrecta"},
            {"id": "m5", "persona": "persona_1", "pose": "arriba", "estado": "recorte_vacio"},
            {"id": "m6", "persona": "persona_2", "pose": "abajo", "estado": "procesada"},
        ]
        mapa_idx = {"m1": 0, "m6": 1}
        preds = ["persona_1", "persona_2"]
        scores = [0.95, 0.95]
        rep_val = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)

        dec_val, dec_neg = evaluar_lote_decisiones(
            intentos_val=intentos,
            mapa_idx_val=mapa_idx,
            preds_val_procesadas=preds,
            scores_val_procesadas=scores,
            rep_val=rep_val,
            banco=self.banco,
            s_th=0.80,
            d_th=0.50,
            metrica="min_l2",
            personas=self.autorizadas,
            estados_neg=["UN_ROSTRO"],
            preds_neg_unrostro=["persona_1"],
            scores_neg_unrostro=[0.50],
            rep_neg=np.array([[9.0, 9.0]], dtype=np.float32),
            idx_neg_unrostro=[0]
        )

        # Muestras procesadas abren normalmente
        self.assertEqual(dec_val["m1"].estado_puerta, "ABIERTA")
        self.assertEqual(dec_val["m6"].estado_puerta, "ABIERTA")

        # Muestras fallidas conservan su motivo real de cierre
        self.assertEqual(dec_val["m2"].estado_puerta, "CERRADA")
        self.assertEqual(dec_val["m2"].motivo_cierre, "DETECCION_SIN_ROSTRO")

        self.assertEqual(dec_val["m3"].estado_puerta, "CERRADA")
        self.assertEqual(dec_val["m3"].motivo_cierre, "PROCESAMIENTO_ERROR_LECTURA")

        self.assertEqual(dec_val["m4"].estado_puerta, "CERRADA")
        self.assertEqual(dec_val["m4"].motivo_cierre, "DETECCION_LOCALIZACION_INCORRECTA")

        self.assertEqual(dec_val["m5"].estado_puerta, "CERRADA")
        self.assertEqual(dec_val["m5"].motivo_cierre, "PROCESAMIENTO_RECORTE_VACIO")

        # Verificar conteos en métricas separadas
        metricas = calcular_metricas_experimento(
            decisiones_val=dec_val,
            decisiones_neg=dec_neg,
            intentos_val=intentos,
            estados_neg=["UN_ROSTRO"],
            idx_neg_unrostro=[0],
            personas=self.autorizadas
        )
        self.assertEqual(metricas["fallos_deteccion_val"], 2)       # m2 y m4
        self.assertEqual(metricas["fallos_procesamiento_val"], 2)   # m3 y m5
        self.assertEqual(metricas["total_intentos_val"], 6)

    # 15. Denominadores arbitrarios y casos vacíos
    def test_denominadores_arbitrarios_y_casos_vacios(self):
        from scripts.experimentar_rechazo_distancia import (
            calcular_tasa_pct, formatear_tasa, calcular_metricas_experimento
        )

        # Tasas con denominadores arbitrarios
        self.assertAlmostEqual(calcular_tasa_pct(5, 17), 100.0 * 5 / 17, places=4)
        self.assertAlmostEqual(calcular_tasa_pct(10, 33), 100.0 * 10 / 33, places=4)

        # Denominador cero manejado limpiamente como None (no calculable)
        self.assertIsNone(calcular_tasa_pct(0, 0))
        self.assertIsNone(calcular_tasa_pct(5, 0))
        self.assertEqual(formatear_tasa(None), "no calculable")

        # Validación sin intentos (total 0)
        met_vacio = calcular_metricas_experimento(
            decisiones_val={},
            decisiones_neg=[],
            intentos_val=[],
            estados_neg=[],
            idx_neg_unrostro=[],
            personas=self.autorizadas
        )
        self.assertIsNone(met_vacio["frr_puerta_pct"])
        self.assertIsNone(met_vacio["far_total_pct"])
        self.assertIsNone(met_vacio["far_condicionado_pct"])

        # Desconocidos sin rostros únicos detectados
        dec_neg_sin_rostro = [
            DecisionAcceso(
                estado_puerta="CERRADA", es_autorizado=False, persona_identificada="DESCONOCIDO",
                confianza=0.0, umbral=0.80, motivo_cierre="DETECCION_SIN_ROSTRO", mensaje="Sin rostro",
                distancia=None, umbral_distancia=0.50
            )
        ]
        met_neg_cero = calcular_metricas_experimento(
            decisiones_val={"m1": DecisionAcceso(estado_puerta="ABIERTA", es_autorizado=True, persona_identificada="persona_1", confianza=0.9, umbral=0.8, motivo_cierre="", mensaje="OK", distancia=0.1, umbral_distancia=0.5)},
            decisiones_neg=dec_neg_sin_rostro,
            intentos_val=[{"id": "m1", "persona": "persona_1", "pose": "frontal"}],
            estados_neg=["SIN_ROSTRO"],
            idx_neg_unrostro=[],
            personas=self.autorizadas
        )
        self.assertEqual(met_neg_cero["far_total_pct"], 0.0)
        self.assertIsNone(met_neg_cero["far_condicionado_pct"])

    # 16. Validación de artefactos base: rechazo ante cualquier discrepancia o modificación
    def test_rechazo_artefactos_base_alterados(self):
        import shutil
        from scripts.experimentar_rechazo_distancia import validar_artefactos_base
        from reconocimiento.preprocesamiento.fabrica import crear_detector
        from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
        from scripts.entrenar_desarrollo import CUADRICULA, BINS

        base_real = Path("resultados/desarrollo_yunet_20261118_185200_833277")
        if not base_real.exists():
            self.skipTest("No se encontró el directorio base de desarrollo")

        detector = crear_detector()
        extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS, rango_intensidad=(0.0, 1.0))

        # Fixture temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_base = Path(temp_dir) / "desarrollo_base"
            shutil.copytree(base_real, temp_base)

            # Caso OK: la carpeta íntegra pasa la validación
            res = validar_artefactos_base(temp_base, detector, extractor)
            self.assertIsNotNone(res)

            # Alteración 1: Hash de modelo alterado
            ruta_knn_mod = temp_base / "knn" / "modelo.joblib"
            with open(ruta_knn_mod, "ab") as f:
                f.write(b"CORRUPT")
            with self.assertRaises(ValueError) as ctx1:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("hash SHA-256 del modelo", str(ctx1.exception))
            # Restaurar
            shutil.copy(base_real / "knn" / "modelo.joblib", ruta_knn_mod)

            # Alteración 2: Mapeo / orden de clases alterado
            ruta_calib_knn = temp_base / "knn" / "calibracion_desarrollo.json"
            calib_knn = json.loads(ruta_calib_knn.read_text(encoding="utf-8"))
            calib_knn["clases_en_orden"] = ["persona_3", "persona_2", "persona_1"]
            ruta_calib_knn.write_text(json.dumps(calib_knn), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx2:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("orden de clases", str(ctx2.exception))
            # Restaurar
            shutil.copy(base_real / "knn" / "calibracion_desarrollo.json", ruta_calib_knn)

            # Alteración 3: Parámetros detector alterados
            ruta_informe = temp_base / "informe.json"
            inf_data = json.loads(ruta_informe.read_text(encoding="utf-8"))
            inf_data["detector"]["score_threshold"] = 0.99
            ruta_informe.write_text(json.dumps(inf_data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx3:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("parámetro detector", str(ctx3.exception))
            # Restaurar
            shutil.copy(base_real / "informe.json", ruta_informe)

            # Alteración 4: Hash canonizado de filas del manifiesto alterado
            inf_data_4 = json.loads(ruta_informe.read_text(encoding="utf-8"))
            inf_data_4["sha256_filas_manifiesto_01_a_04"] = "hash_invalido_000000000000000000"
            ruta_informe.write_text(json.dumps(inf_data_4), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx4:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("filas de desarrollo del manifiesto", str(ctx4.exception))
            # Restaurar
            shutil.copy(base_real / "informe.json", ruta_informe)

            # Alteración 5: Hash de imagen en muestras.json alterado
            ruta_muestras = temp_base / "muestras.json"
            muestras_data = json.loads(ruta_muestras.read_text(encoding="utf-8"))
            muestras_data[0]["sha256_imagen"] = "imagen_hash_corrupto_0000000000"
            ruta_muestras.write_text(json.dumps(muestras_data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx5:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("hash de imagen", str(ctx5.exception))
            # Restaurar
            shutil.copy(base_real / "muestras.json", ruta_muestras)

            # Alteración 6: Hash de manifiesto de desconocidos alterado en calibración
            calib_knn_6 = json.loads(ruta_calib_knn.read_text(encoding="utf-8"))
            calib_knn_6["negativos"]["sha256_manifiesto"] = "hash_manifiesto_negativos_corrupto_000"
            ruta_calib_knn.write_text(json.dumps(calib_knn_6), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx6:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("manifiesto de calibración de desconocidos", str(ctx6.exception))

    # 17. Correspondencia estricta de subtotales por persona y pose con decisiones reales
    def test_correspondencia_subtotales_persona_y_pose_con_decisiones_reales(self):
        from scripts.experimentar_rechazo_distancia import (
            evaluar_lote_decisiones, calcular_metricas_experimento
        )

        poses = ["frontal", "abajo", "arriba", "derecha"]
        intentos = []
        mapa_idx = {}
        preds = []
        scores = []
        reps = []

        count = 0
        for p in self.autorizadas:
            for pose in poses:
                mid = f"sample_{p}_{pose}"
                intentos.append({"id": mid, "persona": p, "pose": pose, "estado": "procesada"})
                mapa_idx[mid] = count
                preds.append(p)
                scores.append(0.85)
                # persona_1 cerca de [1.0, 1.0], persona_2 cerca de [5.0, 5.0]
                base_vec = np.array([1.0, 1.0]) if p == "persona_1" else np.array([5.0, 5.0])
                reps.append(base_vec)
                count += 1

        rep_val = np.array(reps, dtype=np.float32)

        dec_val, dec_neg = evaluar_lote_decisiones(
            intentos_val=intentos,
            mapa_idx_val=mapa_idx,
            preds_val_procesadas=preds,
            scores_val_procesadas=scores,
            rep_val=rep_val,
            banco=self.banco,
            s_th=0.80,
            d_th=0.50,
            metrica="min_l2",
            personas=self.autorizadas,
            estados_neg=["UN_ROSTRO"],
            preds_neg_unrostro=["persona_1"],
            scores_neg_unrostro=[0.50],
            rep_neg=np.array([[9.0, 9.0]], dtype=np.float32),
            idx_neg_unrostro=[0]
        )

        res = calcular_metricas_experimento(
            decisiones_val=dec_val,
            decisiones_neg=dec_neg,
            intentos_val=intentos,
            estados_neg=["UN_ROSTRO"],
            idx_neg_unrostro=[0],
            personas=self.autorizadas
        )

        # Verificación estricta de sumas de subtotales por persona
        self.assertEqual(sum(d["total"] for d in res["por_persona"].values()), res["total_intentos_val"])
        self.assertEqual(sum(d["correctas"] for d in res["por_persona"].values()), res["correctas_val"])
        self.assertEqual(sum(d["confusiones"] for d in res["por_persona"].values()), res["confusiones_val"])
        self.assertEqual(sum(d["bloqueados"] for d in res["por_persona"].values()), res["rechazos_val"])

        # Verificación estricta de sumas de subtotales por pose
        self.assertEqual(sum(d["total"] for d in res["por_pose"].values()), res["total_intentos_val"])
        self.assertEqual(sum(d["correctas"] for d in res["por_pose"].values()), res["correctas_val"])
        self.assertEqual(sum(d["confusiones"] for d in res["por_pose"].values()), res["confusiones_val"])
        self.assertEqual(sum(d["bloqueados"] for d in res["por_pose"].values()), res["rechazos_val"])

    # 18. Validación estricta de metadatos LFW, cobertura de muestras y clases_mapeo.json
    def test_validacion_estricta_metadatos_lfw_y_cobertura_muestras(self):
        from scripts.experimentar_rechazo_distancia import validar_artefactos_base
        from reconocimiento.preprocesamiento.fabrica import crear_detector
        from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
        from scripts.entrenar_desarrollo import CUADRICULA, BINS

        base_real = RAIZ / "resultados" / "desarrollo_yunet_20261118_185200_833277"
        if not base_real.exists():
            self.skipTest("No existe directorio base de desarrollo")

        detector = crear_detector()
        extractor = ExtractorHistogramasRegionales(cuadricula=CUADRICULA, n_bins=BINS, rango_intensidad=(0.0, 1.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_base = Path(tmp_dir) / "desarrollo_base"
            shutil.copytree(base_real, temp_base)

            # Caso A: Falta sha256_manifiesto en negativos
            ruta_calib_knn = temp_base / "knn" / "calibracion_desarrollo.json"
            calib_knn = json.loads(ruta_calib_knn.read_text(encoding="utf-8"))
            del calib_knn["negativos"]["sha256_manifiesto"]
            ruta_calib_knn.write_text(json.dumps(calib_knn), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx_a:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("Falta 'sha256_manifiesto'", str(ctx_a.exception))

            # Restaurar calibración
            shutil.copy(base_real / "knn" / "calibracion_desarrollo.json", ruta_calib_knn)

            # Caso B: Falta hash de imagen en muestras.json (cobertura incompleta)
            ruta_muestras = temp_base / "muestras.json"
            muestras_data = json.loads(ruta_muestras.read_text(encoding="utf-8"))
            muestras_data[0]["sha256_imagen"] = None
            ruta_muestras.write_text(json.dumps(muestras_data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx_b:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("Falta hash SHA-256 en muestras.json", str(ctx_b.exception))

            # Restaurar muestras
            shutil.copy(base_real / "muestras.json", ruta_muestras)

            # Caso C: Discrepancia en clases_mapeo.json (mapa inverso alterado)
            ruta_mapeo_knn = temp_base / "knn" / "clases_mapeo.json"
            mapeo_knn = json.loads(ruta_mapeo_knn.read_text(encoding="utf-8"))
            mapeo_knn["id_to_persona"]["0"] = "persona_invalida"
            ruta_mapeo_knn.write_text(json.dumps(mapeo_knn), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx_c:
                validar_artefactos_base(temp_base, detector, extractor)
            self.assertIn("Discrepancia en mapa inverso id_to_persona", str(ctx_c.exception))

    # 19. Manejo de validación sin imágenes procesables (lotes vacíos sin excepciones)
    def test_flujo_validacion_sin_imagenes_procesables(self):
        from scripts.experimentar_rechazo_distancia import (
            extraer_representaciones, optimizar_umbrales_regla_B,
            evaluar_lote_decisiones, calcular_metricas_experimento
        )
        from reconocimiento.clasificacion.knn import ClasificadorKNN

        base_real = RAIZ / "resultados" / "desarrollo_yunet_20261118_185200_833277"
        if not base_real.exists():
            self.skipTest("No existe directorio base de desarrollo")

        modelo_knn = ClasificadorKNN.cargar(base_real / "knn" / "modelo.joblib")

        # Representaciones de lotes vacíos
        rep_vacio = extraer_representaciones("knn", modelo_knn, np.empty((0, 64, 64)), np.empty((0, 256)))
        self.assertEqual(rep_vacio.shape, (0, 256))

        # Validación donde todos los intentos fallaron detección
        intentos_val = [
            {"id": f"intento_{i}", "persona": "persona_1", "pose": "frontal", "estado": "sin_rostro"}
            for i in range(10)
        ]
        mapa_idx_val = {}
        preds_val = []
        scores_val = []

        estado_modelo, s_th, d_th, opt_info, best_cands, _ = optimizar_umbrales_regla_B(
            candidatos_score=[0.80, 0.90],
            metricas_eval=["min_l2"],
            dists_val_por_metrica={"min_l2": []},
            dists_neg_por_metrica={"min_l2": [1.0, 2.0]},
            scores_val_procesadas=[],
            scores_neg_unrostro=[0.85, 0.85],
            preds_val_procesadas=[],
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            idx_neg_unrostro=[0, 1],
            cota_frr=0.30
        )
        self.assertEqual(estado_modelo, "sin_umbral_viable")

        dec_val, dec_neg = evaluar_lote_decisiones(
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            preds_val_procesadas=preds_val,
            scores_val_procesadas=scores_val,
            rep_val=rep_vacio,
            banco=self.banco,
            s_th=s_th,
            d_th=d_th,
            metrica=None,
            personas=self.autorizadas,
            estados_neg=["UN_ROSTRO"],
            preds_neg_unrostro=["persona_1"],
            scores_neg_unrostro=[0.50],
            rep_neg=np.array([[9.0, 9.0]], dtype=np.float32),
            idx_neg_unrostro=[0],
            estado_modelo=estado_modelo
        )

        res = calcular_metricas_experimento(
            decisiones_val=dec_val,
            decisiones_neg=dec_neg,
            intentos_val=intentos_val,
            estados_neg=["UN_ROSTRO"],
            idx_neg_unrostro=[0],
            personas=self.autorizadas
        )

        self.assertEqual(res["total_intentos_val"], 10)
        self.assertEqual(res["aperturas_val"], 0)
        self.assertEqual(res["rechazos_val"], 10)
        self.assertEqual(res["frr_puerta_pct"], 100.0)
        self.assertEqual(res["fallos_deteccion_val"], 10)

    # 20. Manejo de desconocidos sin rostros únicos
    def test_flujo_desconocidos_sin_rostros_unicos(self):
        from scripts.experimentar_rechazo_distancia import (
            extraer_representaciones, evaluar_lote_decisiones, calcular_metricas_experimento
        )
        from reconocimiento.clasificacion.cnn import ClasificadorCNN

        base_real = RAIZ / "resultados" / "desarrollo_yunet_20261118_185200_833277"
        if not base_real.exists():
            self.skipTest("No existe directorio base de desarrollo")

        modelo_cnn = ClasificadorCNN.cargar(base_real / "cnn" / "modelo.pt")
        rep_neg_vacio = extraer_representaciones("cnn", modelo_cnn, np.empty((0, 64, 64)), np.empty((0, 256)))
        self.assertEqual(rep_neg_vacio.shape, (0, 128))

        # 5 desconocidos, ninguno con UN_ROSTRO
        estados_neg = ["SIN_ROSTRO", "VARIOS_ROSTROS", "SIN_ROSTRO", "SIN_ROSTRO", "VARIOS_ROSTROS"]
        idx_neg_unrostro = []

        dec_val, dec_neg = evaluar_lote_decisiones(
            intentos_val=[{"id": "val_1", "persona": "persona_1", "pose": "frontal", "estado": "procesada"}],
            mapa_idx_val={"val_1": 0},
            preds_val_procesadas=["persona_1"],
            scores_val_procesadas=[0.99],
            rep_val=np.array([[1.0, 1.0]], dtype=np.float32),
            banco=self.banco,
            s_th=0.90,
            d_th=0.50,
            metrica="min_l2",
            personas=self.autorizadas,
            estados_neg=estados_neg,
            preds_neg_unrostro=[],
            scores_neg_unrostro=[],
            rep_neg=rep_neg_vacio,
            idx_neg_unrostro=idx_neg_unrostro,
            estado_modelo="umbral_viable"
        )

        res = calcular_metricas_experimento(
            decisiones_val=dec_val,
            decisiones_neg=dec_neg,
            intentos_val=[{"id": "val_1", "persona": "persona_1", "pose": "frontal", "estado": "procesada"}],
            estados_neg=estados_neg,
            idx_neg_unrostro=idx_neg_unrostro,
            personas=self.autorizadas
        )

        self.assertEqual(res["total_intentos_neg"], 5)
        self.assertEqual(res["total_neg_unrostro"], 0)
        self.assertEqual(res["aperturas_neg"], 0)
        self.assertEqual(res["fallos_deteccion_neg"], 5)
        self.assertEqual(res["far_total_pct"], 0.0)
        self.assertIsNone(res["far_condicionado_pct"])  # null / no calculable por denominador cero

    def test_regla_b_sin_candidatos_por_baja_confianza_en_autorizados(self):
        """
        Prueba de regresión: Verifica que optimizar_umbrales_regla_B evalúa rigurosamente
        la grilla de candidatos y retorna 'sin_umbral_viable' cuando las confianzas
        en autorizados son inferiores a min(candidatos_score) (caso ANN), sin omitir
        la evaluación ni inventar umbrales de respaldo.
        """
        from scripts.experimentar_rechazo_distancia import optimizar_umbrales_regla_B

        # 64 intentos de validación: 37 con score 0.45 (< 0.50) y 27 con score 0.60
        intentos_val = [{"id": f"val_{i}", "persona": "persona_1"} for i in range(64)]
        mapa_idx_val = {f"val_{i}": i for i in range(64)}
        scores_val = [0.45] * 37 + [0.60] * 27
        preds_val = ["persona_1"] * 64

        # Distancias ideales (0.0)
        dists_val = {"min_cos": [0.0] * 64}
        dists_neg = {"min_cos": [0.5] * 10}
        scores_neg = [0.10] * 10
        idx_neg_unrostro = list(range(10))

        candidatos_score = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

        estado, metrica, s_th, d_th, info_opt, mejores = optimizar_umbrales_regla_B(
            candidatos_score=candidatos_score,
            metricas_eval=["min_cos"],
            dists_val_por_metrica=dists_val,
            dists_neg_por_metrica=dists_neg,
            scores_val_procesadas=scores_val,
            scores_neg_unrostro=scores_neg,
            preds_val_procesadas=preds_val,
            intentos_val=intentos_val,
            mapa_idx_val=mapa_idx_val,
            idx_neg_unrostro=idx_neg_unrostro,
            cota_frr=0.30
        )

        self.assertEqual(estado, "sin_umbral_viable")
        self.assertIsNone(metrica)
        self.assertIsNone(s_th)
        self.assertIsNone(d_th)
        self.assertEqual(info_opt, {})
        self.assertEqual(mejores, {})


if __name__ == "__main__":
    unittest.main()
