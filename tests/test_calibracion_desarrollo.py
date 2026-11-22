import unittest

from scripts.calibrar_desarrollo import Intento, elegir_umbral, medir


CLASES = ["persona_1", "persona_2"]


class TestCalibracionDesarrollo(unittest.TestCase):
    def test_igualdad_fallos_deteccion_y_confusion(self):
        positivos = [
            Intento("procesada", "persona_1", "persona_1", 0.80),
            Intento("procesada", "persona_1", "persona_2", 0.80),
            Intento("sin_rostro", "persona_2"),
            Intento("procesada", "persona_2", "persona_2", 0.79),
        ]
        negativos = [
            Intento("UN_ROSTRO", persona_predicha="persona_1", puntaje=0.80),
            Intento("SIN_ROSTRO"),
            Intento("VARIOS_ROSTROS"),
            Intento("UN_ROSTRO", persona_predicha="persona_2", puntaje=0.79),
        ]

        fila = medir(0.80, positivos, negativos, CLASES)

        self.assertEqual(fila["far_pct"], 25.0)
        self.assertEqual(fila["rechazo_correcto"], 3)
        self.assertEqual(fila["frr_puerta_pct"], 50.0)
        self.assertEqual(fila["aperturas_autorizados"], 2)
        self.assertEqual(fila["identificaciones_correctas"], 1)
        self.assertEqual(fila["confusiones_entre_autorizados"], 1)
        self.assertEqual(fila["matriz_identificacion_entre_aperturas"], [[1, 1], [0, 0]])

    def test_sin_umbral_cuando_todos_superan_cota(self):
        positivos = [Intento("sin_rostro", "persona_1")]
        negativos = [Intento("VARIOS_ROSTROS")]
        curva = [medir(u, positivos, negativos, CLASES) for u in (0.50, 1.00)]
        self.assertIsNone(elegir_umbral(curva, len(positivos)))

    def test_minimo_far_y_mayor_umbral_con_empate(self):
        positivos = [Intento("procesada", "persona_1", "persona_1", 0.90)]
        negativos = [Intento("UN_ROSTRO", persona_predicha="persona_1", puntaje=0.60)]
        curva = [medir(u, positivos, negativos, CLASES) for u in (0.50, 0.70, 0.90, 0.91)]
        self.assertEqual(elegir_umbral(curva, len(positivos))["umbral"], 0.90)

    def test_ann_sin_umbral_viable_por_baja_confianza_en_autorizados(self):
        """
        Prueba de regresión: Reproduce el escenario real de ANN en desarrollo donde 37 de 64
        intentos autorizados tienen puntaje < 0.50.
        Verifica que la ausencia de umbral en la grilla [0.50, 1.00] se debe estrictamente
        al incumplimiento de la cota de FRR <= 30% sobre autorizados, no a los negativos.
        """
        # 64 positivos: 37 con score 0.45 (< 0.50) y 27 con score 0.60 (>= 0.50)
        positivos = (
            [Intento("procesada", "persona_1", "persona_1", 0.45) for _ in range(37)] +
            [Intento("procesada", "persona_1", "persona_1", 0.60) for _ in range(27)]
        )
        # 100 negativos (incluso con FAR = 0% en umbral 0.50)
        negativos = [Intento("UN_ROSTRO", persona_predicha="persona_1", puntaje=0.10) for _ in range(100)]

        curva = [medir(paso / 100, positivos, negativos, CLASES) for paso in range(50, 101)]

        # En umbral 0.50, rechaza 37 de 64 autorizados (FRR = 37/64 = 57.81% > 30%)
        medida_050 = next(f for f in curva if f["umbral"] == 0.50)
        self.assertEqual(medida_050["rechazos_autorizados"], 37)
        self.assertAlmostEqual(medida_050["frr_puerta_pct"], 57.8125, places=3)
        self.assertGreater(medida_050["rechazos_autorizados"] * 10, 3 * len(positivos))

        # En ningún umbral >= 0.50 se cumple FRR <= 30%
        self.assertTrue(all(f["rechazos_autorizados"] * 10 > 3 * len(positivos) for f in curva))

        # elegir_umbral debe retornar None sin inventar respaldo
        elegido = elegir_umbral(curva, len(positivos))
        self.assertIsNone(elegido)
