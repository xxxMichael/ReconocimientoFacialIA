import csv
import json
from pathlib import Path
import sys
import unittest
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from scripts.experimentar_4clases_portal_completo import calcular_pesos_identidad_y_clase
from scripts.experimentar_4clases_universidad import cargar_manifiesto_universidad_desarrollo


class TestExperimentoPortalCompleto(unittest.TestCase):
    def test_01_pesos_dos_niveles_equilibran_clases_e_identidades(self):
        # Simular: 3 clases autorizadas (10, 5, 5 muestras) y 3 identidades desconocidas
        # (u_lfw con 1 foto, u_univ con 3 muestras [1+2 aug], u_portal con 15 fotos)
        y = np.array(
            [0] * 10 + [1] * 5 + [2] * 5 + [3] * (1 + 3 + 15),
            dtype=np.int64,
        )
        ids = (
            ["persona_1"] * 10
            + ["persona_2"] * 5
            + ["persona_3"] * 5
            + ["lfw_1"] * 1
            + ["univ_1"] * 3
            + ["portal_1"] * 15
        )
        fuentes = (
            ["autorizado_persona_1"] * 10
            + ["autorizado_persona_2"] * 5
            + ["autorizado_persona_3"] * 5
            + ["desconocido_lfw"] * 1
            + ["desconocido_universidad"] * 3
            + ["desconocido_portal"] * 15
        )

        pesos, resumen = calcular_pesos_identidad_y_clase(y, ids, fuentes)

        # 1. Cada clase suma exactamente 25% del peso efectivo total (N / 4)
        for c_nom, c_inf in resumen["clases"].items():
            self.assertAlmostEqual(c_inf["proporcion_pct"], 25.0, places=1, msg=f"Clase {c_nom} desbalanceada")

        # 2. Dentro de desconocido, las 3 identidades contribuyen lo mismo independientemente de 1 vs 3 vs 15 fotos
        w_lfw = float(np.sum(pesos[20:21]))
        w_univ = float(np.sum(pesos[21:24]))
        w_portal = float(np.sum(pesos[24:39]))
        self.assertAlmostEqual(w_lfw, w_univ, places=4)
        self.assertAlmostEqual(w_lfw, w_portal, places=4)

    def test_02_exclusion_reserva_universitaria(self):
        train_filas, val_filas = cargar_manifiesto_universidad_desarrollo()
        self.assertEqual(len(train_filas), 18)
        self.assertEqual(len(val_filas), 7)
        for r in train_filas + val_filas:
            self.assertNotEqual(r["particion"], "prueba_reservada")

    def test_03_soporte_sample_weight_en_arbol_y_svm(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 8).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10, dtype=np.int32)
        sw = rng.uniform(0.5, 2.0, size=40).astype(np.float32)

        arb = ClasificadorArbol(max_depth=3, class_weight=None, semilla=42)
        arb.entrenar(X, y, sample_weight=sw)
        self.assertEqual(len(arb.clases_), 4)

        svm = ClasificadorSVM(C=1.0, kernel="linear", class_weight=None, semilla=42)
        svm.entrenar(X, y, sample_weight=sw)
        self.assertEqual(len(svm.clases_), 4)

    def test_04_verificar_resultados_si_existen(self):
        dirs = sorted((RAIZ / "resultados").glob("exp_4clases_portal_completo_*"))
        if not dirs:
            self.skipTest("Experimento aún no ejecutado.")
        ultimo = dirs[-1]
        resumen_p = ultimo / "resumen_experimento_portal_completo.json"
        man_p = ultimo / "manifiesto_entrenamiento_experimento.csv"
        if not resumen_p.exists() or not man_p.exists():
            self.skipTest("Artefactos aún incompletos.")

        resumen = json.loads(resumen_p.read_text(encoding="utf-8"))
        # Verificar que todas las poses del portal fueron cubiertas en el manifiesto
        with man_p.open(encoding="utf-8", newline="") as f:
            filas = list(csv.DictReader(f))
        portal_ids = {r["id"] for r in filas if r["fuente"] == "desconocido_portal"}
        self.assertGreaterEqual(len(portal_ids), 70, "Deben cubrirse todas las fotos válidas del portal (72/75)")
        for r in filas:
            self.assertEqual(r["particion"], "entrenamiento")

        # Verificar carga de configuración propuesta si existe
        cfg_p = ultimo / "configuracion_candidato_portal_completo_propuesto.json"
        if cfg_p.exists():
            pipe = PipelineCandidatoCongelado(cfg_p, raiz_proyecto=RAIZ)
            self.assertTrue(pipe.usar_alineacion)
            self.assertEqual(pipe.orden_clases, ["persona_1", "persona_2", "persona_3", "desconocido"])


if __name__ == "__main__":
    unittest.main()
