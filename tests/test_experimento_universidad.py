import csv
import json
from pathlib import Path
import sys
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from scripts.experimentar_4clases_universidad import (
    MANIFIESTO_UNIVERSIDAD,
    cargar_manifiesto_universidad_desarrollo,
)
from scripts.preparar_particiones_universidad import SEMILLA_PARTICION


class TestExperimentoUniversidad(unittest.TestCase):
    def test_01_particiones_sin_solapamiento_y_conteos_exactos(self):
        self.assertTrue(MANIFIESTO_UNIVERSIDAD.exists(), "Debe existir manifiesto_universidad.csv")
        with MANIFIESTO_UNIVERSIDAD.open(encoding="utf-8", newline="") as f:
            filas = list(csv.DictReader(f))

        # 34 archivos reales (18 train, 7 val, 9 prueba_reservada)
        self.assertGreaterEqual(len(filas), 33)

        ids_por_part = {}
        hashes_por_part = {}
        for r in filas:
            part = r["particion"]
            ids_por_part.setdefault(part, set()).add(r["identificador"])
            hashes_por_part.setdefault(part, set()).add(r["hash_sha256"])

        self.assertEqual(len(ids_por_part.get("entrenamiento", set())), 18)
        self.assertEqual(len(ids_por_part.get("validacion", set())), 7)
        self.assertGreaterEqual(len(ids_por_part.get("prueba_reservada", set())), 8)

        # Cero solapamiento entre particiones
        t_h = hashes_por_part["entrenamiento"]
        v_h = hashes_por_part["validacion"]
        p_h = hashes_por_part["prueba_reservada"]
        self.assertEqual(len(t_h & v_h), 0)
        self.assertEqual(len(t_h & p_h), 0)
        self.assertEqual(len(v_h & p_h), 0)

    def test_02_exclusion_estricta_de_prueba_reservada_en_cargador(self):
        train_filas, val_filas = cargar_manifiesto_universidad_desarrollo()
        self.assertEqual(len(train_filas), 18)
        self.assertEqual(len(val_filas), 7)

        for r in train_filas + val_filas:
            self.assertNotEqual(r["particion"], "prueba_reservada")
            self.assertIn(r["particion"], ("entrenamiento", "validacion"))

    def test_03_concordancia_decisiones_y_metricas_si_existe_resultado(self):
        dirs = sorted((RAIZ / "resultados").glob("exp_4clases_universidad_*"))
        if not dirs:
            self.skipTest("Aún no se ha ejecutado el experimento universidad.")
        ultimo = dirs[-1]
        resumen_path = ultimo / "resumen_experimento_universidad.json"
        dec_path = ultimo / "decisiones_individuales_validacion.csv"
        if not resumen_path.exists() or not dec_path.exists():
            self.skipTest("Artefactos del experimento aún incompletos.")

        resumen = json.loads(resumen_path.read_text(encoding="utf-8"))
        with dec_path.open(encoding="utf-8", newline="") as f:
            decisiones = list(csv.DictReader(f))

        # Verificar concordancia de aumentos limitados a entrenamiento (validacion sin aumentos)
        # 64 autorizados + 100 LFW + 7 Univ = 171 registros por modelo en validación
        for nm in ("arbol", "knn", "svm", "ann", "cnn"):
            decs_m = [d for d in decisiones if d["modelo"] == f"nuevo_{nm}"]
            self.assertEqual(len(decs_m), 171, f"Se esperaban 171 decisiones de validación para {nm}")

            cal = resumen["calibracion_nuevos"][nm]
            regla = cal["regla_b"] if cal["regla_b_habilitada"] else cal["regla_a"]

            fa_csv = sum(1 for d in decs_m if d["fuente"].startswith("desconocido") and d["decision"] == "ABIERTA")
            fa_json = regla["far_lfw"]["falsas_aceptaciones"] + regla["far_universidad"]["falsas_aceptaciones"]
            self.assertEqual(fa_csv, fa_json)

            bloq_auth_csv = sum(1 for d in decs_m if d["fuente"] == "autorizado_sesion_04" and d["decision"] == "CERRADA")
            self.assertEqual(bloq_auth_csv, regla["bloqueos_puerta"])


if __name__ == "__main__":
    unittest.main()
