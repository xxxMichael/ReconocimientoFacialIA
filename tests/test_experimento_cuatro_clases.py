"""Pruebas unitarias para el experimento controlado de clasificación con 4 clases."""

import csv
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]

from scripts.seleccionar_desconocidos_4clases import normalizar_identidad
from scripts.experimentar_4clases import calcular_tasa_pct
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.regla_acceso import (
    clave_ordenamiento_jerarquico,
    seleccionar_umbral_optimo_jerarquico,
)
from reconocimiento.clasificacion.svm import ClasificadorSVM


class TestExperimentoCuatroClases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir_desconocidos_train = RAIZ / "data" / "desconocidos_lfw" / "entrenamiento_4clases"
        cls.manifiesto_train = cls.dir_desconocidos_train / "manifiesto.csv"
        cls.manifiesto_calib = RAIZ / "data" / "desconocidos_lfw" / "calibracion" / "manifiesto.csv"
        cls.manifiesto_reserva = RAIZ / "data" / "desconocidos_lfw" / "reserva_prueba" / "manifiesto.csv"

        # Buscar el directorio de resultados corregido más reciente de 4 clases
        dirs = sorted(list((RAIZ / "resultados").glob("desarrollo_4clases_corregido_*")))
        if not dirs:
            dirs = sorted(list((RAIZ / "resultados").glob("desarrollo_4clases_*")))
        if not dirs:
            raise FileNotFoundError("No se encontró ningún directorio de resultados desarrollo_4clases_*")
        cls.dir_resultados = dirs[-1]

    def test_01_aislamiento_estricto_identidades_lfw(self):
        """Verifica que ninguna de las 300 identidades de train solape con calibración ni reserva."""
        self.assertTrue(self.manifiesto_train.exists(), "Falta el manifiesto de desconocidos de train")
        with self.manifiesto_train.open(encoding="utf-8") as f:
            filas_tr = list(csv.DictReader(f))
        self.assertEqual(len(filas_tr), 300, "Deben ser exactamente 300 desconocidos de entrenamiento")

        identidades_tr = {normalizar_identidad(r["identidad_lfw"]) for r in filas_tr}
        hashes_tr = {r["sha256"].lower().strip() for r in filas_tr}

        # Calibración
        with self.manifiesto_calib.open(encoding="utf-8") as f:
            filas_calib = list(csv.DictReader(f))
        identidades_calib = {normalizar_identidad(r["identidad_lfw"]) for r in filas_calib}
        hashes_calib = {r["sha256"].lower().strip() for r in filas_calib}

        # Reserva de prueba (solo leemos manifiesto para verificar disyunción, sin tocar imágenes)
        with self.manifiesto_reserva.open(encoding="utf-8") as f:
            filas_reserva = list(csv.DictReader(f))
        identidades_reserva = {normalizar_identidad(r["identidad_lfw"]) for r in filas_reserva}
        hashes_reserva = {r["sha256"].lower().strip() for r in filas_reserva}

        # Comprobación de disyunción absoluta
        solap_calib_id = identidades_tr.intersection(identidades_calib)
        solap_reserva_id = identidades_tr.intersection(identidades_reserva)
        solap_calib_h = hashes_tr.intersection(hashes_calib)
        solap_reserva_h = hashes_tr.intersection(hashes_reserva)

        self.assertEqual(len(solap_calib_id), 0, f"Solapamiento de identidades con calibración: {solap_calib_id}")
        self.assertEqual(len(solap_reserva_id), 0, f"Solapamiento de identidades con reserva: {solap_reserva_id}")
        self.assertEqual(len(solap_calib_h), 0, f"Solapamiento de hashes con calibración: {solap_calib_h}")
        self.assertEqual(len(solap_reserva_h), 0, f"Solapamiento de hashes con reserva: {solap_reserva_h}")

    def test_02_desconocido_nunca_en_personas_autorizadas(self):
        """Verifica que 'desconocido' no esté en personas_autorizadas.json."""
        ruta_pa = self.dir_resultados / "personas_autorizadas.json"
        self.assertTrue(ruta_pa.exists(), "Falta personas_autorizadas.json")
        autorizadas = json.loads(ruta_pa.read_text(encoding="utf-8"))
        self.assertNotIn("desconocido", autorizadas)
        self.assertListEqual(autorizadas, ["persona_1", "persona_2", "persona_3"])

    def test_03_mapeo_clases_cuatro_salidas(self):
        """Verifica que clases_mapeo.json defina exactamente 4 clases con índice numérico."""
        ruta_cm = self.dir_resultados / "clases_mapeo.json"
        self.assertTrue(ruta_cm.exists(), "Falta clases_mapeo.json")
        mapeo = json.loads(ruta_cm.read_text(encoding="utf-8"))
        self.assertEqual(len(mapeo), 4)
        self.assertEqual(mapeo["0"], "persona_1")
        self.assertEqual(mapeo["1"], "persona_2")
        self.assertEqual(mapeo["2"], "persona_3")
        self.assertEqual(mapeo["3"], "desconocido")

    def test_04_carga_e_inferencia_cuatro_clases_modelos(self):
        """Verifica que los 5 modelos guardados produzcan 4 clases de salida en inferencia."""
        X_dummy_tex = np.ones((1, 256), dtype=np.float32)
        X_dummy_img = np.ones((1, 1, 64, 64), dtype=np.float32)

        knn = ClasificadorKNN.cargar(self.dir_resultados / "knn" / "modelo.joblib")
        probs_knn = knn.predecir_probabilidades(X_dummy_tex)
        self.assertEqual(probs_knn.shape, (1, 4))
        self.assertAlmostEqual(float(np.sum(probs_knn)), 1.0, places=4)

        arbol = ClasificadorArbol.cargar(self.dir_resultados / "arbol" / "modelo.joblib")
        probs_arbol = arbol.predecir_probabilidades(X_dummy_tex)
        self.assertEqual(probs_arbol.shape, (1, 4))

        svm = ClasificadorSVM.cargar(self.dir_resultados / "svm" / "modelo.joblib")
        probs_svm = svm.predecir_probabilidades(X_dummy_tex)
        self.assertEqual(probs_svm.shape, (1, 4))

        ann = ClasificadorANN.cargar(self.dir_resultados / "ann" / "modelo.pt")
        probs_ann = ann.predecir_probabilidades(X_dummy_tex)
        self.assertEqual(probs_ann.shape, (1, 4))
        self.assertAlmostEqual(float(np.sum(probs_ann)), 1.0, places=4)

        cnn = ClasificadorCNN.cargar(self.dir_resultados / "cnn" / "modelo.pt")
        probs_cnn = cnn.predecir_probabilidades(X_dummy_img)
        self.assertEqual(probs_cnn.shape, (1, 4))
        self.assertAlmostEqual(float(np.sum(probs_cnn)), 1.0, places=4)

    def test_05_separacion_estricta_apertura_confusion_bloqueo(self):
        """
        Verifica que una persona autorizada aceptada como otra autorizada
        cuenta como apertura y confusión, pero NUNCA como bloqueo.
        """
        ruta_info = self.dir_resultados / "informe_experimento.json"
        info = json.loads(ruta_info.read_text(encoding="utf-8"))

        for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
            res = info["resultados"][nombre]
            ra = res["regla_a"]

            # Comprobar que aperturas_totales = aperturas_correctas + confusiones
            self.assertEqual(
                ra["aperturas_totales"],
                ra["aperturas_correctas"] + ra["confusiones_autorizados"],
                f"Fallo en aperturas_totales para {nombre}"
            )
            # Comprobar que bloqueos = intentos - aperturas_totales
            self.assertEqual(
                ra["bloqueos_puerta"],
                64 - ra["aperturas_totales"],
                f"Fallo en bloqueos_puerta para {nombre}"
            )

    def test_06_invariante_intentos_suma_bloqueos_aperturas_confusiones(self):
        """
        Verifica la invariante obligatoria:
        intentos = bloqueos + aperturas_correctas + confusiones
        tanto a nivel global como en cada persona y en cada pose.
        """
        ruta_info = self.dir_resultados / "informe_experimento.json"
        info = json.loads(ruta_info.read_text(encoding="utf-8"))

        for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
            for regla_k in ("regla_a", "regla_b"):
                regla = info["resultados"][nombre][regla_k]
                if regla.get("estado") == "sin_umbral_viable":
                    continue

                # Nivel global
                intentos_tot = len(regla["intentos_auth"])
                self.assertEqual(
                    intentos_tot,
                    regla["bloqueos_puerta"] + regla["aperturas_correctas"] + regla["confusiones_autorizados"]
                )

                # Nivel persona
                for p, p_stats in regla["desglose_por_persona"].items():
                    self.assertEqual(
                        p_stats["intentos"],
                        p_stats["bloqueos"] + p_stats["aperturas_correctas"] + p_stats["confusiones"],
                        f"Invariante rota para persona {p} en {nombre} {regla_k}"
                    )

                # Nivel pose
                for pose, pose_stats in regla["desglose_por_pose"].items():
                    self.assertEqual(
                        pose_stats["intentos"],
                        pose_stats["bloqueos"] + pose_stats["aperturas_correctas"] + pose_stats["confusiones"],
                        f"Invariante rota para pose {pose} en {nombre} {regla_k}"
                    )

    def test_07_coincidencia_desgloses_con_totales(self):
        """Verifica que la suma de los desgloses por persona y pose coincida con los totales globales."""
        ruta_info = self.dir_resultados / "informe_experimento.json"
        info = json.loads(ruta_info.read_text(encoding="utf-8"))

        for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
            ra = info["resultados"][nombre]["regla_a"]

            # Suma por persona
            suma_corr_p = sum(s["aperturas_correctas"] for s in ra["desglose_por_persona"].values())
            suma_conf_p = sum(s["confusiones"] for s in ra["desglose_por_persona"].values())
            suma_bloq_p = sum(s["bloqueos"] for s in ra["desglose_por_persona"].values())
            self.assertEqual(suma_corr_p, ra["aperturas_correctas"])
            self.assertEqual(suma_conf_p, ra["confusiones_autorizados"])
            self.assertEqual(suma_bloq_p, ra["bloqueos_puerta"])

            # Suma por pose
            suma_corr_pose = sum(s["aperturas_correctas"] for s in ra["desglose_por_pose"].values())
            suma_conf_pose = sum(s["confusiones"] for s in ra["desglose_por_pose"].values())
            suma_bloq_pose = sum(s["bloqueos"] for s in ra["desglose_por_pose"].values())
            self.assertEqual(suma_corr_pose, ra["aperturas_correctas"])
            self.assertEqual(suma_conf_pose, ra["confusiones_autorizados"])
            self.assertEqual(suma_bloq_pose, ra["bloqueos_puerta"])

    def test_08_criterio_desempate_umbrales_jerarquico(self):
        """
        Prueba unitaria que invoca la función real reutilizable
        seleccionar_umbral_optimo_jerarquico para verificar el orden estricto:
        1. Menor FAR
        2. Menor FRR
        3. Mayor aperturas correctas
        4. Mayor umbral
        """
        # Caso 1: Menor FAR gana independientemente de FRR y aperturas
        cand_far = [
            {"umbral": 0.80, "far_total_pct": 2.0, "frr_puerta_pct": 0.0, "aperturas_correctas": 64},
            {"umbral": 0.40, "far_total_pct": 1.0, "frr_puerta_pct": 10.0, "aperturas_correctas": 50},
        ]
        ganador_far = seleccionar_umbral_optimo_jerarquico(cand_far)
        self.assertEqual(ganador_far["umbral"], 0.40)

        # Caso 2: Ante igual FAR, gana menor FRR de puerta
        cand_frr = [
            {"umbral": 0.79, "far_total_pct": 1.0, "frr_puerta_pct": 12.5, "aperturas_correctas": 56},
            {"umbral": 0.46, "far_total_pct": 1.0, "frr_puerta_pct": 0.0, "aperturas_correctas": 64},
        ]
        ganador_frr = seleccionar_umbral_optimo_jerarquico(cand_frr)
        self.assertEqual(ganador_frr["umbral"], 0.46)

        # Caso 3: Ante igual FAR e igual FRR, gana mayor aperturas correctas
        cand_corr = [
            {"umbral": 0.50, "far_total_pct": 1.0, "frr_puerta_pct": 5.0, "aperturas_correctas": 60},
            {"umbral": 0.45, "far_total_pct": 1.0, "frr_puerta_pct": 5.0, "aperturas_correctas": 62},
        ]
        ganador_corr = seleccionar_umbral_optimo_jerarquico(cand_corr)
        self.assertEqual(ganador_corr["umbral"], 0.45)

        # Caso 4: Ante empate total en FAR, FRR y aperturas correctas, desempata mayor umbral
        cand_th = [
            {"umbral": 0.45, "far_total_pct": 1.0, "frr_puerta_pct": 0.0, "aperturas_correctas": 64},
            {"umbral": 0.55, "far_total_pct": 1.0, "frr_puerta_pct": 0.0, "aperturas_correctas": 64},
            {"umbral": 0.50, "far_total_pct": 1.0, "frr_puerta_pct": 0.0, "aperturas_correctas": 64},
        ]
        ganador_th = seleccionar_umbral_optimo_jerarquico(cand_th)
        self.assertEqual(ganador_th["umbral"], 0.55)

        # Caso 5: Lista vacía retorna None
        self.assertIsNone(seleccionar_umbral_optimo_jerarquico([]))

    def test_09_manejo_denominadores_vacios_y_sin_umbral_viable(self):
        """Verifica que calcular_tasa_pct retorne None ante denominador 0 y que sin_umbral_viable sea reportado."""
        self.assertIsNone(calcular_tasa_pct(5, 0))
        self.assertIsNone(calcular_tasa_pct(0, 0))
        self.assertEqual(calcular_tasa_pct(1, 100), 1.0)

        # SVM debe reportar sin_umbral_viable y umbral None
        ruta_info = self.dir_resultados / "informe_experimento.json"
        info = json.loads(ruta_info.read_text(encoding="utf-8"))
        rb_svm = info["resultados"]["svm"]["regla_b"]
        self.assertEqual(rb_svm.get("estado"), "sin_umbral_viable")
        self.assertIsNone(rb_svm.get("umbral"))

    def test_10_coherencia_metricas_con_decisiones_individuales(self):
        """Verifica que las métricas agregadas coincidan exactamente con la suma de decisiones individuales."""
        ruta_info = self.dir_resultados / "informe_experimento.json"
        info = json.loads(ruta_info.read_text(encoding="utf-8"))

        for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
            ra = info["resultados"][nombre]["regla_a"]
            intentos_auth = ra["intentos_auth"]
            intentos_desc = ra["intentos_desc"]

            # Contar decisiones en listas individuales
            n_corr = sum(1 for item in intentos_auth if item["decision"] == "apertura_correcta")
            n_conf = sum(1 for item in intentos_auth if item["decision"] == "apertura_confundida")
            n_bloq = sum(1 for item in intentos_auth if item["decision"].startswith("bloqueo"))

            self.assertEqual(n_corr, ra["aperturas_correctas"])
            self.assertEqual(n_conf, ra["confusiones_autorizados"])
            self.assertEqual(n_bloq, ra["bloqueos_puerta"])

            n_fa = sum(1 for item in intentos_desc if item["decision"] == "falsa_aceptacion")
            fa_conteo_esperado = int(ra["far_total_conteo"].split("/")[0])
            self.assertEqual(n_fa, fa_conteo_esperado)

    def test_11_modelos_intactos_hashes_sha256(self):
        """Verifica que los modelos coincidan byte a byte con los hashes registrados."""
        ruta_info = self.dir_resultados / "informe_experimento.json"
        info = json.loads(ruta_info.read_text(encoding="utf-8"))
        hashes_registrados = info["sha256_modelos"]

        for nombre in ("knn", "arbol", "svm", "ann", "cnn"):
            arch = "modelo.pt" if nombre in {"ann", "cnn"} else "modelo.joblib"
            ruta_m = self.dir_resultados / nombre / arch
            self.assertTrue(ruta_m.exists(), f"Falta modelo {ruta_m}")
            h_calculado = hashlib.sha256(ruta_m.read_bytes()).hexdigest()
            self.assertEqual(h_calculado, hashes_registrados[nombre], f"El modelo {nombre} fue modificado")


if __name__ == "__main__":
    unittest.main()
