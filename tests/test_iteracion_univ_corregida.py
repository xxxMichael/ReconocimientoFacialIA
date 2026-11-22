import csv
import json
from pathlib import Path
import sys
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.preprocesamiento.detector_yunet import DetectorYuNet
from scripts.experimentar_4clases_universidad import cargar_manifiesto_universidad_desarrollo
from scripts.experimentar_cnn_univ_corregida import (
    RUTA_REVISION_UNIV,
    cargar_universidad_train_con_revisiones,
)


class TestIteracionUnivCorregida(unittest.TestCase):
    def test_01_lectura_revisiones_y_rescate_sin_inventar_landmarks(self):
        self.assertTrue(RUTA_REVISION_UNIV.exists(), "Debe existir revision_universidad.json")
        train_filas, val_filas = cargar_manifiesto_universidad_desarrollo()
        self.assertEqual(len(train_filas), 18)
        self.assertEqual(len(val_filas), 7)

        det = DetectorYuNet(modo_multiorientacion="cascada", alinear_rostro=True)
        ext = ExtractorHistogramasRegionales(cuadricula=(4, 4), n_bins=16)

        _, imgs_64, traz = cargar_universidad_train_con_revisiones(train_filas, det, ext)
        self.assertEqual(len(imgs_64), 18, "Las 18 imágenes de entrenamiento deben procesarse al tener cajas aprobadas")

        por_id = {t["id"]: t for t in traz}
        for id_rescatado in ("universidad_014", "universidad_021", "universidad_028"):
            self.assertTrue(por_id[id_rescatado]["procesada"])
            self.assertFalse(por_id[id_rescatado]["alineado"], "No debe inventar landmarks si YuNet falló")
            self.assertEqual(por_id[id_rescatado]["metodo_prep"], "fallback_recorte_seguro")

        # Las que sí tienen landmarks válidos dentro de la caja conservan alineación 5-puntos
        self.assertTrue(por_id["universidad_001"]["alineado"])
        self.assertEqual(por_id["universidad_001"]["metodo_prep"], "similitud_5puntos")

    def test_02_particiones_aisladas_y_sin_prueba_reservada(self):
        train_filas, val_filas = cargar_manifiesto_universidad_desarrollo()
        ids_dev = {f["identificador"] for f in train_filas + val_filas}
        with (RAIZ / "data" / "universidad_desc" / "manifiesto_universidad.csv").open(encoding="utf-8", newline="") as f:
            ids_res = {r["identificador"] for r in csv.DictReader(f) if r["particion"] == "prueba_reservada"}
        self.assertEqual(len(ids_res), 9)
        self.assertEqual(len(ids_dev & ids_res), 0)

    def test_03_concordancia_auditoria_y_ecuacion_autorizados(self):
        dir_exp = RAIZ / "resultados" / "exp_4clases_univ_corregida_20261120_213750"
        ruta_res = dir_exp / "resumen_iteracion_univ_corregida.json"
        ruta_csv = dir_exp / "auditoria_reconstruida" / "decisiones_individuales_reconstruidas.csv"
        if not ruta_res.exists() or not ruta_csv.exists():
            self.skipTest("Artefactos de auditoría no presentes.")

        res = json.loads(ruta_res.read_text(encoding="utf-8"))
        with ruta_csv.open(encoding="utf-8", newline="") as f:
            filas = list(csv.DictReader(f))

        for etiqueta, key_json in (
            ("referencia_portal_completo_061", "validacion_automatica_referencia_061"),
            ("nueva_cnn_univ_corregida_057", "validacion_automatica_nueva"),
        ):
            rb = res[key_json]["regla_b"]
            auth = [x for x in filas if x["modelo"] == etiqueta and x["fuente"] == "autorizado_sesion_04"]
            desc = [x for x in filas if x["modelo"] == etiqueta and x["fuente"].startswith("desconocido")]

            aciertos = sum(1 for x in auth if x["categoria"] == "identificacion_correcta")
            confusiones = sum(1 for x in auth if x["categoria"] == "confusion_autorizado")
            bloqueos = sum(1 for x in auth if x["decision_puerta"] == "CERRADA")
            self.assertEqual(len(auth), aciertos + confusiones + bloqueos)
            self.assertEqual(aciertos, rb["aperturas_correctas"])
            self.assertEqual(confusiones, rb["confusiones_autorizados"])
            self.assertEqual(bloqueos, rb["bloqueos_puerta"])

            fa = sum(1 for x in desc if x["categoria"] == "falsa_aceptacion")
            rech_d = sum(1 for x in desc if x["categoria"] == "rechazo_desconocido")
            rech_u = sum(1 for x in desc if x["categoria"] == "rechazo_umbral")
            fallos = sum(1 for x in desc if x["categoria"] == "fallo_deteccion")
            self.assertEqual(len(desc), fa + rech_d + rech_u + fallos)
            self.assertEqual(rech_d, 91)
            self.assertEqual(fallos, 16)


if __name__ == "__main__":
    unittest.main()
