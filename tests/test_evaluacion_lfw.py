"""
Pruebas unitarias para la evaluación externa con LFW y equivalencia con la web.
"""
import csv
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from reconocimiento.clasificacion.cnn import ClasificadorCNN
from reconocimiento.evaluacion.lfw import (
    calcular_sha256,
    reproducir_decision_produccion,
    verificar_y_congelar_configuracion,
)
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
from reconocimiento.web.streamer import GestorCamaraWeb


class TestEvaluacionLFW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir_resultados = Path("resultados/evaluacion_externa_lfw_cnn")
        cls.ruta_manifiesto = cls.dir_resultados / "manifiesto_lfw_100.csv"
        cls.ruta_predicciones = cls.dir_resultados / "predicciones.csv"
        cls.ruta_config = cls.dir_resultados / "configuracion_congelada.json"
        cls.ruta_informe = cls.dir_resultados / "informe_evaluacion_lfw.md"

    def test_integridad_artefactos_generados(self):
        """Verifica que los 4 artefactos de auditoría existan y sean válidos."""
        self.assertTrue(self.ruta_config.exists(), "Falta configuracion_congelada.json")
        self.assertTrue(self.ruta_manifiesto.exists(), "Falta manifiesto_lfw_100.csv")
        self.assertTrue(self.ruta_predicciones.exists(), "Falta predicciones.csv")
        self.assertTrue(self.ruta_informe.exists(), "Falta informe_evaluacion_lfw.md")

    def test_manifiesto_100_muestras_unicas(self):
        """Verifica que el manifiesto contenga exactamente 100 personas distintas y 100 hashes válidos."""
        self.assertTrue(self.ruta_manifiesto.exists())
        with open(self.ruta_manifiesto, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            filas = list(reader)

        self.assertEqual(len(filas), 100, "El manifiesto debe contener exactamente 100 registros")

        identidades = [row["identidad_lfw"] for row in filas]
        self.assertEqual(len(set(identidades)), 100, "Las 100 identidades de LFW deben ser estrictamente distintas")

        hashes = [row["sha256"] for row in filas]
        self.assertEqual(len(set(hashes)), 100, "Los 100 hashes de imagen deben ser distintos")

        # Comprobar que los archivos de imagen guardados existen y su hash coincide
        for i in [0, 25, 50, 75, 99]:
            r_img = Path(filas[i]["ruta_imagen"])
            self.assertTrue(r_img.exists(), f"La imagen {r_img} debe existir en disco")
            hash_calculado = calcular_sha256(r_img)
            self.assertEqual(hash_calculado, filas[i]["sha256"], f"El hash de {r_img} debe coincidir")

    def test_predicciones_balance_estricto_100(self):
        """Verifica que aperturas y cierres sumen exactamente 100 y no haya errores técnicos."""
        self.assertTrue(self.ruta_predicciones.exists())
        with open(self.ruta_predicciones, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            filas = list(reader)

        self.assertEqual(len(filas), 100, "Predicciones.csv debe tener exactamente 100 registros")

        conteo_abiertas = 0
        conteo_cerradas = 0
        conteo_sin_rostro = 0
        conteo_varios_rostros = 0
        conteo_rechazo_umbral = 0
        errores_tecnicos = 0

        for row in filas:
            if row["error_tecnico"]:
                errores_tecnicos += 1
            decision = row["decision_puerta"]
            self.assertIn(decision, ["ABIERTA", "CERRADA"])
            if decision == "ABIERTA":
                conteo_abiertas += 1
                self.assertEqual(row["es_falsa_aceptacion"], "True")
                self.assertIn(row["etiqueta_predicha"], ["persona_1", "persona_2", "persona_3"])
                self.assertGreaterEqual(float(row["confianza"]), float(row["umbral"]))
            else:
                conteo_cerradas += 1
                self.assertEqual(row["es_falsa_aceptacion"], "False")
                est_det = row["estado_detector"]
                if est_det == "SIN_ROSTRO":
                    conteo_sin_rostro += 1
                elif est_det == "VARIOS_ROSTROS":
                    conteo_varios_rostros += 1
                else:
                    conteo_rechazo_umbral += 1

        self.assertEqual(errores_tecnicos, 0, "No debe haber errores técnicos registrados")
        self.assertEqual(conteo_abiertas + conteo_cerradas, 100, "Cierres + aperturas deben sumar 100")
        self.assertEqual(conteo_sin_rostro + conteo_varios_rostros + conteo_rechazo_umbral, conteo_cerradas)

    def test_coherencia_decision_evaluador_vs_web(self):
        """
        Verifica que el evaluador independiente y el GestorCamaraWeb de producción
        lleguen exactamente a la misma decisión para una misma entrada.
        """
        gestor = GestorCamaraWeb()
        gestor.cargar_modelo(Path("resultados/cnn_propio"), umbral=0.65)

        detector = DetectorFacial()
        modelo = ClasificadorCNN.cargar(Path("resultados/cnn_propio/modelo.pt"))
        clases_autorizadas = ["persona_1", "persona_2", "persona_3"]
        id_to_persona = {"0": "persona_1", "1": "persona_2", "2": "persona_3"}

        # 1. Probar escena vacía (negra): ambos deben dar SIN_ROSTRO -> CERRADA
        frame_vacio = np.zeros((480, 640, 3), dtype=np.uint8)
        dec_eval = reproducir_decision_produccion(
            frame_bgr=frame_vacio,
            detector=detector,
            modelo=modelo,
            clases_autorizadas=clases_autorizadas,
            id_to_persona=id_to_persona,
            umbral_confianza=0.65,
        )

        # En GestorCamaraWeb
        res_det_web = gestor.detector.detectar(frame_vacio, es_pre_recortada=False)
        self.assertEqual(dec_eval["estado_detector"], res_det_web.estado.value)
        self.assertEqual(dec_eval["decision_puerta"], "CERRADA")

        # 2. Probar sobre una de las imágenes de LFW guardadas
        ruta_img1 = self.dir_resultados / "imagenes_lfw" / "lfw_001_Natasha_Lyonne.jpg"
        if ruta_img1.exists():
            img_lfw = cv2.imread(str(ruta_img1))
            self.assertIsNotNone(img_lfw)

            # Decisión del evaluador
            dec_eval = reproducir_decision_produccion(
                frame_bgr=img_lfw,
                detector=detector,
                modelo=modelo,
                clases_autorizadas=clases_autorizadas,
                id_to_persona=id_to_persona,
                umbral_confianza=0.65,
            )

            # Decisión de GestorCamaraWeb
            res_det_web = gestor.detector.detectar(img_lfw, es_pre_recortada=False)
            self.assertEqual(dec_eval["estado_detector"], res_det_web.estado.value)

            # Preprocesar en web
            recorte = res_det_web.rostro_recortado
            from reconocimiento.preprocesamiento.normalizador import (
                a_escala_de_grises,
                normalizar_intensidades,
                normalizar_resolucion,
            )
            from reconocimiento.preprocesamiento.filtros import suavizado_gaussiano

            gris = a_escala_de_grises(recorte) if len(recorte.shape) == 3 else recorte
            norm_res = normalizar_resolucion(gris, (64, 64))
            norm_int = normalizar_intensidades(norm_res, "float32_0_1")
            suavizada = suavizado_gaussiano(norm_int, sigma=0.5)
            in_tensor = np.expand_dims(suavizada, axis=0)

            probs_web = gestor.modelo_reconocimiento.predecir_probabilidades(in_tensor)[0]
            pred_idx = int(np.argmax(probs_web))
            conf_web = float(probs_web[pred_idx])
            nom_web = gestor.mapeo_clases["id_to_persona"][str(pred_idx)]
            estado_puerta_web = "ABIERTA" if (conf_web >= 0.65 and nom_web in gestor.mapeo_clases["personas_autorizadas"]) else "CERRADA"

            self.assertEqual(dec_eval["decision_puerta"], estado_puerta_web)
            self.assertEqual(dec_eval["etiqueta_predicha"], nom_web)
            self.assertAlmostEqual(dec_eval["confianza"], conf_web, places=4)

    def test_regla_decision_canonica_compartida(self):
        """Verifica la función canónica evaluar_decision_acceso bajo todas las condiciones y con desconocido."""
        from reconocimiento.clasificacion.regla_acceso import evaluar_decision_acceso

        autorizados = ["persona_1", "persona_2", "persona_3"]

        # 1. Autorizado con confianza suficiente -> ABIERTA
        dec_ok = evaluar_decision_acceso(
            nombre_predicho="persona_1",
            confianza=0.85,
            umbral_confianza=0.80,
            personas_autorizadas=autorizados,
        )
        self.assertEqual(dec_ok.estado_puerta, "ABIERTA")
        self.assertTrue(dec_ok.es_autorizado)
        self.assertEqual(dec_ok.persona_identificada, "persona_1")
        self.assertEqual(dec_ok.motivo_cierre, "")

        # 2. Autorizado con confianza inferior al umbral -> CERRADA
        dec_bajo = evaluar_decision_acceso(
            nombre_predicho="persona_1",
            confianza=0.75,
            umbral_confianza=0.80,
            personas_autorizadas=autorizados,
        )
        self.assertEqual(dec_bajo.estado_puerta, "CERRADA")
        self.assertFalse(dec_bajo.es_autorizado)
        self.assertEqual(dec_bajo.persona_identificada, "DESCONOCIDO")
        self.assertIn("RECHAZO_UMBRAL", dec_bajo.motivo_cierre)

        # 3. Sujeto no autorizado (externo no listado) -> CERRADA
        dec_no_aut = evaluar_decision_acceso(
            nombre_predicho="persona_99",
            confianza=0.99,
            umbral_confianza=0.80,
            personas_autorizadas=autorizados,
        )
        self.assertEqual(dec_no_aut.estado_puerta, "CERRADA")
        self.assertFalse(dec_no_aut.es_autorizado)
        self.assertIn("CLASE_NO_AUTORIZADA", dec_no_aut.motivo_cierre)

        # 4. Clase desconocida explícita ("desconocido", "DESCONOCIDA", "unknown") -> CERRADA
        for etiqueta_desc in ["desconocido", "DESCONOCIDA", "unknown", "rechazo", "no_autorizado"]:
            dec_desc = evaluar_decision_acceso(
                nombre_predicho=etiqueta_desc,
                confianza=0.99,
                umbral_confianza=0.80,
                personas_autorizadas=autorizados,
            )
            self.assertEqual(dec_desc.estado_puerta, "CERRADA", f"Fallo al bloquear {etiqueta_desc}")
            self.assertFalse(dec_desc.es_autorizado)
            self.assertIn("CLASE_DESCONOCIDO", dec_desc.motivo_cierre)

    def test_umbral_dinamico_sin_forzar_065(self):
        """Verifica que el auditor no fuerce 0.65 y admita legítimamente un umbral calibrado de 0.80."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dir_mod = tmp_path / "modelo_prueba"
            dir_mod.mkdir()

            # Crear archivo de modelo simulado
            (dir_mod / "modelo.pt").write_bytes(b"dummy_weights_test")
            (dir_mod / "clases_mapeo.json").write_text(
                json.dumps({
                    "personas_autorizadas": ["persona_1", "persona_2"],
                    "id_to_persona": {"0": "persona_1", "1": "persona_2"},
                    "umbral_rechazo_recomendado": 0.80
                }),
                encoding="utf-8"
            )
            (dir_mod / "calibracion_umbral.json").write_text(
                json.dumps({"umbral_recomendado": 0.80}),
                encoding="utf-8"
            )

            # Archivo modelo_seleccionado.json con 0.80
            ruta_sel = tmp_path / "modelo_seleccionado.json"
            ruta_sel.write_text(
                json.dumps({
                    "modelo_elegido": "cnn",
                    "directorio_modelo": str(dir_mod).replace("\\", "/"),
                    "umbral_recomendado": 0.80
                }),
                encoding="utf-8"
            )

            dir_salida = tmp_path / "salida_eval"
            config = verificar_y_congelar_configuracion(
                ruta_modelo_seleccionado=ruta_sel,
                directorio_salida=dir_salida,
            )
            self.assertEqual(config["umbral_efectivo_puerta"], 0.80)

            # Caso de discrepancia: modelo calibrado a 0.80 pero web / selección configurada a 0.70
            ruta_sel_disc = tmp_path / "modelo_seleccionado_disc.json"
            ruta_sel_disc.write_text(
                json.dumps({
                    "modelo_elegido": "cnn",
                    "directorio_modelo": str(dir_mod).replace("\\", "/"),
                    "umbral_recomendado": 0.70  # Discrepancia intencional con 0.80
                }),
                encoding="utf-8"
            )
            with self.assertRaises(ValueError) as ctx:
                verificar_y_congelar_configuracion(
                    ruta_modelo_seleccionado=ruta_sel_disc,
                    directorio_salida=dir_salida,
                )
            self.assertIn("Discrepancia detectada", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
