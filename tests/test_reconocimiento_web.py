"""
Pruebas dirigidas para el sistema de reconocimiento facial en vivo
y la simulación de apertura de puerta en la interfaz web.

Verifica:
1. Concesión de acceso (PUERTA ABIERTA) cuando un rostro autorizado supera el umbral de confianza.
2. Denegación de acceso (PUERTA CERRADA) y rotulación como DESCONOCIDO cuando la confianza es inferior al umbral.
3. Denegación de acceso (PUERTA CERRADA) estricta ante estados SIN_ROSTRO y VARIOS_ROSTROS.
4. Consulta del estado del control de acceso vía endpoint /api/reconocimiento/estado.
5. Configuración dinámica del umbral de rechazo y alternancia de modelos mediante /api/reconocimiento/configurar.
"""
import unittest
from pathlib import Path
import numpy as np

from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import GestorCamaraWeb, FotogramaBuffer


class MockModeloReconocimiento:
    """Modelo simulado para pruebas de umbral y rechazo."""
    def __init__(self, probabilidades: np.ndarray):
        self.probabilidades = probabilidades

    def predecir_probabilidades(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.probabilidades])


class TestReconocimientoPuertaWeb(unittest.TestCase):

    def setUp(self):
        self.app = crear_app(indice_inicial=-1)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.gestor = self.app.config["GESTOR_CAMARA"]
        self.gestor.detener()
        self.gestor.detener_evento.clear()
        with self.gestor.lock_buffer:
            self.gestor.buffer.clear()
            self.gestor.contador_frames = 0

    def tearDown(self):
        if self.gestor.capturador is not None or (self.gestor.hilo_trabajador and self.gestor.hilo_trabajador.is_alive()):
            self.gestor.detener()

    def test_reconocimiento_abrir_puerta_con_autorizado(self):
        """Verifica que un rostro autorizado con confianza >= umbral abra la puerta."""
        # Configurar modelo con alta certeza para persona_1 (clase 0)
        self.gestor.modelo_reconocimiento = MockModeloReconocimiento(np.array([0.92, 0.05, 0.03]))
        self.gestor.es_cnn = True
        self.gestor.umbral_confianza = 0.65
        self.gestor.mapeo_clases = {
            "id_to_persona": {"0": "persona_1", "1": "persona_2", "2": "persona_3"},
            "personas_autorizadas": ["persona_1", "persona_2", "persona_3"]
        }

        # Simular procesamiento de un fotograma con UN_ROSTRO
        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        rostro_recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=rostro_recorte
        )

        self.gestor._procesar_y_almacenar_fotograma(frame_raw, res_det)
        ultimo = self.gestor.obtener_ultimo_fotograma()

        self.assertIsNotNone(ultimo)
        self.assertEqual(ultimo.estado_puerta, "ABIERTA")
        self.assertEqual(ultimo.persona_reconocida, "persona_1")
        self.assertTrue(ultimo.es_autorizado)
        self.assertGreaterEqual(ultimo.confianza, 65.0)

    def test_reconocimiento_cerrar_puerta_con_desconocido(self):
        """Verifica que un rostro con confianza inferior al umbral sea rechazado (PUERTA CERRADA)."""
        # Distribución difusa (desconocido o rostro fuera de distribución)
        self.gestor.modelo_reconocimiento = MockModeloReconocimiento(np.array([0.40, 0.35, 0.25]))
        self.gestor.es_cnn = True
        self.gestor.umbral_confianza = 0.65
        self.gestor.mapeo_clases = {
            "id_to_persona": {"0": "persona_1", "1": "persona_2", "2": "persona_3"},
            "personas_autorizadas": ["persona_1", "persona_2", "persona_3"]
        }

        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        rostro_recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=rostro_recorte
        )

        self.gestor._procesar_y_almacenar_fotograma(frame_raw, res_det)
        ultimo = self.gestor.obtener_ultimo_fotograma()

        self.assertIsNotNone(ultimo)
        self.assertEqual(ultimo.estado_puerta, "CERRADA")
        self.assertEqual(ultimo.persona_reconocida, "DESCONOCIDO")
        self.assertFalse(ultimo.es_autorizado)

    def test_reconocimiento_cerrar_puerta_sin_rostro(self):
        """Verifica que ante la ausencia de rostro la puerta permanezca estrictamente CERRADA."""
        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.SIN_ROSTRO,
            recuadro=None,
            total_rostros_detectados=0,
            rostro_recortado=None
        )

        self.gestor._procesar_y_almacenar_fotograma(frame_raw, res_det)
        ultimo = self.gestor.obtener_ultimo_fotograma()

        self.assertIsNotNone(ultimo)
        self.assertEqual(ultimo.estado_puerta, "CERRADA")
        self.assertFalse(ultimo.es_autorizado)

    def test_reconocimiento_cerrar_puerta_varios_rostros(self):
        """Verifica que ante múltiples rostros la puerta se bloquee estrictamente (PUERTA CERRADA)."""
        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.VARIOS_ROSTROS,
            recuadro=None,
            total_rostros_detectados=2,
            rostro_recortado=None
        )

        self.gestor._procesar_y_almacenar_fotograma(frame_raw, res_det)
        ultimo = self.gestor.obtener_ultimo_fotograma()

        self.assertIsNotNone(ultimo)
        self.assertEqual(ultimo.estado_puerta, "CERRADA")
        self.assertFalse(ultimo.es_autorizado)

    def test_api_reconocimiento_estado(self):
        """Verifica que el endpoint /api/reconocimiento/estado devuelva el estado de control de acceso."""
        resp = self.client.get("/api/reconocimiento/estado")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("estado_puerta", data)
        self.assertIn("persona_reconocida", data)
        self.assertIn("confianza", data)
        self.assertIn("umbral", data)

    def test_api_reconocimiento_configurar_umbral(self):
        """Verifica la actualización dinámica del umbral de rechazo vía API."""
        resp = self.client.post("/api/reconocimiento/configurar", json={"umbral": 0.80})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.gestor.umbral_confianza, 0.80)

        # Verificar respuesta JSON
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    def test_api_reconocimiento_configurar_modelo(self):
        """Verifica la alternancia dinámica de modelos entrenados (ej. SVM o ANN)."""
        resp = self.client.post("/api/reconocimiento/configurar", json={"modelo": "svm"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("SVM", data["modelo_activo"])

    def test_api_reconocimiento_modelo_inexistente(self):
        """Verifica manejo de error al solicitar un modelo no entrenado o inexistente."""
        resp = self.client.post("/api/reconocimiento/configurar", json={"modelo": "modelo_ficticio_xyz"})
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")

    def test_conceder_acceso_detener_camara_pasa_inmediatamente_a_cerrada(self):
        """
        Prueba de regresión:
        Conceder acceso (PUERTA ABIERTA), detener la cámara y consultar el estado
        antes de recibir otro fotograma. La puerta DEBE pasar inmediatamente a CERRADA.
        """
        self.gestor.modelo_reconocimiento = MockModeloReconocimiento(np.array([0.95, 0.03, 0.02]))
        self.gestor.es_cnn = True
        self.gestor.umbral_confianza = 0.65
        self.gestor.mapeo_clases = {
            "id_to_persona": {"0": "persona_1", "1": "persona_2", "2": "persona_3"},
            "personas_autorizadas": ["persona_1", "persona_2", "persona_3"]
        }

        frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
        rostro_recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
        res_det = ResultadoDeteccion(
            estado=EstadoDeteccion.UN_ROSTRO,
            recuadro=(100, 100, 80, 80),
            total_rostros_detectados=1,
            rostro_recortado=rostro_recorte
        )

        # 1. Conceder acceso
        self.gestor._procesar_y_almacenar_fotograma(frame_raw, res_det)
        ultimo = self.gestor.obtener_ultimo_fotograma()
        self.assertEqual(ultimo.estado_puerta, "ABIERTA")
        self.assertTrue(ultimo.es_autorizado)

        # 2. Detener la cámara inmediatamente
        self.gestor.detener()

        # 3. Consultar el estado antes de recibir otro fotograma (vía método de gestor)
        estado_gestor = self.gestor.obtener_estado()
        self.assertFalse(estado_gestor["activa"])
        self.assertEqual(estado_gestor["reconocimiento"]["estado_puerta"], "CERRADA")
        self.assertFalse(estado_gestor["reconocimiento"]["es_autorizado"])

        # 4. Consultar el estado antes de recibir otro fotograma (vía endpoint HTTP)
        resp_api = self.client.get("/api/reconocimiento/estado")
        self.assertEqual(resp_api.status_code, 200)
        data_api = resp_api.get_json()
        self.assertEqual(data_api["estado_puerta"], "CERRADA")
        self.assertFalse(data_api["es_autorizado"])

        # 5. Consultar último fotograma del búfer (debe forzar CERRADA al estar detenida)
        ultimo_detenido = self.gestor.obtener_ultimo_fotograma()
        self.assertEqual(ultimo_detenido.estado_puerta, "CERRADA")
        self.assertFalse(ultimo_detenido.es_autorizado)

    def test_knn_cargar_restaura_entrenamiento_y_predicciones(self):
        """Verifica que ClasificadorKNN.guardar y cargar preserven los datos y reproduzcan predicciones."""
        import tempfile
        from reconocimiento.clasificacion.knn import ClasificadorKNN

        X_tr = np.array([[0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95]], dtype=np.float32)
        y_tr = np.array([0, 0, 1, 1], dtype=np.int32)
        X_eval = np.array([[0.12, 0.22], [0.82, 0.92]], dtype=np.float32)

        knn_orig = ClasificadorKNN(k=3, metrica="euclidean")
        knn_orig.entrenar(X_tr, y_tr)
        preds_orig = knn_orig.predecir(X_eval)
        probs_orig = knn_orig.predecir_probabilidades(X_eval)

        tmp_path = Path(tempfile.mktemp(suffix=".joblib"))
        try:
            knn_orig.guardar(tmp_path)
            self.assertTrue(tmp_path.exists())

            knn_recargado = ClasificadorKNN.cargar(tmp_path)
            self.assertIsNotNone(knn_recargado.X_entrenamiento)
            self.assertIsNotNone(knn_recargado.y_entrenamiento)

            preds_rec = knn_recargado.predecir(X_eval)
            probs_rec = knn_recargado.predecir_probabilidades(X_eval)

            np.testing.assert_array_equal(preds_orig, preds_rec)
            np.testing.assert_allclose(probs_orig, probs_rec, rtol=1e-5)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_modelo_seleccionado_por_validacion(self):
        """Verifica que el modelo elegido se guarde y cargue desde resultados/modelo_seleccionado.json."""
        import json
        ruta_sel = Path("resultados/modelo_seleccionado.json")
        self.assertTrue(ruta_sel.exists(), "Debe existir resultados/modelo_seleccionado.json")

        with open(ruta_sel, "r", encoding="utf-8") as f:
            datos_sel = json.load(f)

        self.assertIn("modelo_elegido", datos_sel)
        self.assertIn("exactitud_validacion_pct", datos_sel)
        self.assertEqual(datos_sel["modelo_elegido"], "cnn")
        self.assertEqual(datos_sel["exactitud_validacion_pct"], 85.19)

        # Consultar endpoint /api/reconocimiento/seleccion
        resp = self.client.get("/api/reconocimiento/seleccion")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["modelo_elegido"], "cnn")
        self.assertIn("ranking_validacion", data)

    def test_clases_autorizadas_unicamente_de_entrenamiento(self):
        """Verifica que personas_autorizadas en clases_mapeo.json no incluya identidades desconocidas."""
        import json
        for mod in ["cnn_propio", "ann_propio", "svm_propio", "arbol_propio", "knn_propio"]:
            ruta_map = Path(f"resultados/{mod}/clases_mapeo.json")
            if ruta_map.exists():
                with open(ruta_map, "r", encoding="utf-8") as f:
                    mapeo = json.load(f)
                aut = mapeo.get("personas_autorizadas", [])
                self.assertEqual(aut, ["persona_1", "persona_2", "persona_3"])
                self.assertNotIn("desconocido", aut)

    def test_separacion_estricta_calibracion_y_prueba_final(self):
        """Verifica que calibración y prueba final estén estrictamente separadas en dos etapas."""
        import json
        for mod in ["cnn_propio", "ann_propio"]:
            ruta_calib = Path(f"resultados/{mod}/calibracion_umbral.json")
            ruta_prueba = Path(f"resultados/{mod}/prueba_final_acceso.json")
            self.assertTrue(ruta_calib.exists(), f"Debe existir calibracion_umbral.json para {mod}")
            self.assertTrue(ruta_prueba.exists(), f"Debe existir prueba_final_acceso.json para {mod}")

            with open(ruta_calib, "r", encoding="utf-8") as f:
                calib = json.load(f)
            with open(ruta_prueba, "r", encoding="utf-8") as f:
                prueba = json.load(f)

            self.assertEqual(calib["etapa"], "calibracion_validacion")
            self.assertIn("curva_calibracion_validacion", calib)
            self.assertEqual(calib["validacion"]["desconocidos_control_olivetti"], 10)

            self.assertEqual(prueba["etapa"], "prueba_final_reservada")
            self.assertEqual(prueba["umbral_congelado"], calib["umbral_congelado"])
            self.assertEqual(prueba["prueba"]["desconocidos_control_olivetti"], 10)
            self.assertIn("far_end_to_end_pct", prueba["prueba"])
            self.assertIn("frr_end_to_end_pct", prueba["prueba"])

    def test_resolucion_empate_validacion_sin_consultar_prueba(self):
        """Verifica que el empate de validación entre CNN y ANN se resuelva por criterio de seguridad sin consultar prueba."""
        import json
        ruta_sel = Path("resultados/modelo_seleccionado.json")
        self.assertTrue(ruta_sel.exists())
        with open(ruta_sel, "r", encoding="utf-8") as f:
            datos_sel = json.load(f)

        self.assertTrue(datos_sel["empate_en_validacion"])
        self.assertIn("cnn", datos_sel["modelos_empatados_validacion"])
        self.assertIn("ann", datos_sel["modelos_empatados_validacion"])

        ranking = {item["modelo"]: item for item in datos_sel["ranking_validacion"]}
        self.assertEqual(ranking["cnn"]["val_acc"], 85.19)
        self.assertEqual(ranking["ann"]["val_acc"], 85.19)
        # Desempate por seguridad en validación: CNN tuvo menor FAR en validación (80.0% vs 90.0%)
        self.assertLessEqual(ranking["cnn"]["far_val_pct"], ranking["ann"]["far_val_pct"])
        self.assertEqual(datos_sel["modelo_elegido"], "cnn")

    def test_distincion_apertura_puerta_vs_acierto_identidad(self):
        """Verifica que el reporte registre por separado apertura física de puerta y acierto de identidad."""
        import json
        ruta_cnn = Path("resultados/cnn_propio/prueba_final_acceso.json")
        self.assertTrue(ruta_cnn.exists())
        with open(ruta_cnn, "r", encoding="utf-8") as f:
            p = json.load(f)["prueba"]

        # 30 intentos autorizados en conjunto de prueba
        self.assertEqual(p["total_intentos_autorizados"], 30)
        # Aperturas físicas: 16 de 30 (53.33%)
        self.assertEqual(p["puerta_abierta_autorizados"], 16)
        self.assertEqual(p["tar_puerta_end_to_end_pct"], 53.33)
        # Bloqueos físicos: 14 de 30 (46.67%) [9 fallos Haar + 5 rechazos por umbral]
        self.assertEqual(p["puerta_cerrada_autorizados_bloqueos"], 14)
        self.assertEqual(p["frr_puerta_end_to_end_pct"], 46.67)

        # Identificación biométrica: 10 aciertos (33.33%) y 6 confusiones entre autorizados (20.00%)
        self.assertEqual(p["identificaciones_correctas"], 10)
        self.assertEqual(p["exactitud_identidad_end_to_end_pct"], 33.33)
        self.assertEqual(p["confusiones_entre_autorizados"], 6)
        self.assertEqual(p["confusion_identidad_end_to_end_pct"], 20.0)

        # Invariantes aritméticos
        self.assertEqual(p["puerta_abierta_autorizados"], p["identificaciones_correctas"] + p["confusiones_entre_autorizados"])
        self.assertEqual(p["total_intentos_autorizados"], p["puerta_cerrada_autorizados_bloqueos"] + p["puerta_abierta_autorizados"])

    def test_modelo_sin_calibracion_excluido_de_desempate(self):
        """Verifica que un modelo sin bloque 'validacion' sea marcado pendiente_calibrar y excluido de desempate."""
        import json
        import shutil
        import tempfile
        from reconocimiento.clasificacion.seleccion import seleccionar_mejor_modelo_propio

        temp_dir = Path(tempfile.mkdtemp(prefix="test_sel_"))
        try:
            # Modelo A (CNN): Calibrado en validación
            dir_a = temp_dir / "cnn_propio"
            dir_a.mkdir(parents=True)
            with open(dir_a / "metricas_evaluacion.json", "w") as f:
                json.dump({"reporte_val": {"exactitud_porcentaje": 85.0}, "reporte_train": {"exactitud_porcentaje": 100.0}}, f)
            with open(dir_a / "calibracion_umbral.json", "w") as f:
                json.dump({"etapa": "calibracion_validacion", "umbral_congelado": 0.70, "validacion": {"far_pct": 20.0, "frr_puerta_pct": 15.0}}, f)

            # Modelo B (ANN): Mismo val_acc (85.0%), pero sin archivo de calibración (o solo con evaluacion_rechazo de prueba antiguo)
            dir_b = temp_dir / "ann_propio"
            dir_b.mkdir(parents=True)
            with open(dir_b / "metricas_evaluacion.json", "w") as f:
                json.dump({"reporte_val": {"exactitud_porcentaje": 85.0}, "reporte_train": {"exactitud_porcentaje": 100.0}}, f)
            # Solo tiene evaluacion_rechazo.json global antiguo (sin validacion)
            with open(dir_b / "evaluacion_rechazo.json", "w") as f:
                json.dump({"far_pct": 5.0, "total_desconocidos": 20}, f)

            res = seleccionar_mejor_modelo_propio(dir_resultados=temp_dir, ruta_destino=temp_dir / "sel.json")
            # El modelo A (calibrado) DEBE ganar el desempate aunque B tenga un far_pct antiguo de prueba de 5%
            self.assertEqual(res["modelo_elegido"], "cnn")
            ranking = {item["modelo"]: item for item in res["ranking_validacion"]}
            self.assertTrue(ranking["cnn"]["calibrado"])
            self.assertFalse(ranking["ann"]["calibrado"])
            self.assertTrue(ranking["ann"]["pendiente_calibrar"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_umbral_calibrado_llega_a_la_web(self):
        """Verifica que el umbral calibrado en modelo_seleccionado.json llegue intacto a la web."""
        import json
        with open("resultados/modelo_seleccionado.json", "r", encoding="utf-8") as f:
            umbral_sel = float(json.load(f)["umbral_recomendado"])

        # Verificar que el gestor web tenga ese umbral configurado
        self.assertAlmostEqual(self.gestor.umbral_confianza, umbral_sel, places=2)
        resp = self.client.get("/api/reconocimiento/estado")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertAlmostEqual(data["umbral"], round(umbral_sel * 100, 1), places=1)

    def test_cambio_modelo_api_actualiza_umbral_calibrado_secuencia_cnn_svm_ann(self):
        """
        Prueba de aceptación:
        Al cambiar de modelo en la API (CNN -> SVM -> ANN), se debe cargar dinámicamente
        el umbral calibrado de cada modelo de destino y sincronizar el deslizador (0.65 -> 0.99 -> 0.60).
        """
        # 1. Cambiar a CNN -> Umbral debe ser 0.65 (65.0%)
        resp_cnn = self.client.post("/api/reconocimiento/configurar", json={"modelo": "cnn"})
        self.assertEqual(resp_cnn.status_code, 200)
        data_cnn = resp_cnn.get_json()
        self.assertEqual(data_cnn["status"], "ok")
        self.assertAlmostEqual(data_cnn["umbral_decimal"], 0.65, places=2)
        self.assertAlmostEqual(data_cnn["umbral"], 65.0, places=1)
        self.assertAlmostEqual(self.gestor.umbral_confianza, 0.65, places=2)

        # 2. Cambiar a SVM -> Umbral debe actualizarse a 0.99 (99.0%), sin arrastrar el 0.65 de CNN
        resp_svm = self.client.post("/api/reconocimiento/configurar", json={"modelo": "svm"})
        self.assertEqual(resp_svm.status_code, 200)
        data_svm = resp_svm.get_json()
        self.assertEqual(data_svm["status"], "ok")
        self.assertAlmostEqual(data_svm["umbral_decimal"], 0.99, places=2)
        self.assertAlmostEqual(data_svm["umbral"], 99.0, places=1)
        self.assertAlmostEqual(self.gestor.umbral_confianza, 0.99, places=2)

        # 3. Cambiar a ANN -> Umbral debe actualizarse a 0.60 (60.0%), sin arrastrar el 0.99 de SVM
        resp_ann = self.client.post("/api/reconocimiento/configurar", json={"modelo": "ann"})
        self.assertEqual(resp_ann.status_code, 200)
        data_ann = resp_ann.get_json()
        self.assertEqual(data_ann["status"], "ok")
        self.assertAlmostEqual(data_ann["umbral_decimal"], 0.60, places=2)
        self.assertAlmostEqual(data_ann["umbral"], 60.0, places=1)
        self.assertAlmostEqual(self.gestor.umbral_confianza, 0.60, places=2)

        # Verificar que el estado del gestor refleje fielmente el umbral final de ANN (60.0%)
        resp_est = self.client.get("/api/reconocimiento/estado")
        self.assertEqual(resp_est.status_code, 200)
        data_est = resp_est.get_json()
        self.assertAlmostEqual(data_est["umbral"], 60.0, places=1)

    def test_modelo_sin_calibracion_completa_no_puede_ser_elegido_incluso_con_mayor_exactitud(self):
        """
        Verifica que un modelo sin calibración completa quede estrictamente descalificado
        para participar en la selección automática, incluso si supera en exactitud (ej. 90% vs 85%).
        Asimismo, un archivo con far_pct que carezca de umbral explícito debe ser rechazado como no calibrado.
        """
        import json
        import shutil
        import tempfile
        from reconocimiento.clasificacion.seleccion import seleccionar_mejor_modelo_propio

        temp_dir = Path(tempfile.mkdtemp(prefix="test_sel_calib_"))
        try:
            # Modelo 1 (SVM): 90.0% de exactitud en validación, PERO sin calibración completa
            # Caso 1a: archivo con far_pct pero sin umbral explícito (no debe asumir 0.65)
            dir_svm = temp_dir / "svm_propio"
            dir_svm.mkdir(parents=True)
            with open(dir_svm / "metricas_evaluacion.json", "w") as f:
                json.dump({"reporte_val": {"exactitud_porcentaje": 90.0}, "reporte_train": {"exactitud_porcentaje": 100.0}}, f)
            with open(dir_svm / "calibracion_umbral.json", "w") as f:
                # Tiene bloque validacion y far_pct, pero CARECE de umbral_congelado/umbral_recomendado/umbral
                json.dump({"etapa": "calibracion_validacion", "validacion": {"far_pct": 10.0, "frr_puerta_pct": 5.0}}, f)

            # Modelo 2 (Árbol): 92.0% de exactitud en validación, PERO totalmente sin calibración
            dir_arbol = temp_dir / "arbol_propio"
            dir_arbol.mkdir(parents=True)
            with open(dir_arbol / "metricas_evaluacion.json", "w") as f:
                json.dump({"reporte_val": {"exactitud_porcentaje": 92.0}, "reporte_train": {"exactitud_porcentaje": 100.0}}, f)

            # Modelo 3 (CNN): 85.0% de exactitud en validación, CON CALIBRACIÓN COMPLETA
            dir_cnn = temp_dir / "cnn_propio"
            dir_cnn.mkdir(parents=True)
            with open(dir_cnn / "metricas_evaluacion.json", "w") as f:
                json.dump({"reporte_val": {"exactitud_porcentaje": 85.0}, "reporte_train": {"exactitud_porcentaje": 100.0}}, f)
            with open(dir_cnn / "calibracion_umbral.json", "w") as f:
                json.dump({
                    "etapa": "calibracion_validacion",
                    "umbral_congelado": 0.65,
                    "umbral_recomendado": 0.65,
                    "validacion": {"far_pct": 30.0, "frr_puerta_pct": 12.0}
                }, f)

            res = seleccionar_mejor_modelo_propio(dir_resultados=temp_dir, ruta_destino=temp_dir / "sel.json")

            # El modelo CNN (85% calibrado) DEBE ganar sobre SVM (90% sin umbral) y Árbol (92% sin calibración)
            self.assertEqual(res["modelo_elegido"], "cnn")
            self.assertEqual(res["umbral_recomendado"], 0.65)
            self.assertTrue(res["calibracion_completa"])
            self.assertEqual(res["total_modelos_calibrados"], 1)

            # Verificar ranking y estado de calibración
            ranking = {item["modelo"]: item for item in res["ranking_validacion"]}
            self.assertTrue(ranking["cnn"]["calibrado"])
            self.assertFalse(ranking["svm"]["calibrado"])
            self.assertIsNone(ranking["svm"]["umbral_calibrado"])
            self.assertFalse(ranking["arbol"]["calibrado"])
            self.assertIsNone(ranking["arbol"]["umbral_calibrado"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ningun_modelo_calibrado_mantiene_captura_web_y_desactiva_autorizacion(self):
        """
        Verifica que si ningún modelo cuenta con calibración completa:
        1. El selector genera modelo_elegido = 'ninguno' y calibracion_completa = False.
        2. La interfaz web mantiene desactivada la autorización (puerta CERRADA ante cualquier rostro).
        3. La captura de imágenes permanece 100% disponible y operativa en la web.
        """
        import json
        import shutil
        import tempfile
        from reconocimiento.clasificacion.seleccion import seleccionar_mejor_modelo_propio

        temp_dir = Path(tempfile.mkdtemp(prefix="test_sel_none_"))
        try:
            # Modelo sin archivo de calibración
            dir_ann = temp_dir / "ann_propio"
            dir_ann.mkdir(parents=True)
            with open(dir_ann / "metricas_evaluacion.json", "w") as f:
                json.dump({"reporte_val": {"exactitud_porcentaje": 88.0}, "reporte_train": {"exactitud_porcentaje": 99.0}}, f)

            ruta_sel = temp_dir / "modelo_seleccionado.json"
            res = seleccionar_mejor_modelo_propio(dir_resultados=temp_dir, ruta_destino=ruta_sel)

            self.assertEqual(res["modelo_elegido"], "ninguno")
            self.assertFalse(res["calibracion_completa"])
            self.assertEqual(res["total_modelos_calibrados"], 0)

            # Comprobar comportamiento en el gestor web
            gestor = GestorCamaraWeb()
            gestor.desactivar_modelo("Ningún modelo cuenta con calibración completa.")
            self.assertIsNone(gestor.modelo_reconocimiento)

            # Simular fotograma con rostro detectado
            frame_raw = np.zeros((480, 640, 3), dtype=np.uint8)
            rostro_recorte = np.ones((80, 80, 3), dtype=np.uint8) * 128
            res_det = ResultadoDeteccion(
                estado=EstadoDeteccion.UN_ROSTRO,
                recuadro=(100, 100, 80, 80),
                total_rostros_detectados=1,
                rostro_recortado=rostro_recorte
            )

            # Procesar fotograma sin modelo cargado
            gestor._procesar_y_almacenar_fotograma(frame_raw, res_det)
            ultimo = gestor.obtener_ultimo_fotograma()

            # La autorización permanece desactivada y la puerta estrictamente cerrada
            self.assertIsNotNone(ultimo)
            self.assertEqual(ultimo.estado_puerta, "CERRADA")
            self.assertFalse(ultimo.es_autorizado)
            self.assertEqual(ultimo.persona_reconocida, "")
            self.assertIn("Autorización desactivada", ultimo.mensaje_acceso)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)




if __name__ == "__main__":
    unittest.main()
