import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado
from reconocimiento.portal_reconocimiento.app import crear_portal_reconocimiento_app
from reconocimiento.portal_reconocimiento.servicio import ServicioPortalReconocimiento
from reconocimiento.web.streamer import GestorCamaraWeb


class TestPortalReconocimientoRemoto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.ruta_intentos = Path(cls.tmp_dir.name) / "intentos.jsonl"
        cls.app = crear_portal_reconocimiento_app(
            raiz_proyecto=RAIZ,
            ruta_registro_intentos=cls.ruta_intentos,
            max_inferencias_simultaneas=2,
        )
        cls.client = cls.app.test_client()
        cls.ruta_img = RAIZ / "data/propio/imagenes/persona_1/sesion_04/frontal/persona_1_sesion_04_frontal_f01.jpg"
        cls.img_bytes = cls.ruta_img.read_bytes() if cls.ruta_img.exists() else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_01_carga_ambos_candidatos_y_sin_slider_umbral(self):
        resp_salud = self.client.get("/api/salud")
        self.assertEqual(resp_salud.status_code, 200)
        data = resp_salud.get_json()
        self.assertEqual(data["candidato_principal_id"], "cnn_estandar_portal_061")
        cands = {c["id"]: c for c in data["candidatos"]}
        self.assertIn("cnn_antialiasing_052", cands)
        self.assertIn("cnn_estandar_portal_061", cands)
        self.assertAlmostEqual(cands["cnn_antialiasing_052"]["umbral"], 0.52, places=2)
        self.assertAlmostEqual(cands["cnn_estandar_portal_061"]["umbral"], 0.61, places=2)

        resp_html = self.client.get("/")
        self.assertEqual(resp_html.status_code, 200)
        html = resp_html.get_data(as_text=True)
        self.assertIn("CNN Antialiasing — 0.52", html)
        self.assertIn("CNN Estándar Portal Completo — 0.61", html)
        self.assertNotIn("slider-umbral", html)

    def test_02_igualdad_decisiones_con_pipeline_canonico(self):
        if self.img_bytes is None:
            self.skipTest("Imagen de prueba no disponible")
        frame_bgr = cv2.imdecode(np.frombuffer(self.img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

        for cid, cfg_name in [
            ("cnn_antialiasing_052", "configuracion_candidato_antialiasing_052.json"),
            ("cnn_estandar_portal_061", "configuracion_candidato_estandar_portal_061.json"),
        ]:
            with self.subTest(candidato=cid):
                pipe = PipelineCandidatoCongelado(RAIZ / cfg_name, raiz_proyecto=RAIZ)
                res_det_canon, dec_canon, _ = pipe.evaluar_fotograma_completo(frame_bgr)

                # Codificar JPEG exactamente como hace el navegador antes de enviar
                ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                self.assertTrue(ok)
                resp = self.client.post(
                    "/api/reconocer",
                    data={"frame": (io.BytesIO(buf.tobytes()), "frame.jpg"), "candidato_id": cid, "frame_seq": "7"},
                    content_type="multipart/form-data",
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                self.assertEqual(data["status"], "ok")
                self.assertEqual(data["frame_seq"], 7)
                self.assertEqual(data["estado_detector"], res_det_canon.estado.value)
                self.assertEqual(data["decision_puerta"], dec_canon.estado_puerta)

    def test_03_ausencia_rutas_dataset_y_revision(self):
        rutas_prohibidas = [
            ("GET", "/api/revision/lista"),
            ("GET", "/api/pipeline/dataset_imagenes"),
            ("GET", "/api/protocolo/particiones"),
            ("GET", "/api/camaras"),
            ("GET", "/api/frame/latest"),
            ("POST", "/api/capturar"),
            ("POST", "/api/reconocimiento/configurar"),
        ]
        for metodo, ruta in rutas_prohibidas:
            resp = self.client.get(ruta) if metodo == "GET" else self.client.post(ruta)
            self.assertEqual(resp.status_code, 404, f"Ruta prohibida expuesta: {metodo} {ruta}")

    def test_04_rechaza_rutas_arbitrarias_y_aisla_visitantes(self):
        if self.img_bytes is None:
            self.skipTest("Imagen de prueba no disponible")

        resp_bad = self.client.post(
            "/api/reconocer",
            data={"frame": (io.BytesIO(self.img_bytes), "f.jpg"), "candidato_id": "../../configuracion_candidato_alineado.json"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp_bad.status_code, 400)

        errores = []
        resultados = {}

        def _visitante(cid: str, clave: str):
            try:
                c = self.app.test_client()
                for i in range(3):
                    r = c.post(
                        "/api/reconocer",
                        data={"frame": (io.BytesIO(self.img_bytes), "f.jpg"), "candidato_id": cid, "frame_seq": str(i)},
                        content_type="multipart/form-data",
                    )
                    if r.status_code != 200:
                        errores.append((clave, r.status_code, r.get_json()))
                        return
                    d = r.get_json()
                    resultados.setdefault(clave, []).append((d["candidato_id"], d["umbral"]))
            except Exception as e:
                errores.append((clave, str(e)))

        t1 = threading.Thread(target=_visitante, args=("cnn_antialiasing_052", "v1"))
        t2 = threading.Thread(target=_visitante, args=("cnn_estandar_portal_061", "v2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errores, [])
        self.assertTrue(all(cid == "cnn_antialiasing_052" and umb == 0.52 for cid, umb in resultados["v1"]))
        self.assertTrue(all(cid == "cnn_estandar_portal_061" and umb == 0.61 for cid, umb in resultados["v2"]))

    def test_05_sin_rostro_errores_y_registro_unico_sin_fotos(self):
        negra = np.zeros((120, 120, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", negra)
        self.assertTrue(ok)

        # Registro de intento con fallo de detección SIN_ROSTRO y deduplicación por id_intento
        id_unico = "intento_test_dedup_001"
        for idx_rep in range(2):
            r = self.client.post(
                "/api/intento",
                data={
                    "frame": (io.BytesIO(buf.tobytes()), "negra.jpg"),
                    "candidato_id": "cnn_antialiasing_052",
                    "categoria_real": "desconocido",
                    "participante_id": "part_test_01",
                    "id_intento": id_unico,
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(r.status_code, 200)
            payload = r.get_json()
            self.assertEqual(payload["duplicado"], idx_rep == 1)

        # Verificar que no se escribieron imágenes ni base64, y solo hay 1 línea para id_unico
        archivos_tmp = list(Path(self.tmp_dir.name).rglob("*"))
        self.assertEqual([p.name for p in archivos_tmp], ["intentos.jsonl"])
        lineas = self.ruta_intentos.read_text(encoding="utf-8").strip().splitlines()
        regs = [json.loads(ln) for ln in lineas if json.loads(ln)["id_intento"] == id_unico]
        self.assertEqual(len(regs), 1)
        self.assertEqual(regs[0]["estado_detector"], "SIN_ROSTRO")
        self.assertEqual(regs[0]["decision_puerta"], "CERRADA")
        self.assertNotIn("base64", lineas[0].lower())

    def test_06_concurrencia_cambio_modelo_en_gestor_camara_web(self):
        if self.img_bytes is None:
            self.skipTest("Imagen de prueba no disponible")
        frame_bgr = cv2.imdecode(np.frombuffer(self.img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

        gestor = GestorCamaraWeb()
        cfg_aa = RAIZ / "configuracion_candidato_antialiasing_052.json"
        cfg_std = RAIZ / "configuracion_candidato_estandar_portal_061.json"
        gestor.cargar_desde_configuracion_congelada(cfg_aa)

        inconsistencias = []
        detener = threading.Event()

        def _cambiador():
            for i in range(6):
                gestor.cargar_desde_configuracion_congelada(cfg_std if i % 2 == 0 else cfg_aa)
            detener.set()

        def _lector_inferencia():
            while not detener.is_set():
                snap = gestor._obtener_instantanea_modelo()
                pipe = snap["pipeline_candidato"]
                if pipe is not None:
                    # Detector, método de remuestreo y umbral deben corresponder al mismo candidato
                    if abs(snap["umbral_confianza"] - 0.52) < 1e-3 and snap["detector"].metodo_remuestreo != "supersampling_area":
                        inconsistencias.append("Mezcla 0.52 con estandar")
                    if abs(snap["umbral_confianza"] - 0.61) < 1e-3 and snap["detector"].metodo_remuestreo != "estandar":
                        inconsistencias.append("Mezcla 0.61 con supersampling_area")
                res = gestor.procesar_fotograma_para_reconocimiento(frame_bgr)
                if res["umbral"] not in (52.0, 61.0):
                    inconsistencias.append(f"Umbral inesperado: {res['umbral']}")

        t_sw = threading.Thread(target=_cambiador)
        t_inf = threading.Thread(target=_lector_inferencia)
        t_inf.start()
        t_sw.start()
        t_sw.join()
        t_inf.join()

        self.assertEqual(inconsistencias, [])

    def test_07_reinicio_servidor_recupera_registro_original_sin_duplicar(self):
        if self.img_bytes is None:
            self.skipTest("Imagen de prueba no disponible")
        with tempfile.TemporaryDirectory() as tmp:
            ruta_jsonl = Path(tmp) / "intentos_reinicio.jsonl"
            srv1 = ServicioPortalReconocimiento(raiz_proyecto=RAIZ, ruta_registro_intentos=ruta_jsonl)
            res1 = srv1.registrar_intento(
                frame_bytes=self.img_bytes,
                candidato_id="cnn_antialiasing_052",
                categoria_real="persona_1",
                participante_id="part_reinicio",
                id_intento="intento_reinicio_001",
            )
            self.assertFalse(res1["duplicado"])
            reg_orig = res1["registro"]

            # Simular reinicio del servidor cargando desde el mismo archivo JSONL
            srv2 = ServicioPortalReconocimiento(raiz_proyecto=RAIZ, ruta_registro_intentos=ruta_jsonl)
            res2 = srv2.registrar_intento(
                frame_bytes=self.img_bytes,
                candidato_id="cnn_estandar_portal_061",
                categoria_real="desconocido",
                participante_id="part_reinicio",
                id_intento="intento_reinicio_001",
            )
            self.assertTrue(res2["duplicado"])
            self.assertEqual(res2["registro"], reg_orig)
            lineas = ruta_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lineas), 1)

    def test_08_frontend_js_ejecutable(self):
        proc = subprocess.run(
            ["node", str(RAIZ / "tests/test_portal_reconocimiento_frontend.mjs")],
            cwd=str(RAIZ),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, f"Fallo en pruebas JS:\n{proc.stdout}\n{proc.stderr}")

    def test_09_fase1_filtro_calidad_no_altera_estandar_y_bloquea_oscuridad_y_oclusion(self):
        if self.img_bytes is None:
            self.skipTest("Imagen de prueba no disponible")
        frame_bgr = cv2.imdecode(np.frombuffer(self.img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

        pipe_std = PipelineCandidatoCongelado(RAIZ / "configuracion_candidato_estandar_portal_061.json", raiz_proyecto=RAIZ)
        pipe_cal = PipelineCandidatoCongelado(RAIZ / "configuracion_candidato_estandar_portal_061_calidad.json", raiz_proyecto=RAIZ)

        self.assertFalse(pipe_std.config_calidad.habilitado)
        self.assertTrue(pipe_cal.config_calidad.habilitado)
        self.assertEqual(pipe_std.sha256_modelo, pipe_cal.sha256_modelo)

        # Limpia: ambas deben aceptar a persona_1
        det_s, dec_s, _ = pipe_std.evaluar_fotograma_completo(frame_bgr)
        det_c, dec_c, _ = pipe_cal.evaluar_fotograma_completo(frame_bgr)
        self.assertEqual(dec_s.estado_puerta, "ABIERTA")
        self.assertEqual(dec_c.estado_puerta, "ABIERTA")

        # Oscuridad severa: filtro de calidad bloquea antes de clasificar
        frame_osc = np.clip(frame_bgr.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
        _, dec_osc, probs_osc = pipe_cal.evaluar_fotograma_completo(frame_osc)
        self.assertEqual(dec_osc.estado_puerta, "CERRADA")
        if probs_osc is None and det_s.recuadro is not None:
            self.assertIn(dec_osc.motivo_cierre, {"ILUMINACION_INSUFICIENTE", "CONTRASTE_INSUFICIENTE"})

        # Oclusión inferior: filtro de calidad bloquea antes de clasificar
        if det_s.recuadro is not None:
            x, y, w, h = det_s.recuadro
            frame_ocl = frame_bgr.copy()
            frame_ocl[y + int(0.6 * h):y + h, x:x + w] = 0
            _, dec_ocl, probs_ocl = pipe_cal.evaluar_fotograma_completo(frame_ocl)
            self.assertEqual(dec_ocl.estado_puerta, "CERRADA")
            if probs_ocl is None:
                self.assertIn(dec_ocl.motivo_cierre, {"OCLUSION_INFERIOR", "DETALLE_INFERIOR_INSUFICIENTE", "ILUMINACION_INSUFICIENTE"})


if __name__ == "__main__":
    unittest.main()
