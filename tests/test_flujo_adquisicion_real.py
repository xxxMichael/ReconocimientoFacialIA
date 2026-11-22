"""
Pruebas exhaustivas para el flujo real de adquisición de voluntarios desconocidos en la web.
Verifica:
1. Captura de desconocido válida con metadatos completos y guardado en disco y manifiesto.
2. Registro de fallo de detección preservando metadatos y estado histórico.
3. Rechazo estricto de captura sin consentimiento informado explícito.
4. Rechazo cuando falta grupo experimental o distancia física.
5. Rechazo cuando el grupo experimental contradice la partición reglamentaria.
6. Rechazo cuando una identidad externa intenta cambiar de grupo experimental (fuga de identidad).
7. Rechazo cuando un sujeto desconocido usa el identificador de una persona autorizada (conflicto de rol).
8. Comprobación de que una captura rechazada no deja archivos huérfanos en disco.
9. Congelado e inspección de fotograma conservando metadatos idénticos al confirmar.
10. Compatibilidad hacia atrás: capturas de autorizados existentes continúan operando sin grupo ni distancia.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from reconocimiento.adquisicion.propio import ManifiestoDataset, RegistroCaptura
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import FotogramaBuffer, GestorCamaraWeb


class TestFlujoAdquisicionReal(unittest.TestCase):

    def setUp(self):
        # Directorio temporal aislado para datos de prueba
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.base_tmp = Path(self.tmp_dir_obj.name)
        self.dir_propio = self.base_tmp / "data" / "propio"
        self.dir_ensayos = self.base_tmp / "data" / "ensayos"
        self.dir_propio.mkdir(parents=True, exist_ok=True)
        self.dir_ensayos.mkdir(parents=True, exist_ok=True)

        self.gestor = GestorCamaraWeb(tamano_buffer=10)
        self.gestor.detener()
        self.gestor.buffer.clear()
        self.gestor.cache_idempotencia.clear()

        # Redirigir directorios del gestor al entorno temporal aislado
        self.gestor.dir_propio = self.dir_propio
        self.gestor.dir_ensayos = self.dir_ensayos

        self.app = crear_app(indice_inicial=-1)
        self.client = self.app.test_client()

    def tearDown(self):
        self.gestor.detener()
        self.gestor.dir_propio = Path("data/propio")
        self.gestor.dir_ensayos = Path("data/ensayos")
        self.tmp_dir_obj.cleanup()

    def _inyectar_fotograma(self, id_frame: int, estado: EstadoDeteccion, valor_pixel: int = 120):
        """Inyecta un fotograma sintético al búfer del gestor web."""
        frame = np.full((480, 640, 3), valor_pixel, dtype=np.uint8)
        recuadro = (50, 50, 120, 120) if estado == EstadoDeteccion.UN_ROSTRO else None
        item = FotogramaBuffer(
            id_fotograma=id_frame,
            timestamp=2000.0 + id_frame,
            frame_raw_bgr=frame,
            estado_deteccion=estado,
            recuadro=recuadro,
            total_rostros=1 if estado == EstadoDeteccion.UN_ROSTRO else (2 if estado == EstadoDeteccion.VARIOS_ROSTROS else 0),
            frame_display_jpeg=b"fake_display_jpeg",
            ancho=640,
            alto=480
        )
        with self.gestor.lock_buffer:
            self.gestor.buffer.append(item)

    def test_01_captura_desconocido_valida(self):
        """1. Captura de desconocido válida con metadatos completos y guardado atómico."""
        self._inyectar_fotograma(id_frame=1, estado=EstadoDeteccion.UN_ROSTRO)

        payload = {
            "id_fotograma": 1,
            "modo": "valida",
            "persona": "externo_001",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_ajuste",
            "distancia": "cercana",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": False
        }

        resp = self.client.post("/api/capturar", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["modo"], "valida")
        self.assertTrue(Path(data["ruta"]).is_file())

        # Verificar manifiesto en disco temporal
        manifiesto = ManifiestoDataset(self.dir_propio / "manifiesto.csv")
        reg = manifiesto.buscar_registro(data["identificador"])
        self.assertIsNotNone(reg)
        self.assertEqual(reg.persona, "externo_001")
        self.assertEqual(reg.tipo_sujeto, "desconocido")
        self.assertEqual(reg.destino_experimental, "desconocidos_ajuste")
        self.assertEqual(reg.distancia, "cercana")
        self.assertEqual(reg.particion, "entrenamiento")
        self.assertTrue(reg.consentimiento)

    def test_02_registro_fallo_preservando_metadatos(self):
        """2. Registro de fallo de detección preservando metadatos y estado histórico."""
        self._inyectar_fotograma(id_frame=2, estado=EstadoDeteccion.SIN_ROSTRO)

        payload = {
            "id_fotograma": 2,
            "modo": "fallo",
            "persona": "externo_001",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_ajuste",
            "distancia": "media",
            "sesion": "sesion_01",
            "pose": "derecha",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": False
        }

        resp = self.client.post("/api/capturar", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["modo"], "fallo")
        self.assertEqual(data["estado_deteccion"], "SIN_ROSTRO")

        manifiesto = ManifiestoDataset(self.dir_propio / "manifiesto.csv")
        reg = manifiesto.buscar_registro(data["identificador"])
        self.assertIsNotNone(reg)
        self.assertEqual(reg.estado_historico_detector, "SIN_ROSTRO")
        self.assertIn("modo=fallo", reg.anotaciones)
        self.assertEqual(reg.distancia, "media")
        self.assertEqual(reg.destino_experimental, "desconocidos_ajuste")

    def test_03_rechazo_captura_sin_consentimiento(self):
        """3. Rechazo estricto de captura sin consentimiento informado explícito."""
        self._inyectar_fotograma(id_frame=3, estado=EstadoDeteccion.UN_ROSTRO)

        payload = {
            "id_fotograma": 3,
            "modo": "valida",
            "persona": "externo_002",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_calibracion",
            "distancia": "media",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "validacion",
            "consentimiento": False,  # Sin consentimiento
            "es_ensayo": False
        }

        resp = self.client.post("/api/capturar", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["tipo"], "VALIDACION")
        self.assertIn("consentimiento", data["mensaje"].lower())

    def test_04_rechazo_si_falta_grupo_o_distancia(self):
        """4. Rechazo cuando falta grupo experimental o distancia física para desconocido."""
        self._inyectar_fotograma(id_frame=4, estado=EstadoDeteccion.UN_ROSTRO)

        # Falta grupo
        payload_sin_grupo = {
            "id_fotograma": 4,
            "modo": "valida",
            "persona": "externo_002",
            "tipo_sujeto": "desconocido",
            "distancia": "media",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "validacion",
            "consentimiento": True,
            "es_ensayo": False
        }
        resp = self.client.post("/api/capturar", json=payload_sin_grupo)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("grupo experimental", resp.get_json()["mensaje"].lower())

        # Falta distancia
        payload_sin_distancia = {
            "id_fotograma": 4,
            "modo": "valida",
            "persona": "externo_002",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_calibracion",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "validacion",
            "consentimiento": True,
            "es_ensayo": False
        }
        resp = self.client.post("/api/capturar", json=payload_sin_distancia)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("distancia", resp.get_json()["mensaje"].lower())

    def test_05_rechazo_si_grupo_contradice_particion(self):
        """5. Rechazo cuando el grupo experimental contradice la partición reglamentaria."""
        self._inyectar_fotograma(id_frame=5, estado=EstadoDeteccion.UN_ROSTRO)

        # desconocidos_ajuste debe ir a 'entrenamiento', pero se especifica 'prueba'
        payload = {
            "id_fotograma": 5,
            "modo": "valida",
            "persona": "externo_002",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_ajuste",
            "distancia": "media",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "prueba",  # Contradicción
            "consentimiento": True,
            "es_ensayo": False
        }

        resp = self.client.post("/api/capturar", json=payload)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("inconsistencia de partición", data["mensaje"].lower())

    def test_06_rechazo_si_identidad_externa_cambia_de_grupo(self):
        """6. Rechazo cuando una identidad externa intenta cambiar de grupo experimental (fuga de identidad)."""
        self._inyectar_fotograma(id_frame=6, estado=EstadoDeteccion.UN_ROSTRO)

        # Registrar externo_005 en desconocidos_calibracion
        p1 = {
            "id_fotograma": 6,
            "modo": "valida",
            "persona": "externo_005",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_calibracion",
            "distancia": "cercana",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "validacion",
            "consentimiento": True,
            "es_ensayo": False
        }
        resp1 = self.client.post("/api/capturar", json=p1)
        self.assertEqual(resp1.status_code, 200)

        self._inyectar_fotograma(id_frame=7, estado=EstadoDeteccion.UN_ROSTRO)

        # Intentar registrar el mismo externo_005 en desconocidos_prueba
        p2 = {
            "id_fotograma": 7,
            "modo": "valida",
            "persona": "externo_005",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_prueba",  # Cambio de grupo no permitido
            "distancia": "lejana",
            "sesion": "sesion_01",
            "pose": "izquierda",
            "particion": "prueba",
            "consentimiento": True,
            "es_ensayo": False
        }
        resp2 = self.client.post("/api/capturar", json=p2)
        self.assertEqual(resp2.status_code, 400)
        data2 = resp2.get_json()
        self.assertIn("fuga de identidad externa", data2["mensaje"].lower())

    def test_07_rechazo_si_desconocido_usa_identificador_autorizado(self):
        """7. Rechazo cuando un desconocido usa el identificador de un autorizado (conflicto de rol)."""
        self._inyectar_fotograma(id_frame=8, estado=EstadoDeteccion.UN_ROSTRO)

        for nombre_invalido in ["persona_1", "PERSONA_2", "  persona_3  "]:
            payload = {
                "id_fotograma": 8,
                "modo": "valida",
                "persona": nombre_invalido,
                "tipo_sujeto": "desconocido",
                "grupo": "desconocidos_calibracion",
                "distancia": "media",
                "sesion": "sesion_01",
                "pose": "frontal",
                "particion": "validacion",
                "consentimiento": True,
                "es_ensayo": False
            }
            resp = self.client.post("/api/capturar", json=payload)
            self.assertEqual(resp.status_code, 400)
            data = resp.get_json()
            self.assertIn("conflicto de rol", data["mensaje"].lower())

    def test_08_captura_rechazada_no_deja_archivos_huerfanos(self):
        """8. Comprobar que una captura rechazada no deja archivos huérfanos en disco."""
        self._inyectar_fotograma(id_frame=9, estado=EstadoDeteccion.UN_ROSTRO)

        # Contar archivos físicos antes de la petición fallida
        archivos_antes = set(self.dir_propio.rglob("*.jpg"))

        # Petición que fallará por falta de distancia
        payload_invalido = {
            "id_fotograma": 9,
            "modo": "valida",
            "persona": "externo_099",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_ajuste",
            "distancia": "",  # Vacía -> error
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": False
        }
        resp = self.client.post("/api/capturar", json=payload_invalido)
        self.assertEqual(resp.status_code, 400)

        # Contar archivos físicos después: no debe existir ningún archivo huérfano
        archivos_despues = set(self.dir_propio.rglob("*.jpg"))
        self.assertEqual(archivos_antes, archivos_despues)

    def test_09_congelar_e_inspeccionar_manteniendo_metadatos_al_confirmar(self):
        """9. Congelar e inspeccionar fotograma manteniendo metadatos idénticos al confirmar."""
        self._inyectar_fotograma(id_frame=10, estado=EstadoDeteccion.UN_ROSTRO)

        metadatos_congelar = {
            "id_fotograma": 10,
            "persona": "externo_010",
            "tipo_sujeto": "desconocido",
            "grupo": "desconocidos_prueba",
            "distancia": "lejana",
            "sesion": "sesion_01",
            "pose": "arriba",
            "particion": "prueba",
            "consentimiento": True
        }

        # 1. Congelar
        resp_c = self.client.post("/api/captura/congelar", json=metadatos_congelar)
        self.assertEqual(resp_c.status_code, 200)
        datos_c = resp_c.get_json()
        self.assertEqual(datos_c["status"], "ok")
        self.assertEqual(datos_c["persona"], "externo_010")
        self.assertEqual(datos_c["grupo"], "desconocidos_prueba")
        self.assertEqual(datos_c["distancia"], "lejana")
        self.assertTrue(datos_c["consentimiento"])
        self.assertTrue(len(datos_c["recorte_base64"]) > 0)

        # 2. Confirmar usando exactamente los metadatos congelados
        payload_confirmar = {
            "id_fotograma": datos_c["id_fotograma"],
            "modo": "valida",
            "persona": datos_c["persona"],
            "tipo_sujeto": datos_c["tipo_sujeto"],
            "grupo": datos_c["grupo"],
            "distancia": datos_c["distancia"],
            "sesion": datos_c["sesion"],
            "pose": datos_c["pose"],
            "particion": datos_c["particion"],
            "consentimiento": datos_c["consentimiento"],
            "caja": datos_c["recuadro"],
            "decision": "aprobada",
            "origen_caja": "vivo",
            "es_ensayo": False
        }

        resp_guardar = self.client.post("/api/capturar", json=payload_confirmar)
        self.assertEqual(resp_guardar.status_code, 200)
        datos_g = resp_guardar.get_json()
        self.assertEqual(datos_g["status"], "ok")

        manifiesto = ManifiestoDataset(self.dir_propio / "manifiesto.csv")
        reg = manifiesto.buscar_registro(datos_g["identificador"])
        self.assertIsNotNone(reg)
        self.assertEqual(reg.persona, "externo_010")
        self.assertEqual(reg.destino_experimental, "desconocidos_prueba")
        self.assertEqual(reg.distancia, "lejana")
        self.assertEqual(reg.particion, "prueba")
        self.assertEqual(reg.pose, "arriba")

    def test_10_capturas_autorizados_existentes_continuan_funcionando(self):
        """10. Compatibilidad hacia atrás: capturas autorizadas funcionan sin especificar grupo ni distancia."""
        self._inyectar_fotograma(id_frame=11, estado=EstadoDeteccion.UN_ROSTRO)

        payload_autorizado = {
            "id_fotograma": 11,
            "modo": "valida",
            "persona": "persona_1",
            "tipo_sujeto": "autorizado",
            "sesion": "sesion_01",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": False
            # Sin 'grupo' ni 'distancia' (omisión histórica permitida para autorizados)
        }

        resp = self.client.post("/api/capturar", json=payload_autorizado)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

        manifiesto = ManifiestoDataset(self.dir_propio / "manifiesto.csv")
        reg = manifiesto.buscar_registro(data["identificador"])
        self.assertIsNotNone(reg)
        self.assertEqual(reg.persona, "persona_1")
        self.assertEqual(reg.tipo_sujeto, "autorizado")
        self.assertIsNone(reg.destino_experimental)


if __name__ == "__main__":
    unittest.main()
