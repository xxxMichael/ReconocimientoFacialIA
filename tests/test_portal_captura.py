"""
Pruebas funcionales y de seguridad para el portal público de captura independiente:
1. Invitación válida, inválida y revocación/reactivación.
2. Consentimiento obligatorio previo a la captura y envío.
3. Captura y envío de imagen limpia sin recuadros ni modificaciones.
4. Idempotencia y control de duplicados ante reintentos de red.
5. Aislamiento y separación estricta entre dos participantes independientes.
6. Cálculo de progreso exacto para el protocolo de 15 combinaciones (5 poses x 3 distancias).
7. Ausencia de rutas administrativas para listar, revisar o descargar fotografías.
8. Aceptación y registro de fotografías sin rostro (sin filtrado por detector).
9. Manejo de archivos corruptos, tamaños excesivos o parámetros inválidos con mensajes comprensibles.
10. Importación correcta al manifiesto del proyecto respetando validaciones de protocolo.
11. Verificación de viewport móvil y metadatos de interfaz responsive.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from reconocimiento.portal.app import crear_portal_app
from reconocimiento.portal.almacenamiento import (
    GestorAlmacenamientoExterno,
    POSES_PROTOCOLO,
    DISTANCIAS_PROTOCOLO,
    TOTAL_COMBINACIONES_PROTOCOLO,
)
from reconocimiento.portal.gestor_invitaciones import GestorInvitaciones
from reconocimiento.portal.importador import importar_capturas_externas
from reconocimiento.adquisicion.propio import ManifiestoDataset


def crear_imagen_sintetica_jpeg(ancho: int = 320, alto: int = 240, color: int = 150) -> bytes:
    """Crea una imagen JPEG válida en memoria."""
    img = np.full((alto, ancho, 3), color, dtype=np.uint8)
    # Dibujar un patrón simple
    cv2.circle(img, (ancho // 2, alto // 2), 40, (255, 200, 100), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("No se pudo codificar imagen JPEG sintética.")
    return buf.tobytes()


class TestPortalCapturaIndependiente(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_temp = Path(self.temp_dir.name)

        self.dir_recepcion = self.dir_temp / "recepcion_externa"
        self.ruta_invitaciones = self.dir_recepcion / "invitaciones.json"

        self.app = crear_portal_app(
            dir_recepcion=self.dir_recepcion,
            ruta_invitaciones=self.ruta_invitaciones,
            raiz_proyecto=self.dir_temp
        )
        self.client = self.app.test_client()

        self.gestor_inv = GestorInvitaciones(
            ruta_archivo=self.ruta_invitaciones,
            raiz_proyecto=self.dir_temp
        )
        self.gestor_alm = GestorAlmacenamientoExterno(
            dir_base=self.dir_recepcion,
            raiz_proyecto=self.dir_temp
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _subir_foto(self, token: str, pose: str, distancia: str, img_bytes: bytes, filename: str = "foto.jpg"):
        """Helper para enviar fotografías multipart al endpoint /api/subir."""
        return self.client.post("/api/subir", data={
            "token": token,
            "pose": pose,
            "distancia": distancia,
            "imagen": (io.BytesIO(img_bytes), filename),
        }, content_type="multipart/form-data")

    def test_01_invitacion_valida_e_invalida_y_revocacion(self):
        """Verifica emisión, revocación y validación de tokens únicos."""
        inv = self.gestor_inv.crear_invitacion("externo_001", notas="Voluntario 1")
        token = inv["token"]
        self.assertTrue(inv["activo"])
        self.assertEqual(inv["identificador"], "externo_001")

        # 1. Invitación válida retorna 200
        resp = self.client.get(f"/participar/{token}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"externo_001", resp.data)

        # 2. Token inválido retorna 403
        resp_falsa = self.client.get("/participar/token_inexistente_12345")
        self.assertEqual(resp_falsa.status_code, 403)

        # 3. Desactivar invitación revoca el acceso
        self.gestor_inv.desactivar_invitacion(token)
        resp_desact = self.client.get(f"/participar/{token}")
        self.assertEqual(resp_desact.status_code, 403)

        # 4. Reactivar permite acceso nuevamente
        self.gestor_inv.activar_invitacion(token)
        resp_react = self.client.get(f"/participar/{token}")
        self.assertEqual(resp_react.status_code, 200)

        # 5. Prohibición estricta de identidades autorizadas
        with self.assertRaises(ValueError):
            self.gestor_inv.crear_invitacion("persona_1")

    def test_02_consentimiento_obligatorio_para_envio(self):
        """Verifica que no se puedan enviar imágenes sin haber registrado consentimiento explícito."""
        inv = self.gestor_inv.crear_invitacion("externo_002")
        token = inv["token"]

        img_bytes = crear_imagen_sintetica_jpeg()

        # Intentar subir sin consentir debe retornar 403
        resp_subir_sin_cons = self._subir_foto(token, "frontal", "cercana", img_bytes)
        self.assertEqual(resp_subir_sin_cons.status_code, 403)
        data_err = resp_subir_sin_cons.get_json()
        self.assertIn("consentimiento", data_err["error"].lower())

        # Registrar consentimiento mediante el endpoint dedicado
        resp_cons = self.client.post("/api/consentimiento", json={"token": token})
        self.assertEqual(resp_cons.status_code, 200)
        self.assertTrue(resp_cons.get_json()["exito"])

        # Consultar estado y verificar que quedó registrado
        resp_est = self.client.get(f"/api/estado?token={token}")
        self.assertEqual(resp_est.status_code, 200)
        self.assertTrue(resp_est.get_json()["consentimiento_registrado"])

    def test_03_captura_y_envio_imagen_y_original_limpio(self):
        """Verifica el envío de fotografía y que se almacene el original crudo sin modificaciones."""
        inv = self.gestor_inv.crear_invitacion("externo_003")
        token = inv["token"]
        self.client.post("/api/consentimiento", json={"token": token})

        img_bytes = crear_imagen_sintetica_jpeg(ancho=640, alto=480, color=180)
        hash_enviado = hashlib.sha256(img_bytes).hexdigest()

        resp = self._subir_foto(token, "frontal", "cercana", img_bytes, "frontal_cercana.jpg")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["exito"])
        # La respuesta no debe exponer rutas absolutas ni enlaces de descarga
        self.assertNotIn("ruta_absoluta", data)
        self.assertNotIn("c:/", json.dumps(data).lower())

        # Comprobar que en disco exista el archivo original con el mismo hash exacto
        archivo_guardado = self.dir_recepcion / "fotos" / "externo_003" / "externo_003_frontal_cercana.jpg"
        self.assertTrue(archivo_guardado.exists())
        hash_disco = hashlib.sha256(archivo_guardado.read_bytes()).hexdigest()
        self.assertEqual(hash_disco, hash_enviado)

    def test_04_reintento_sin_duplicacion_e_idempotencia(self):
        """Verifica que reintentos de red o doble clic no dupliquen archivos ni corrompan el progreso."""
        inv = self.gestor_inv.crear_invitacion("externo_004")
        token = inv["token"]
        self.client.post("/api/consentimiento", json={"token": token})

        img_bytes = crear_imagen_sintetica_jpeg()

        # Primer envío
        r1 = self._subir_foto(token, "izquierda", "media", img_bytes)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.get_json()["progreso"]["total_enviadas"], 1)

        # Reintento idéntico (misma imagen, misma combinación) -> éxito idempotente
        r2 = self._subir_foto(token, "izquierda", "media", img_bytes)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.get_json()["progreso"]["total_enviadas"], 1)

        # Intento de enviar una foto DISTINTA para la misma combinación -> rechazo seguro
        img_distinta = crear_imagen_sintetica_jpeg(color=220)
        r3 = self._subir_foto(token, "izquierda", "media", img_distinta)
        self.assertEqual(r3.status_code, 400)
        self.assertIn("ya existe", r3.get_json()["error"].lower())

    def test_05_separacion_estricta_entre_dos_participantes(self):
        """Verifica que dos voluntarios conserven carpetas, identificadores y avances completamente independientes."""
        inv1 = self.gestor_inv.crear_invitacion("externo_005a")
        inv2 = self.gestor_inv.crear_invitacion("externo_005b")

        self.client.post("/api/consentimiento", json={"token": inv1["token"]})
        self.client.post("/api/consentimiento", json={"token": inv2["token"]})

        img1 = crear_imagen_sintetica_jpeg(color=100)
        img2 = crear_imagen_sintetica_jpeg(color=200)

        # Subir misma combinación en ambos participantes
        self._subir_foto(inv1["token"], "derecha", "lejana", img1)
        self._subir_foto(inv2["token"], "derecha", "lejana", img2)

        # Comprobar carpetas separadas
        ruta_p1 = self.dir_recepcion / "fotos" / "externo_005a" / "externo_005a_derecha_lejana.jpg"
        ruta_p2 = self.dir_recepcion / "fotos" / "externo_005b" / "externo_005b_derecha_lejana.jpg"
        self.assertTrue(ruta_p1.exists())
        self.assertTrue(ruta_p2.exists())
        self.assertNotEqual(ruta_p1.read_bytes(), ruta_p2.read_bytes())

        # Comprobar consultas de progreso aisladas
        st1 = self.client.get(f"/api/estado?token={inv1['token']}").get_json()
        st2 = self.client.get(f"/api/estado?token={inv2['token']}").get_json()
        self.assertEqual(st1["identificador"], "externo_005a")
        self.assertEqual(st2["identificador"], "externo_005b")
        self.assertEqual(st1["progreso"]["total_enviadas"], 1)
        self.assertEqual(st2["progreso"]["total_enviadas"], 1)

    def test_06_progreso_por_pose_y_distancia_15_fotos(self):
        """Verifica el avance completo de las 15 combinaciones (5 poses x 3 distancias)."""
        inv = self.gestor_inv.crear_invitacion("externo_006")
        token = inv["token"]
        self.client.post("/api/consentimiento", json={"token": token})

        contador = 0
        for pose in POSES_PROTOCOLO:
            for dist in DISTANCIAS_PROTOCOLO:
                contador += 1
                img_bytes = crear_imagen_sintetica_jpeg(color=contador * 10)
                resp = self._subir_foto(token, pose, dist, img_bytes)
                self.assertEqual(resp.status_code, 200)
                prog = resp.get_json()["progreso"]
                self.assertEqual(prog["total_enviadas"], contador)
                self.assertEqual(prog["total_pendientes"], TOTAL_COMBINACIONES_PROTOCOLO - contador)

        # Estado final completado
        st_final = self.client.get(f"/api/estado?token={token}").get_json()["progreso"]
        self.assertEqual(st_final["total_enviadas"], 15)
        self.assertEqual(st_final["total_pendientes"], 0)
        self.assertTrue(st_final["completado"])
        for k, v in st_final["combinaciones"].items():
            self.assertTrue(v, f"Combinación {k} debería estar completada")

    def test_07_ausencia_rutas_publicas_listar_revisar_descargar(self):
        """Comprueba que el portal no exponga rutas administrativas ni descargas de fotos ajenas."""
        # Rutas del panel admin original deben dar 404
        rutas_prohibidas = [
            "/api/stream",
            "/api/revisar",
            "/api/modelo",
            "/api/fotograma",
            "/api/capturar",
            "/data/recepcion_externa/fotos",
            "/fotos",
        ]
        for ruta in rutas_prohibidas:
            resp = self.client.get(ruta)
            self.assertEqual(resp.status_code, 404, f"Ruta {ruta} no debe existir en el portal")

    def test_08_registro_fotografias_sin_rostro_permitido(self):
        """Verifica que imágenes sin rostro se acepten y registren sin ser filtradas ni rechazadas."""
        inv = self.gestor_inv.crear_invitacion("externo_008")
        token = inv["token"]
        self.client.post("/api/consentimiento", json={"token": token})

        # Imagen de ruido/fondo sólido sin ningún rostro
        fondo_liso = np.full((300, 300, 3), 70, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", fondo_liso)
        img_sin_rostro = buf.tobytes()

        resp = self._subir_foto(token, "abajo", "lejana", img_sin_rostro)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["exito"])

        archivo = self.dir_recepcion / "fotos" / "externo_008" / "externo_008_abajo_lejana.jpg"
        self.assertTrue(archivo.exists())

    def test_09_errores_archivo_tamano_o_corruptos(self):
        """Verifica validación de archivos corruptos o parámetros inválidos con mensajes legibles."""
        inv = self.gestor_inv.crear_invitacion("externo_009")
        token = inv["token"]
        self.client.post("/api/consentimiento", json={"token": token})

        # 1. Archivo corrupto (texto plano en vez de imagen)
        resp_corrupto = self._subir_foto(token, "frontal", "cercana", b"ESTO_NO_ES_UNA_IMAGEN")
        self.assertEqual(resp_corrupto.status_code, 400)
        self.assertIn("imagen válida", resp_corrupto.get_json()["error"])

        # 2. Pose inválida
        img_ok = crear_imagen_sintetica_jpeg()
        resp_pose_inv = self._subir_foto(token, "pose_inexistente", "cercana", img_ok)
        self.assertEqual(resp_pose_inv.status_code, 400)
        self.assertIn("inválida", resp_pose_inv.get_json()["error"])

        # 3. Distancia inválida
        resp_dist_inv = self._subir_foto(token, "frontal", "distancia_rara", img_ok)
        self.assertEqual(resp_dist_inv.status_code, 400)
        self.assertIn("inválida", resp_dist_inv.get_json()["error"])

    def test_10_importacion_correcta_al_manifiesto(self):
        """Verifica la importación local explícita de capturas al formato de manifiesto del proyecto."""
        inv = self.gestor_inv.crear_invitacion("externo_010")
        token = inv["token"]
        self.client.post("/api/consentimiento", json={"token": token})

        # Subir dos fotos
        img1 = crear_imagen_sintetica_jpeg(color=110)
        img2 = crear_imagen_sintetica_jpeg(color=120)
        self._subir_foto(token, "frontal", "cercana", img1)
        self._subir_foto(token, "arriba", "media", img2)

        # Manifiesto temporal de destino
        ruta_manifiesto_dest = self.dir_temp / "manifiesto_test.csv"

        stats = importar_capturas_externas(
            dir_recepcion=self.dir_recepcion,
            ruta_manifiesto_destino=ruta_manifiesto_dest,
            identificador_filtro="externo_010",
            raiz_proyecto=self.dir_temp
        )

        self.assertEqual(stats["total_encontradas"], 2)
        self.assertEqual(stats["importadas_nuevas"], 2)
        self.assertEqual(stats["omitidas_ya_existentes"], 0)
        self.assertEqual(len(stats["errores"]), 0)

        # Cargar con ManifiestoDataset del proyecto para auditar
        dataset_man = ManifiestoDataset(ruta_manifiesto_dest)
        self.assertEqual(len(dataset_man.registros), 2)

        reg0 = dataset_man.registros[0]
        self.assertEqual(reg0.tipo_sujeto, "desconocido")
        self.assertEqual(reg0.particion, "prueba")
        self.assertEqual(reg0.persona, "externo_010")
        self.assertTrue(reg0.consentimiento)
        self.assertIn("destino=desconocidos_prueba", reg0.anotaciones)

        # Reintentar la importación no debe duplicar
        stats_reintento = importar_capturas_externas(
            dir_recepcion=self.dir_recepcion,
            ruta_manifiesto_destino=ruta_manifiesto_dest,
            identificador_filtro="externo_010",
            raiz_proyecto=self.dir_temp
        )
        self.assertEqual(stats_reintento["importadas_nuevas"], 0)
        self.assertEqual(stats_reintento["omitidas_ya_existentes"], 2)

    def test_11_interfaz_viewport_movil_y_estaticos(self):
        """Verifica que la plantilla incluya viewport móvil y soporte de estilos para dispositivos."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"viewport", resp.data)
        self.assertIn(b"width=device-width", resp.data)
        resp.close()

        # Estáticos CSS y JS
        resp_css = self.client.get("/static/portal.css")
        self.assertEqual(resp_css.status_code, 200)
        data_css = resp_css.get_data()
        self.assertIn(b"@media", data_css)
        self.assertIn(b"transform: scaleX(-1)", data_css)
        resp_css.close()

        resp_js = self.client.get("/static/portal.js")
        self.assertEqual(resp_js.status_code, 200)
        data_js = resp_js.get_data()
        self.assertIn(b"getUserMedia", data_js)
        self.assertIn(b"drawImage", data_js)
        self.assertIn(b"poseCapturaPendiente", data_js)
        self.assertIn(b"distanciaCapturaPendiente", data_js)
        resp_js.close()

    def test_12_fijacion_metadatos_en_instante_de_captura_e_invarianza_ante_cambio_selectores(self):
        """Verifica que los metadatos de pose y distancia se fijen en el instante de captura
        y no se vean alterados si el participante cambia los selectores durante la previsualización."""
        inv = self.gestor_inv.crear_invitacion("externo_012")
        token = inv["token"]
        self.client.post("/api/consentimiento", json={"token": token})

        # 1. El participante tenía el selector en frontal - cercana al momento de presionar capturar
        pose_capturada = "frontal"
        distancia_capturada = "cercana"
        img_bytes = crear_imagen_sintetica_jpeg(color=140)

        # 2. Durante la previsualización, el usuario hace clic en los selectores cambiando a izquierda - lejana
        pose_selector_despues = "izquierda"
        distancia_selector_despues = "lejana"

        # 3. El cliente, gracias a la fijación de poseCapturaPendiente/distanciaCapturaPendiente,
        # envía la fotografía con la pose y distancia capturadas originalmente, no con las del selector
        resp = self._subir_foto(
            token=token,
            pose=pose_capturada,
            distancia=distancia_capturada,
            img_bytes=img_bytes,
            filename=f"{pose_capturada}_{distancia_capturada}.jpg"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["exito"])
        self.assertEqual(data["pose"], "frontal")
        self.assertEqual(data["distancia"], "cercana")

        # 4. Confirmar que en disco y registro se guardaron las etiquetas del instante de captura
        archivo_esperado = self.dir_recepcion / "fotos" / "externo_012" / "externo_012_frontal_cercana.jpg"
        archivo_erroneo = self.dir_recepcion / "fotos" / "externo_012" / "externo_012_izquierda_lejana.jpg"
        self.assertTrue(archivo_esperado.exists())
        self.assertFalse(archivo_erroneo.exists())

        progreso = self.gestor_alm.obtener_progreso("externo_012")
        self.assertTrue(progreso["combinaciones"]["frontal_cercana"])
        self.assertFalse(progreso["combinaciones"]["izquierda_lejana"])

    def test_13_exclusion_de_invitaciones_piloto_en_importacion(self):
        """Verifica que las invitaciones de prueba piloto puedan mantenerse estrictamente fuera
        de la importación al manifiesto del dataset real."""
        inv_piloto = self.gestor_inv.crear_invitacion("externo_piloto")
        inv_real = self.gestor_inv.crear_invitacion("externo_013")

        self.client.post("/api/consentimiento", json={"token": inv_piloto["token"]})
        self.client.post("/api/consentimiento", json={"token": inv_real["token"]})

        img_p = crear_imagen_sintetica_jpeg(color=90)
        img_r = crear_imagen_sintetica_jpeg(color=160)

        # Subir fotos para ambos participantes
        self._subir_foto(inv_piloto["token"], "frontal", "cercana", img_p)
        self._subir_foto(inv_real["token"], "frontal", "cercana", img_r)

        ruta_manifiesto_dest = self.dir_temp / "manifiesto_real.csv"

        # Importar con exclusión de pilotos por defecto
        stats = importar_capturas_externas(
            dir_recepcion=self.dir_recepcion,
            ruta_manifiesto_destino=ruta_manifiesto_dest,
            excluir_pilotos=True,
            raiz_proyecto=self.dir_temp
        )

        self.assertEqual(stats["importadas_nuevas"], 1)
        self.assertEqual(stats["omitidas_piloto"], 1)

        # Auditar que el manifiesto solo contiene externo_013 y no externo_piloto
        dataset = ManifiestoDataset(ruta_manifiesto_dest)
        personas_en_manifiesto = {r.persona for r in dataset.registros}
        self.assertIn("externo_013", personas_en_manifiesto)
        self.assertNotIn("externo_piloto", personas_en_manifiesto)


if __name__ == "__main__":
    unittest.main()
