"""
Pruebas unitarias para la coherencia estricta de particiones según el protocolo
del dataset propio (sesiones 1-3 -> entrenamiento, 4 -> validacion, 5 -> prueba).

Verifica:
1. Mapeo reglamentario de particiones para sesiones 1-3, 4 y 5.
2. Endpoint /api/protocolo/particiones.
3. Rechazo preventivo del servidor ante peticiones contradictorias (ej. sesion_05 + entrenamiento).
4. Guardado correcto de sesion_05 en prueba tanto para capturas válidas como para fallos.
5. Peticiones rechazadas no dejan archivos huérfanos en disco ni filas en el manifiesto.
6. Preservación y carga del manifiesto combinado local (manifiesto_local.csv) sin alterar registros.
7. Los ensayos admiten sesion_05 en entrenamiento.
8. Los manifiestos externos quedan fuera del protocolo propio.
"""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    MAPEO_SESION_PARTICION_PROPIO,
    obtener_particion_esperada_propio,
    validar_coherencia_sesion_particion_propio,
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import FotogramaBuffer, GestorCamaraWeb


class TestParticionesProtocolo(unittest.TestCase):

    def setUp(self):
        directorio_original = Path.cwd()
        directorio_temporal = TemporaryDirectory()
        self.addCleanup(directorio_temporal.cleanup)
        self.addCleanup(os.chdir, directorio_original)
        os.chdir(directorio_temporal.name)
        self.ruta_manifiesto_propio = Path("data/propio/manifiesto.csv")
        self.ruta_manifiesto_ensayo = Path("data/ensayos/manifiesto_ensayo.csv")

        self.gestor = GestorCamaraWeb(tamano_buffer=10)
        self.addCleanup(self.gestor.detener)

        self.app = crear_app(indice_inicial=-1)
        self.app.config["TESTING"] = True
        self.app.config["GESTOR_CAMARA"] = self.gestor
        self.client = self.app.test_client()

    def _inyectar_fotograma_al_buffer(self, id_frame: int, estado: EstadoDeteccion, valor_pixel: int = 120):
        frame = np.full((480, 640, 3), valor_pixel, dtype=np.uint8)
        item = FotogramaBuffer(
            id_fotograma=id_frame,
            timestamp=1000.0 + id_frame,
            frame_raw_bgr=frame,
            estado_deteccion=estado,
            recuadro=(50, 50, 100, 100) if estado == EstadoDeteccion.UN_ROSTRO else None,
            total_rostros=1 if estado == EstadoDeteccion.UN_ROSTRO else 0,
            frame_display_jpeg=b"fake_jpeg",
            ancho=640,
            alto=480
        )
        with self.gestor.lock_buffer:
            self.gestor.buffer.append(item)

    # -------------------------------------------------------------------------
    # a) Asignación de partición para sesiones 1-3, 4 y 5
    # -------------------------------------------------------------------------
    def test_a_asignacion_particiones_protocolo(self):
        """Verifica que el mapeo centralizado asigna las particiones correctas."""
        self.assertEqual(obtener_particion_esperada_propio("sesion_01"), "entrenamiento")
        self.assertEqual(obtener_particion_esperada_propio("sesion_02"), "entrenamiento")
        self.assertEqual(obtener_particion_esperada_propio("sesion_03"), "entrenamiento")
        self.assertEqual(obtener_particion_esperada_propio("sesion_04"), "validacion")
        self.assertEqual(obtener_particion_esperada_propio("sesion_05"), "prueba")
        self.assertEqual(obtener_particion_esperada_propio("sesion_cercana"), "entrenamiento")
        self.assertEqual(obtener_particion_esperada_propio("sesion_media"), "entrenamiento")
        self.assertEqual(obtener_particion_esperada_propio("sesion_lejana"), "entrenamiento")

        # Comprobación de validación
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_01", "entrenamiento")[0])
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_02", "entrenamiento")[0])
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_03", "entrenamiento")[0])
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_04", "validacion")[0])
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_05", "prueba")[0])
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_cercana", "entrenamiento")[0])
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_media", "entrenamiento")[0])
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_lejana", "entrenamiento")[0])
        self.assertFalse(validar_coherencia_sesion_particion_propio("sesion_cercana", "validacion")[0])
        self.assertFalse(validar_coherencia_sesion_particion_propio("sesion_lejana", "prueba")[0])

        # Ensayos o sesiones externas no se restringen
        self.assertTrue(validar_coherencia_sesion_particion_propio("sesion_ensayo_01", "prueba")[0])
        self.assertIsNone(obtener_particion_esperada_propio("sesion_desconocida"))

        # Endpoint HTTP
        resp = self.client.get("/api/protocolo/particiones")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["mapeo_sesion_particion"], MAPEO_SESION_PARTICION_PROPIO)

    # -------------------------------------------------------------------------
    # b) Rechazo del servidor ante peticiones contradictorias
    # -------------------------------------------------------------------------
    def test_b_rechazo_servidor_peticion_contradictoria(self):
        """Verifica que el servidor rechaza peticiones contradictorias con HTTP 400."""
        self._inyectar_fotograma_al_buffer(id_frame=1, estado=EstadoDeteccion.UN_ROSTRO)

        payload_invalido = {
            "id_fotograma": 1,
            "modo": "valida",
            "persona": "persona_1",
            "tipo_sujeto": "autorizado",
            "sesion": "sesion_05",
            "pose": "frontal",
            "particion": "entrenamiento",  # CONTRADICTORIO con sesion_05
            "consentimiento": True,
            "es_ensayo": False,
            "idempotency_key": "idemp_invalida_01"
        }

        resp = self.client.post("/api/capturar", json=payload_invalido)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["tipo"], "INCONSISTENCIA_PROTOCOLO")
        self.assertIn("Inconsistencia de protocolo", data["mensaje"])
        self.assertIn("prueba", data["mensaje"])

        # Verificar rechazo a nivel gestor/backend (ValueError)
        with self.assertRaises(ValueError):
            self.gestor.procesar_captura_web(
                id_fotograma=1,
                modo="valida",
                persona="persona_1",
                tipo_sujeto="autorizado",
                sesion="sesion_05",
                pose="frontal",
                particion="entrenamiento",
                consentimiento=True,
                es_ensayo=False
            )

        # Verificar rechazo en ManifiestoDataset.agregar_registro
        manifiesto = ManifiestoDataset(self.ruta_manifiesto_propio)
        registro_incoherente = RegistroCaptura(
            identificador="propio_persona_1_sesion_05_frontal_001",
            ruta="data/propio/imagenes/persona_1/sesion_05/frontal/p1.jpg",
            procedencia="propio",
            persona="persona_1",
            tipo_sujeto="autorizado",
            sesion="sesion_05",
            pose="frontal",
            particion="entrenamiento",  # Incoherente
            consentimiento=True
        )
        with self.assertRaises(ValueError):
            manifiesto.agregar_registro(registro_incoherente)

        ruta_alternativa_propia = Path.cwd() / "data/propio/sesiones/manifiesto.csv"
        with self.assertRaises(ValueError):
            ManifiestoDataset(ruta_alternativa_propia).agregar_registro(registro_incoherente)

        ruta_externa = Path.cwd() / "data/externo/manifiesto.csv"
        manifiesto_externo = ManifiestoDataset(ruta_externa)
        manifiesto_externo.agregar_registro(registro_incoherente)
        manifiesto_externo.guardar()
        self.assertEqual(ManifiestoDataset(ruta_externa).registros, [registro_incoherente])

    # -------------------------------------------------------------------------
    # c) Guardado correcto con sesion_05 y prueba (captura válida y fallo)
    # -------------------------------------------------------------------------
    def test_c_guardado_correcto_sesion05_prueba_valida_y_fallo(self):
        """Verifica guardado correcto en sesion_05 + prueba tanto para válida como para fallo."""
        # Captura Válida
        self._inyectar_fotograma_al_buffer(id_frame=10, estado=EstadoDeteccion.UN_ROSTRO)
        payload_valida = {
            "id_fotograma": 10,
            "modo": "valida",
            "persona": "persona_1",
            "tipo_sujeto": "autorizado",
            "sesion": "sesion_05",
            "pose": "frontal",
            "particion": "prueba",
            "consentimiento": True,
            "es_ensayo": False,
            "idempotency_key": "idemp_valida_ses05"
        }
        resp1 = self.client.post("/api/capturar", json=payload_valida)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.get_json()
        self.assertEqual(data1["status"], "ok")
        self.assertEqual(data1["decision"], "aprobada")

        # Captura con Fallo de detección
        self._inyectar_fotograma_al_buffer(id_frame=11, estado=EstadoDeteccion.SIN_ROSTRO)
        payload_fallo = {
            "id_fotograma": 11,
            "modo": "fallo",
            "persona": "persona_1",
            "tipo_sujeto": "autorizado",
            "sesion": "sesion_05",
            "pose": "izquierda",
            "particion": "prueba",
            "consentimiento": True,
            "es_ensayo": False,
            "idempotency_key": "idemp_fallo_ses05"
        }
        resp2 = self.client.post("/api/capturar", json=payload_fallo)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertEqual(data2["status"], "ok")
        self.assertEqual(data2["decision"], "pendiente")

        registros = ManifiestoDataset(self.ruta_manifiesto_propio).registros
        self.assertEqual(len(registros), 2)
        for registro, respuesta in zip(registros, (data1, data2)):
            self.assertEqual(registro.identificador, respuesta["identificador"])
            self.assertEqual(registro.sesion, "sesion_05")
            self.assertEqual(registro.particion, "prueba")
            ruta_imagen = Path(registro.ruta).resolve()
            self.assertEqual(ruta_imagen, Path(respuesta["ruta"]).resolve())
            self.assertTrue(ruta_imagen.is_relative_to(Path("data/propio/imagenes").resolve()))
            self.assertTrue(ruta_imagen.is_file())
        self.assertFalse(self.ruta_manifiesto_ensayo.exists())

    def test_ensayo_sesion05_en_entrenamiento(self):
        self._inyectar_fotograma_al_buffer(id_frame=12, estado=EstadoDeteccion.UN_ROSTRO)
        respuesta = self.client.post("/api/capturar", json={
            "id_fotograma": 12,
            "modo": "valida",
            "persona": "persona_1",
            "tipo_sujeto": "autorizado",
            "sesion": "sesion_05",
            "pose": "frontal",
            "particion": "entrenamiento",
            "consentimiento": True,
            "es_ensayo": True,
            "idempotency_key": "idemp_ensayo_ses05",
        })
        self.assertEqual(respuesta.status_code, 200)
        datos = respuesta.get_json()
        self.assertEqual(datos["status"], "ok")
        registros = ManifiestoDataset(self.ruta_manifiesto_ensayo).registros
        self.assertEqual(len(registros), 1)
        registro = registros[0]
        self.assertEqual(registro.identificador, datos["identificador"])
        self.assertEqual(registro.sesion, "sesion_05")
        self.assertEqual(registro.particion, "entrenamiento")
        ruta_imagen = Path(registro.ruta).resolve()
        self.assertTrue(ruta_imagen.is_relative_to(Path("data/ensayos/imagenes").resolve()))
        self.assertTrue(ruta_imagen.is_file())
        self.assertFalse(self.ruta_manifiesto_propio.exists())

    # -------------------------------------------------------------------------
    # d) Intento rechazado no deja archivos huérfanos ni filas
    # -------------------------------------------------------------------------
    def test_d_rechazo_no_deja_archivos_huerfanos_ni_filas(self):
        """Verifica que una petición rechazada no escriba archivos en disco ni filas en manifiesto."""
        self._inyectar_fotograma_al_buffer(id_frame=20, estado=EstadoDeteccion.UN_ROSTRO)

        ManifiestoDataset(self.ruta_manifiesto_propio).guardar()
        contenido_antes = self.ruta_manifiesto_propio.read_bytes()

        payload_invalido = {
            "id_fotograma": 20,
            "modo": "valida",
            "persona": "persona_1",
            "tipo_sujeto": "autorizado",
            "sesion": "sesion_05",
            "pose": "derecha",
            "particion": "entrenamiento",  # Inválido por protocolo
            "consentimiento": True,
            "es_ensayo": False,
            "idempotency_key": "idemp_rechazo_limpio"
        }

        resp = self.client.post("/api/capturar", json=payload_invalido)
        self.assertEqual(resp.status_code, 400)

        # Verificar que no se creó ningún archivo nuevo
        self.assertEqual(list(Path("data").rglob("*.jpg")), [])

        # Verificar que el manifiesto no fue alterado
        self.assertEqual(self.ruta_manifiesto_propio.read_bytes(), contenido_antes)

    # -------------------------------------------------------------------------
    # e) Preservación y carga del manifiesto combinado local
    # -------------------------------------------------------------------------
    def test_e_preservacion_manifiesto_local_combinado(self):
        """Verifica que se prefiera el manifiesto local y se conserven sus registros."""
        ManifiestoDataset(self.ruta_manifiesto_propio).guardar()
        ruta_local = self.ruta_manifiesto_propio.with_name("manifiesto_local.csv")
        fixture = ManifiestoDataset(ruta_local)
        for sesion, particion in (
            ("sesion_01", "entrenamiento"),
            ("sesion_04", "validacion"),
            ("sesion_05", "prueba"),
        ):
            registro = RegistroCaptura(
                identificador=f"persona_1_{sesion}_frontal_01",
                ruta=f"data/propio/imagenes/persona_1/{sesion}/frontal/muestra.jpg",
                procedencia="propio",
                persona="persona_1",
                tipo_sujeto="autorizado",
                sesion=sesion,
                pose="frontal",
                particion=particion,
                consentimiento=True,
            )
            fixture.agregar_registro(registro)
        fixture.guardar()
        contenido_local = ruta_local.read_bytes()

        manifiesto = ManifiestoDataset()
        self.assertEqual(manifiesto.ruta_archivo, ruta_local)
        self.assertEqual(manifiesto.registros, fixture.registros)

        # Validar aislamiento de sesiones y duplicados
        val_ses, err_ses = manifiesto.validar_aislamiento_sesiones()
        self.assertTrue(val_ses, f"Errores de aislamiento: {err_ses}")

        val_dup, err_dup = manifiesto.validar_duplicados_entre_particiones()
        self.assertTrue(val_dup, f"Errores de duplicados: {err_dup}")

        self.assertEqual(ruta_local.read_bytes(), contenido_local)


if __name__ == "__main__":
    unittest.main()
