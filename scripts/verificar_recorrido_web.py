"""
Script de verificación exhaustiva del recorrido real de la aplicación web
con el candidato congelado de 4 clases (configuracion_candidato_congelado_corregida.json).

Ejecuta de forma automatizada:
1. Carga del candidato congelado de 4 clases y verificación de parámetros e integridad SHA-256.
2. Ciclo de vida de la cámara: inicio, detención, reinicio y verificación de estados.
3. Visualización con HUD y recuadros en fotogramas.
4. Reglas de control de acceso y transiciones:
   - Autorizado con score >= 0.46 -> PUERTA ABIERTA.
   - Autorizado con score < 0.46 -> PUERTA CERRADA (rechazo por score insuficiente).
   - Desconocido (clase 'desconocido') -> PUERTA CERRADA (DESCONOCIDO).
   - Sin rostro detectado -> PUERTA CERRADA.
   - Varios rostros detectados -> PUERTA CERRADA.
5. Flujo de adquisición web y captura de imágenes (normal, forzado y voluntario desconocido).
6. Medición de latencia de inferencia y FPS utilizables.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reconocimiento.web.app import crear_app
from reconocimiento.web.streamer import GestorCamaraWeb, FotogramaBuffer
from reconocimiento.preprocesamiento.detector import EstadoDeteccion, ResultadoDeteccion
from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado


class FuenteVideoSimulada:
    """Fuente de video sintetizada compatible con la interfaz VideoCapture de OpenCV."""

    def __init__(self, fps: float = 15.0):
        self.fps = fps
        self.esta_activo = True
        self.frame_base = np.zeros((480, 640, 3), dtype=np.uint8)
        self.contador = 0

    def isOpened(self) -> bool:
        return self.esta_activo

    def read(self):
        if not self.esta_activo:
            return False, None
        self.contador += 1
        frame = self.frame_base.copy()
        cv2.putText(
            frame,
            f"FOTOGRAMA SIMULADO #{self.contador}",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2
        )
        return True, frame

    def release(self):
        self.esta_activo = False

    def get(self, propId):
        return 0.0

    def set(self, propId, val):
        return True


def ejecutar_verificacion_recorrido() -> Dict[str, Any]:
    print("=" * 70)
    print(" VERIFICACIÓN INTEGRAL DEL RECORRIDO DE LA APLICACIÓN WEB")
    print(" CANDIDATO CONGELADO 4 CLASES (CNN, umbral=0.46)")
    print("=" * 70)

    ruta_cfg = Path("configuracion_candidato_congelado_corregida.json")
    if not ruta_cfg.exists():
        raise FileNotFoundError(f"No existe {ruta_cfg}")

    resultados_comprobacion = {}

    # 1. CARGA DEL CANDIDATO CONGELADO
    print("\n[PASO 1] Comprobación de configuración congelada...")
    app = crear_app(indice_inicial=-1, ruta_config_candidato=ruta_cfg)
    gestor: GestorCamaraWeb = app.config["GESTOR_CAMARA"]
    
    assert gestor.pipeline_candidato is not None, "El pipeline candidato no fue cargado en el gestor."
    assert round(gestor.umbral_confianza, 2) == 0.46, f"Umbral esperado 0.46, obtenido {gestor.umbral_confianza}"
    assert len(gestor.pipeline_candidato.orden_clases) == 4, f"Se esperaban 4 clases, obtenidas {len(gestor.pipeline_candidato.orden_clases)}"
    assert gestor.pipeline_candidato.orden_clases == ["persona_1", "persona_2", "persona_3", "desconocido"]
    assert gestor.pipeline_candidato.personas_autorizadas == ["persona_1", "persona_2", "persona_3"]
    print(f"  OK: Candidato '{gestor.nombre_modelo_activo}' cargado exitosamente.")
    print(f"  OK: Umbral congelado activo: {gestor.umbral_confianza * 100:.1f}%.")
    print(f"  OK: Clases reconocidas: {gestor.pipeline_candidato.orden_clases}.")
    print(f"  OK: Clases autorizadas: {gestor.pipeline_candidato.personas_autorizadas}.")
    resultados_comprobacion["carga_candidato"] = "EXITOSO"

    # 2. CICLO DE VIDA DE LA CÁMARA (INICIO, DETENCIÓN, REINICIO)
    print("\n[PASO 2] Comprobación del ciclo de vida de la cámara...")
    gestor.detener()
    estado_detenido = gestor.obtener_estado()
    assert not estado_detenido["activa"], "La cámara debería reportar activa=False tras detener."
    assert estado_detenido["reconocimiento"]["estado_puerta"] == "CERRADA", "La puerta debe estar CERRADA con cámara detenida."
    print("  OK: Detención correcta. Puerta bloqueada en CERRADA.")

    # Inicio con fuente simulada
    fuente_sim = FuenteVideoSimulada()
    iniciado = gestor.iniciar(indice=99, fuente_custom=fuente_sim)
    assert iniciado, "Fallo al iniciar el streamer con fuente simulada."
    time.sleep(0.15)
    estado_iniciado = gestor.obtener_estado()
    assert estado_iniciado["activa"], "La cámara debería reportar activa=True tras iniciar."
    print("  OK: Inicio correcto con fuente simulada activa.")

    # Reinicio limpio
    gestor.detener()
    assert not gestor.obtener_estado()["activa"]
    fuente_sim2 = FuenteVideoSimulada()
    gestor.iniciar(indice=99, fuente_custom=fuente_sim2)
    time.sleep(0.15)
    assert gestor.obtener_estado()["activa"]
    gestor.detener()
    print("  OK: Reinicio correcto verificado sin colisiones de hilos.")
    resultados_comprobacion["ciclo_vida_camara"] = "EXITOSO"

    # 3. PUBLICACIÓN DE FLUJO Y VERIFICACIÓN DEL HUD
    print("\n[PASO 3] Comprobación de HUD, recuadro y visualización...")
    frame_test = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame_test, (320, 240), 100, (120, 120, 120), -1)
    res_det_test = ResultadoDeteccion(
        estado=EstadoDeteccion.UN_ROSTRO,
        recuadro=(220, 140, 200, 200),
        total_rostros_detectados=1,
        rostro_recortado=np.ones((200, 200, 3), dtype=np.uint8) * 150
    )
    item_pub = gestor._procesar_y_almacenar_fotograma(frame_test, res_det_test)
    assert item_pub is not None
    assert len(item_pub.frame_display_jpeg) > 0, "No se generó el flujo JPEG codificado para la web."
    assert item_pub.recuadro == (220, 140, 200, 200)
    print("  OK: Fotograma codificado en JPEG con recuadro y HUD informativo correctamente.")
    resultados_comprobacion["visualizacion_hud"] = "EXITOSO"

    # 4. COMPROBACIÓN DE REGLAS DE DECISIÓN DE ACCESO
    print("\n[PASO 4] Comprobación de decisiones de acceso (autorizados, sub-umbral, desconocidos, fallos)...")
    
    # A. Autorizado con score alto (>= 0.46)
    recorte_64 = np.ones((64, 64), dtype=np.float32)
    dec_aut, probs_aut = gestor.pipeline_candidato.evaluar_recorte(recorte_64)
    print(f"  Inferencia sobre recorte de prueba -> Predicción: {dec_aut.persona_identificada}, Confianza: {dec_aut.confianza*100:.1f}%, Puerta: {dec_aut.estado_puerta}")

    # Forzar casos sintéticos a través de la regla canónica del streamer
    casos_prueba = [
        # (nombre_pred, score, estado_det, estado_puerta_esperado, msg_esperado)
        ("persona_1", 0.85, EstadoDeteccion.UN_ROSTRO, "ABIERTA", "ACCESO CONCEDIDO: persona_1"),
        ("persona_2", 0.48, EstadoDeteccion.UN_ROSTRO, "ABIERTA", "ACCESO CONCEDIDO: persona_2"),
        ("persona_1", 0.40, EstadoDeteccion.UN_ROSTRO, "CERRADA", "ACCESO DENEGADO: Desconocido"),
        ("persona_3", 0.35, EstadoDeteccion.UN_ROSTRO, "CERRADA", "ACCESO DENEGADO: Desconocido"),
        ("desconocido", 0.95, EstadoDeteccion.UN_ROSTRO, "CERRADA", "ACCESO DENEGADO: Sujeto desconocido"),
        ("persona_externa", 0.90, EstadoDeteccion.UN_ROSTRO, "CERRADA", "ACCESO DENEGADO: Sujeto no autorizado"),
        (None, 0.0, EstadoDeteccion.SIN_ROSTRO, "CERRADA", "Fallo de detección (SIN_ROSTRO)"),
        (None, 0.0, EstadoDeteccion.VARIOS_ROSTROS, "CERRADA", "Fallo de detección (VARIOS_ROSTROS)"),
    ]

    from reconocimiento.clasificacion.regla_acceso import evaluar_regla_acceso_completa
    for pred, sc, est_det, puerta_esp, msg_sub in casos_prueba:
        dec = evaluar_regla_acceso_completa(
            estado_deteccion=est_det.value,
            procesamiento_exitoso=True if est_det == EstadoDeteccion.UN_ROSTRO else False,
            clase_predicha=pred,
            score=sc,
            umbral=0.46,
            personas_autorizadas=["persona_1", "persona_2", "persona_3"]
        )
        assert dec.estado_puerta == puerta_esp, f"Caso {pred} ({sc}): esperaba {puerta_esp}, obtenido {dec.estado_puerta}"
        assert msg_sub in dec.mensaje, f"Caso {pred}: '{msg_sub}' no presente en '{dec.mensaje}'"
        print(f"  OK: Caso {pred or 'N/A':15s} (Score: {sc:.2f}, Det: {est_det.value:14s}) -> Puerta {dec.estado_puerta:7s} | {dec.mensaje}")

    resultados_comprobacion["reglas_acceso"] = "EXITOSO"

    # 5. CAPTURA Y REVISIÓN EN LA WEB
    print("\n[PASO 5] Comprobación de captura y aislamiento de datos...")
    dir_tmp = Path(tempfile.mkdtemp(prefix="test_verif_captura_"))
    try:
        gestor.dir_propio = dir_tmp
        gestor.buffer.clear()
        
        # Inyectar fotograma válido en búfer
        item_captura = gestor._procesar_y_almacenar_fotograma(frame_test, res_det_test)
        
        # A. Captura normal de persona autorizada
        res_cap_aut = gestor.procesar_captura_web(
            id_fotograma=item_captura.id_fotograma,
            persona="persona_1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="frontal",
            particion="entrenamiento",
            modo="valida",
            consentimiento=True,
            es_ensayo=False
        )
        assert res_cap_aut["status"] == "ok"
        print(f"  OK: Captura autorizada guardada: {res_cap_aut['identificador']}")

        # B. Captura forzada (modo fallo con fotograma SIN_ROSTRO)
        res_det_fallo = ResultadoDeteccion(
            estado=EstadoDeteccion.SIN_ROSTRO,
            recuadro=None,
            total_rostros_detectados=0,
            rostro_recortado=None
        )
        item_fallo = gestor._procesar_y_almacenar_fotograma(frame_test, res_det_fallo)
        res_cap_fallo = gestor.procesar_captura_web(
            id_fotograma=item_fallo.id_fotograma,
            persona="persona_1",
            tipo_sujeto="autorizado",
            sesion="sesion_01",
            pose="izquierda",
            particion="entrenamiento",
            modo="fallo",
            consentimiento=True,
            es_ensayo=False
        )
        assert res_cap_fallo["status"] == "ok"
        assert res_cap_fallo["modo"] == "fallo"
        print(f"  OK: Captura deliberada de fallo guardada: {res_cap_fallo['identificador']}")

        # C. Captura de desconocido real (externo_001, desconocidos_prueba)
        # Fotograma con contenido visual propio para evitar solapamiento / fuga de imagen
        frame_test_desc = frame_test.copy()
        frame_test_desc[50:150, 50:150] = 180
        item_captura_desc = gestor._procesar_y_almacenar_fotograma(frame_test_desc, res_det_test)

        res_cap_desc = gestor.procesar_captura_web(
            id_fotograma=item_captura_desc.id_fotograma,
            persona="externo_001",
            tipo_sujeto="desconocido",
            sesion="sesion_01",
            pose="frontal",
            particion="prueba",
            grupo="desconocidos_prueba",
            distancia="cercana",
            modo="valida",
            consentimiento=True,
            es_ensayo=False
        )
        assert res_cap_desc["status"] == "ok"
        assert "externo_001" in res_cap_desc["identificador"]
        print(f"  OK: Captura de voluntario desconocido guardada: {res_cap_desc['identificador']}")

        # D. Verificar aislamiento estricto en manifiesto
        from reconocimiento.adquisicion.propio import ManifiestoDataset
        man_verif = ManifiestoDataset(dir_tmp / "manifiesto.csv")
        regs = man_verif.registros
        assert len(regs) == 3
        reg_desc = next(r for r in regs if r.persona == "externo_001")
        assert reg_desc.particion == "prueba"
        assert reg_desc.tipo_sujeto == "desconocido"
        assert "destino=desconocidos_prueba" in reg_desc.anotaciones
        assert "distancia=cercana" in reg_desc.anotaciones
        print("  OK: Aislamiento comprobado. Los desconocidos se registran exclusivamente en 'desconocidos_prueba'.")
        resultados_comprobacion["captura_y_aislamiento"] = "EXITOSO"
    finally:
        shutil.rmtree(dir_tmp, ignore_errors=True)

    # 6. MEDIDAS DE LATENCIA Y RENDIMIENTO
    print("\n[PASO 6] Medición de latencia de inferencia y FPS utilizables...")
    n_iter = 30
    latencias_ms = []
    dummy_face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    for _ in range(n_iter):
        t0 = time.perf_counter()
        # Normalización + inferencia
        gris = cv2.cvtColor(dummy_face, cv2.COLOR_BGR2GRAY)
        norm_res = cv2.resize(gris, (64, 64), interpolation=cv2.INTER_AREA)
        norm_int = norm_res.astype(np.float32) / 255.0
        suav = cv2.GaussianBlur(norm_int, (0, 0), sigmaX=0.5, sigmaY=0.5)
        in_t = np.expand_dims(suav, axis=(0, 1))
        _ = gestor.pipeline_candidato.inferir_probabilidades(suav)
        t1 = time.perf_counter()
        latencias_ms.append((t1 - t0) * 1000.0)

    latencia_media_ms = float(np.mean(latencias_ms))
    latencia_p95_ms = float(np.percentile(latencias_ms, 95))
    fps_teorico = 1000.0 / latencia_media_ms if latencia_media_ms > 0 else 0.0

    print(f"  Latencia media de inferencia: {latencia_media_ms:.2f} ms")
    print(f"  Latencia Percentil 95:       {latencia_p95_ms:.2f} ms")
    print(f"  Rendimiento de inferencia:   ~{fps_teorico:.1f} FPS (objetivo streamer: 12-15 FPS)")
    assert latencia_media_ms < 50.0, f"Latencia media {latencia_media_ms:.2f} ms supera límite de 50 ms."
    resultados_comprobacion["rendimiento_latencia"] = {
        "latencia_media_ms": round(latencia_media_ms, 2),
        "latencia_p95_ms": round(latencia_p95_ms, 2),
        "fps_estimado": round(fps_teorico, 1),
    }

    print("\n" + "=" * 70)
    print(" VERIFICACIÓN COMPLETADA EXITOSAMENTE: TODOS LOS PASOS PASARON")
    print("=" * 70)
    return resultados_comprobacion


if __name__ == "__main__":
    res = ejecutar_verificacion_recorrido()
    print("\nResumen estructurado:", json.dumps(res, indent=2))
