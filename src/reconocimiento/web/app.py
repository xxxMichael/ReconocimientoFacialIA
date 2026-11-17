"""
Aplicación web Flask para la adquisición de imágenes faciales y control de acceso simulado.
Permite seleccionar cámara, observar la vista previa con detección facial en tiempo real,
reconocer identidades autorizadas y simular la apertura o cierre de la puerta según
las reglas de rechazo definidas.
"""
import atexit
import base64
import json
from pathlib import Path
import time
from typing import Optional
import cv2
from flask import Flask, Response, jsonify, render_template, request

from reconocimiento.adquisicion.camara import sondear_camaras
from reconocimiento.adquisicion.propio import ManifiestoDataset
from reconocimiento.web.streamer import GestorCamaraWeb, calcular_progreso_sesion


def crear_app(indice_inicial: int = 1) -> Flask:
    """Crea y configura la aplicación Flask para la interfaz web de adquisición y acceso."""
    directorio_base = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(directorio_base / "templates"),
        static_folder=str(directorio_base / "static")
    )
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    gestor = GestorCamaraWeb()
    app.config["GESTOR_CAMARA"] = gestor

    # Cargar explícitamente el modelo seleccionado por validación
    ruta_seleccion = Path("resultados/modelo_seleccionado.json")
    if not ruta_seleccion.exists():
        try:
            from reconocimiento.clasificacion.seleccion import seleccionar_mejor_modelo_propio
            seleccionar_mejor_modelo_propio(dir_resultados=Path("resultados"), ruta_destino=ruta_seleccion)
        except Exception as e_sel:
            print(f"[RECONOCIMIENTO] Aviso al generar selección de modelos: {e_sel}")

    modelo_cargado_exitosamente = False
    if ruta_seleccion.exists():
        try:
            with open(ruta_seleccion, "r", encoding="utf-8") as f_sel:
                info_sel = json.load(f_sel)
            modelo_elegido = info_sel.get("modelo_elegido", "")
            tiene_calibracion = info_sel.get("calibracion_completa", True)
            dir_mod_str = info_sel.get("directorio_modelo", "")

            if tiene_calibracion and modelo_elegido and (modelo_elegido.lower() != "ninguno") and dir_mod_str:
                dir_mod = Path(dir_mod_str)
                if dir_mod.exists():
                    val_u = info_sel.get("umbral_recomendado")
                    if val_u is None:
                        val_u = info_sel.get("umbral_congelado", info_sel.get("umbral", 0.65))
                    umbral_mod = float(val_u)
                    res_load = gestor.cargar_modelo(directorio_modelo=dir_mod, umbral=umbral_mod)
                    modelo_cargado_exitosamente = res_load.get("status") == "ok"
                    print(f"[RECONOCIMIENTO] Modelo seleccionado por validación: {modelo_elegido.upper()} "
                          f"(Val Acc: {info_sel.get('exactitud_validacion_pct')}%, Umbral: {umbral_mod*100:.0f}%) -> {res_load.get('mensaje')}")
        except Exception as e_load_sel:
            print(f"[RECONOCIMIENTO] Fallo al cargar modelo seleccionado: {e_load_sel}")

    if not modelo_cargado_exitosamente:
        # Si ningún modelo cuenta con calibración completa, la interfaz web sigue 100%
        # disponible para captura de imágenes, pero la autorización de acceso permanece
        # estrictamente desactivada (puerta CERRADA).
        gestor.desactivar_modelo("Ningún modelo cuenta con calibración completa. Autorización desactivada (Puerta Cerrada).")
        print("[RECONOCIMIENTO] Autorización desactivada: no hay modelos con calibración completa disponible.")

    # Intentar encender la cámara inicial silenciosamente si el índice es >= 0
    if indice_inicial >= 0:
        try:
            gestor.iniciar(indice=indice_inicial)
        except Exception as e:
            print(f"[AVISO] No se pudo iniciar automáticamente la cámara en índice {indice_inicial}: {e}")
            print("Puede seleccionarla o sondear dispositivos desde la interfaz web.")

    # Garantizar liberación al cerrar el proceso Flask
    atexit.register(gestor.detener)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/camaras", methods=["GET"])
    def api_camaras():
        max_idx = request.args.get("max_indices", default=5, type=int)
        dispositivos = sondear_camaras(max_indices=max_idx)
        return jsonify({"dispositivos": dispositivos})

    @app.route("/api/camara/iniciar", methods=["POST"])
    def api_camara_iniciar():
        data = request.get_json(silent=True) or {}
        indice = int(data.get("indice", 1))
        try:
            gestor.iniciar(indice=indice)
            return jsonify({"status": "ok", "mensaje": f"Cámara en índice {indice} iniciada."})
        except Exception as e:
            return jsonify({"status": "error", "mensaje": str(e)}), 500

    @app.route("/api/camara/detener", methods=["POST"])
    def api_camara_detener():
        try:
            gestor.detener()
            return jsonify({"status": "ok", "mensaje": "Cámara detenida y liberada."})
        except Exception as e:
            return jsonify({"status": "error", "mensaje": str(e)}), 500

    @app.route("/api/estado", methods=["GET"])
    def api_estado():
        return jsonify(gestor.obtener_estado())

    @app.route("/api/reconocimiento/estado", methods=["GET"])
    def api_reconocimiento_estado():
        """Retorna el estado actual de reconocimiento y puerta simulada."""
        estado = gestor.obtener_estado()
        return jsonify(estado.get("reconocimiento", {}))

    @app.route("/api/reconocimiento/seleccion", methods=["GET"])
    def api_reconocimiento_seleccion():
        """Retorna el informe del modelo elegido por validación y su justificación."""
        ruta_seleccion = Path("resultados/modelo_seleccionado.json")
        if ruta_seleccion.exists():
            try:
                with open(ruta_seleccion, "r", encoding="utf-8") as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({"status": "error", "mensaje": str(e)}), 500
        return jsonify({"status": "error", "mensaje": "No se ha generado modelo_seleccionado.json"}), 404

    @app.route("/api/reconocimiento/configurar", methods=["POST"])
    def api_reconocimiento_configurar():
        """Configura dinámicamente el modelo activo o el umbral de confianza de rechazo."""
        data = request.get_json(silent=True) or {}
        modelo = data.get("modelo", "").strip().lower()
        umbral = data.get("umbral", None)

        # Si el usuario solo envía umbral (movió el deslizador manualmente sin cambiar de modelo):
        if umbral is not None and not modelo:
            gestor.actualizar_umbral(float(umbral))
            return jsonify({
                "status": "ok",
                "mensaje": f"Umbral actualizado a {gestor.umbral_confianza*100:.1f}%.",
                "umbral": round(gestor.umbral_confianza * 100, 1),
                "umbral_decimal": gestor.umbral_confianza,
                "reconocimiento": gestor.obtener_estado().get("reconocimiento", {})
            })

        if modelo:
            if modelo == "ninguno":
                res = gestor.desactivar_modelo("Modelo desactivado manualmente.")
                return jsonify(res)

            ruta_m = Path(f"resultados/{modelo}_propio")
            if not ruta_m.exists():
                ruta_m = Path(f"resultados/{modelo}")
            if not ruta_m.exists():
                return jsonify({"status": "error", "mensaje": f"No se encontró carpeta de modelo: {ruta_m}"}), 404

            # Buscar umbral calibrado del modelo de destino
            umbral_destino = None
            ruta_calib = ruta_m / "calibracion_umbral.json"
            if ruta_calib.exists():
                try:
                    with open(ruta_calib, "r", encoding="utf-8") as f_c:
                        datos_c = json.load(f_c)
                    val_u = datos_c.get("umbral_congelado")
                    if val_u is None:
                        val_u = datos_c.get("umbral_recomendado")
                    if val_u is None:
                        val_u = datos_c.get("umbral")
                    if val_u is not None:
                        umbral_destino = float(val_u)
                except Exception as e_c:
                    print(f"[RECONOCIMIENTO] Error al leer calibración de {ruta_m}: {e_c}")

            # Al cambiar de modelo se carga estrictamente el umbral calibrado del modelo destino.
            # Si el request especificó un umbral explícito y el modelo no cuenta con calibración, se usa el especificado.
            if umbral_destino is not None:
                umbral_a_usar = umbral_destino
            elif umbral is not None:
                umbral_a_usar = float(umbral)
            else:
                umbral_a_usar = gestor.umbral_confianza

            res = gestor.cargar_modelo(directorio_modelo=ruta_m, umbral=umbral_a_usar)
            # Garantizar que la respuesta devuelva el nuevo umbral cargado para actualizar el deslizador web
            res["umbral"] = round(gestor.umbral_confianza * 100, 1)
            res["umbral_decimal"] = gestor.umbral_confianza
            return jsonify(res)

        return jsonify({
            "status": "ok",
            "reconocimiento": gestor.obtener_estado().get("reconocimiento", {})
        })

    @app.route("/api/frame/latest", methods=["GET"])
    def api_frame_latest():
        """
        Retorna el fotograma más reciente junto con su id_fotograma, metadatos de detección,
        reconocimiento facial y simulación de puerta.
        """
        item = gestor.obtener_ultimo_fotograma()
        if item is None:
            estado = gestor.obtener_estado()
            return jsonify({
                "disponible": False,
                "estado_camara": estado,
                "mensaje": "No hay fotogramas disponibles (cámara detenida o en calentamiento)."
            })

        b64_img = base64.b64encode(item.frame_display_jpeg).decode("ascii")
        return jsonify({
            "disponible": True,
            "id_fotograma": item.id_fotograma,
            "timestamp": item.timestamp,
            "estado": item.estado_deteccion.value,
            "recuadro": item.recuadro,
            "total_rostros": item.total_rostros,
            "ancho": item.ancho,
            "alto": item.alto,
            "fps": gestor.fps_efectivo,
            "persona_reconocida": item.persona_reconocida,
            "confianza": item.confianza,
            "es_autorizado": item.es_autorizado,
            "estado_puerta": item.estado_puerta,
            "mensaje_acceso": item.mensaje_acceso,
            "image_base64": b64_img
        })

    @app.route("/video_feed")
    def video_feed():
        """Flujo continuo MJPEG alternativo."""
        def generar_mjpeg():
            ultimo_id = -1
            while True:
                item = gestor.obtener_ultimo_fotograma()
                if item is not None and item.id_fotograma != ultimo_id:
                    ultimo_id = item.id_fotograma
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + item.frame_display_jpeg + b"\r\n"
                    )
                else:
                    time.sleep(0.04)

        return Response(
            generar_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/api/progreso", methods=["GET"])
    def api_progreso():
        """Recupera el progreso de capturas desde el archivo de manifiesto en disco."""
        es_ensayo = request.args.get("es_ensayo", default="true").lower() in ("true", "1", "si")
        persona = request.args.get("persona", default="persona_1").strip()
        sesion = request.args.get("sesion", default="sesion_01").strip()

        if es_ensayo:
            ruta_manifiesto = Path("data/ensayos/manifiesto_ensayo.csv")
        else:
            ruta_manifiesto = Path("data/propio/manifiesto.csv")

        manifiesto = ManifiestoDataset(ruta_manifiesto)
        # Sincronización automática: si se borraron imágenes manualmente del disco, limpiar el manifiesto
        registros_reales = [r for r in manifiesto.registros if Path(r.ruta).is_file()]
        if len(registros_reales) != len(manifiesto.registros):
            manifiesto.registros = registros_reales
            try:
                manifiesto.guardar()
            except Exception as e_sinc:
                print(f"[MANIFIESTO] Aviso al sincronizar archivos borrados: {e_sinc}")

        progreso = calcular_progreso_sesion(manifiesto, persona, sesion)
        return jsonify({
            "status": "ok",
            "manifiesto": str(ruta_manifiesto),
            "es_ensayo": es_ensayo,
            "progreso": progreso
        })

    @app.route("/api/captura/congelar", methods=["POST"])
    def api_captura_congelar():
        """
        Congela el fotograma mostrado para inspección previa a la confirmación de guardado.
        Genera el recorte exacto que recibiría el clasificador.
        """
        data = request.get_json(silent=True) or {}
        id_fotograma = data.get("id_fotograma")
        item = None
        if id_fotograma is not None and int(id_fotograma) > 0:
            item = gestor.obtener_fotograma_por_id(int(id_fotograma))
        if item is None:
            item = gestor.obtener_ultimo_fotograma()

        if item is None:
            return jsonify({"status": "error", "mensaje": "No hay fotograma disponible en el búfer."}), 404

        frame_raw = item.frame_raw_bgr.copy()
        h, w = frame_raw.shape[:2]

        _, buf_raw = cv2.imencode(".jpg", frame_raw, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_raw = base64.b64encode(buf_raw.tobytes()).decode("ascii")

        recorte_b64 = ""
        recuadro = item.recuadro
        if recuadro is not None:
            try:
                recorte = gestor.detector.recortar_con_margen_seguro(frame_raw, recuadro)
                if recorte is not None and recorte.size > 0:
                    _, buf_rec = cv2.imencode(".jpg", recorte, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    recorte_b64 = base64.b64encode(buf_rec.tobytes()).decode("ascii")
            except Exception as e_crop:
                print(f"[CONGELAR] Error al generar recorte: {e_crop}")

        return jsonify({
            "status": "ok",
            "id_fotograma": item.id_fotograma,
            "timestamp": item.timestamp,
            "estado": item.estado_deteccion.value,
            "recuadro": item.recuadro,
            "total_rostros": item.total_rostros,
            "ancho": w,
            "alto": h,
            "image_base64": b64_raw,
            "recorte_base64": recorte_b64
        })

    @app.route("/api/capturar", methods=["POST"])
    def api_capturar():
        """Procesa una solicitud de captura asociando el id_fotograma mostrado."""
        data = request.get_json(silent=True) or {}

        try:
            id_fotograma = int(data.get("id_fotograma", 0))
            modo = str(data.get("modo", "valida")).lower()
            persona = str(data.get("persona", "persona_1")).strip()
            tipo_sujeto = str(data.get("tipo_sujeto", "autorizado")).strip()
            sesion = str(data.get("sesion", "sesion_01")).strip()
            pose = str(data.get("pose", "frontal")).strip()
            particion = str(data.get("particion", "entrenamiento")).strip()
            consentimiento = bool(data.get("consentimiento", False))
            es_ensayo = bool(data.get("es_ensayo", True))
            idempotency_key = str(data.get("idempotency_key", "")).strip()
            caja = data.get("caja", None)
            decision = data.get("decision", None)
            origen_caja = data.get("origen_caja", None)

            resultado = gestor.procesar_captura_web(
                id_fotograma=id_fotograma,
                modo=modo,
                persona=persona,
                tipo_sujeto=tipo_sujeto,
                sesion=sesion,
                pose=pose,
                particion=particion,
                consentimiento=consentimiento,
                es_ensayo=es_ensayo,
                idempotency_key=idempotency_key,
                caja=caja,
                decision=decision,
                origen_caja=origen_caja
            )
            return jsonify(resultado)

        except KeyError as ke:
            return jsonify({"status": "error", "tipo": "CADUCADO", "mensaje": str(ke)}), 410
        except ValueError as ve:
            return jsonify({"status": "error", "tipo": "VALIDACION", "mensaje": str(ve)}), 400
        except FileExistsError as fe:
            return jsonify({"status": "error", "tipo": "COLISION", "mensaje": str(fe)}), 409
        except Exception as e:
            return jsonify({"status": "error", "tipo": "INTERNO", "mensaje": f"Error inesperado: {e}"}), 500

    @app.route("/api/revision/capturas", methods=["GET"])
    def api_revision_capturas():
        """Retorna lista de capturas del manifiesto con metadatos de revisión y filtros."""
        es_ensayo = request.args.get("es_ensayo", default="false").lower() in ("true", "1", "si")
        persona_f = request.args.get("persona", default="").strip()
        sesion_f = request.args.get("sesion", default="").strip()
        pose_f = request.args.get("pose", default="").strip()
        particion_f = request.args.get("particion", default="").strip()
        decision_f = request.args.get("decision", default="").strip()

        ruta_manifiesto = Path("data/ensayos/manifiesto_ensayo.csv") if es_ensayo else Path("data/propio/manifiesto.csv")
        manifiesto = ManifiestoDataset(ruta_manifiesto)

        items = []
        for reg in manifiesto.registros:
            if persona_f and persona_f.lower() != "todas" and reg.persona != persona_f:
                continue
            if sesion_f and sesion_f.lower() != "todas" and reg.sesion != sesion_f:
                continue
            if pose_f and pose_f.lower() != "todas" and reg.pose != pose_f:
                continue
            if particion_f and particion_f.lower() != "todas" and reg.particion != particion_f:
                continue
            if decision_f and decision_f.lower() != "todas" and reg.decision_humana != decision_f:
                continue

            existe_archivo = Path(reg.ruta).is_file()
            caja_actual = reg.caja_aprobada
            origen = reg.origen_caja

            # Si es foto antigua sin caja registrada, calcular detección bajo demanda
            es_recalculada = False
            if caja_actual is None and existe_archivo:
                try:
                    img_disco = cv2.imread(str(reg.ruta))
                    if img_disco is not None:
                        res_det = gestor.detector.detectar(img_disco)
                        if res_det.recuadro is not None:
                            caja_actual = res_det.recuadro
                            origen = "recalculada"
                            es_recalculada = True
                except Exception as e_det:
                    print(f"[REVISION] Error al recalcular detección para {reg.identificador}: {e_det}")

            items.append({
                "identificador": reg.identificador,
                "persona": reg.persona,
                "sesion": reg.sesion,
                "pose": reg.pose,
                "particion": reg.particion,
                "tipo_sujeto": reg.tipo_sujeto,
                "ruta": reg.ruta,
                "existe_disco": existe_archivo,
                "decision_humana": reg.decision_humana,
                "estado_historico_detector": reg.estado_historico_detector,
                "caja": list(caja_actual) if caja_actual else None,
                "origen_caja": origen if origen else ("recalculada" if es_recalculada else ""),
                "es_recalculada": es_recalculada,
                "rotulo_caja": "recalculada" if es_recalculada else (origen if origen else "sin caja"),
                "anotaciones": reg.anotaciones
            })

        return jsonify({
            "status": "ok",
            "total": len(items),
            "es_ensayo": es_ensayo,
            "manifiesto": str(ruta_manifiesto),
            "capturas": items
        })

    @app.route("/api/revision/detalle/<identificador>", methods=["GET"])
    def api_revision_detalle(identificador: str):
        """Retorna imagen original y recorte exacto que recibiría el clasificador para inspección detallada."""
        es_ensayo = request.args.get("es_ensayo", default="false").lower() in ("true", "1", "si")
        ruta_manifiesto = Path("data/ensayos/manifiesto_ensayo.csv") if es_ensayo else Path("data/propio/manifiesto.csv")
        manifiesto = ManifiestoDataset(ruta_manifiesto)
        reg = manifiesto.buscar_registro(identificador)
        if reg is None:
            return jsonify({"status": "error", "mensaje": f"Captura '{identificador}' no encontrada en manifiesto."}), 404

        p_img = Path(reg.ruta)
        if not p_img.is_file():
            return jsonify({"status": "error", "mensaje": f"Archivo de imagen no existe en disco: {reg.ruta}"}), 404

        img = cv2.imread(str(p_img))
        if img is None:
            return jsonify({"status": "error", "mensaje": "No se pudo decodificar la imagen de disco."}), 500

        h, w = img.shape[:2]
        caja = reg.caja_aprobada
        origen = reg.origen_caja
        es_recalculada = False

        if caja is None:
            res_det = gestor.detector.detectar(img)
            if res_det.recuadro is not None:
                caja = res_det.recuadro
                origen = "recalculada"
                es_recalculada = True
            else:
                origen = "sin_rostro"

        # Generar recorte exacto que recibiría el clasificador
        recorte_b64 = ""
        if caja is not None:
            try:
                recorte = gestor.detector.recortar_con_margen_seguro(img, caja)
                if recorte is not None and recorte.size > 0:
                    _, buf_rec = cv2.imencode(".jpg", recorte, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    recorte_b64 = base64.b64encode(buf_rec.tobytes()).decode("ascii")
            except Exception as e_crop:
                print(f"[REVISION] Error al generar recorte para {identificador}: {e_crop}")

        _, buf_img = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_img = base64.b64encode(buf_img.tobytes()).decode("ascii")

        return jsonify({
            "status": "ok",
            "identificador": reg.identificador,
            "persona": reg.persona,
            "sesion": reg.sesion,
            "pose": reg.pose,
            "particion": reg.particion,
            "tipo_sujeto": reg.tipo_sujeto,
            "ruta": reg.ruta,
            "ancho": w,
            "alto": h,
            "decision_humana": reg.decision_humana,
            "estado_historico_detector": reg.estado_historico_detector,
            "caja": list(caja) if caja else None,
            "origen_caja": origen,
            "es_recalculada": es_recalculada,
            "aviso_recalculada": "recalculada: puede diferir de la que apareció en vivo" if es_recalculada else "",
            "imagen_base64": b64_img,
            "recorte_base64": recorte_b64
        })

    @app.route("/api/revision/decidir", methods=["POST"])
    def api_revision_decidir():
        """Aplica la decisión humana (Aprobar, Corregir recuadro, Excluir, Pendiente) de forma atómica."""
        data = request.get_json(silent=True) or {}
        identificador = str(data.get("identificador", "")).strip()
        decision = str(data.get("decision", "")).strip().lower()
        caja = data.get("caja", None)
        es_ensayo = bool(data.get("es_ensayo", False))

        if not identificador:
            return jsonify({"status": "error", "mensaje": "Se requiere 'identificador'."}), 400

        if decision not in {"aprobada", "corregida", "excluida", "pendiente"}:
            return jsonify({"status": "error", "mensaje": f"Decisión '{decision}' inválida."}), 400

        ruta_manifiesto = Path("data/ensayos/manifiesto_ensayo.csv") if es_ensayo else Path("data/propio/manifiesto.csv")
        manifiesto = ManifiestoDataset(ruta_manifiesto)
        reg = manifiesto.buscar_registro(identificador)
        if reg is None:
            return jsonify({"status": "error", "mensaje": f"Registro '{identificador}' no encontrado en manifiesto."}), 404

        caja_tupla = tuple([int(c) for c in caja]) if (caja is not None and len(caja) == 4) else None
        origen_caja = "manual" if decision == "corregida" else (reg.origen_caja or "recalculada")

        reg.actualizar_revision(decision=decision, caja=caja_tupla, origen_caja=origen_caja)
        manifiesto.guardar()

        return jsonify({
            "status": "ok",
            "identificador": identificador,
            "decision": decision,
            "caja": list(caja_tupla) if caja_tupla else None,
            "origen_caja": origen_caja,
            "mensaje": f"Captura '{identificador}' actualizada como '{decision}'."
        })

    return app
