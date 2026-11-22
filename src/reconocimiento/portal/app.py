"""
Aplicación Flask del portal público de captura independiente.
Diseñada para publicarse de forma segura mediante túnel HTTPS (Cloudflare Tunnel).
No contiene rutas administrativas, no carga modelos de inferencia, no lista fotos
almacenadas ni expone rutas del sistema de archivos local.
"""

import base64
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

from reconocimiento.portal.almacenamiento import GestorAlmacenamientoExterno
from reconocimiento.portal.gestor_invitaciones import GestorInvitaciones


def crear_portal_app(
    dir_recepcion: Optional[Path] = None,
    ruta_invitaciones: Optional[Path] = None,
    raiz_proyecto: Optional[Path] = None
) -> Flask:
    """Fábrica de la aplicación web independiente para el portal de captura de voluntarios."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="/static"
    )

    # Configuración de límites y seguridad
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    gestor_inv = GestorInvitaciones(ruta_archivo=ruta_invitaciones, raiz_proyecto=raiz_proyecto)
    gestor_alm = GestorAlmacenamientoExterno(dir_base=dir_recepcion, raiz_proyecto=raiz_proyecto)

    @app.after_request
    def agregar_cabeceras_seguridad(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Desactivar caché para datos de estado
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.route("/", methods=["GET"])
    def index():
        token = request.args.get("token", "").strip()
        if token:
            inv = gestor_inv.obtener_invitacion(token)
            if inv and inv.get("activo"):
                return render_template("portal.html", token=token, identificador=inv.get("identificador"))
            else:
                return render_template("error_invitacion.html", mensaje="El enlace de invitación no es válido o ha sido desactivado.")
        return render_template("bienvenida.html")

    @app.route("/participar/<token>", methods=["GET"])
    def participar(token: str):
        tok_limpio = str(token).strip()
        inv = gestor_inv.obtener_invitacion(tok_limpio)
        if not inv or not inv.get("activo"):
            return render_template("error_invitacion.html", mensaje="El enlace de invitación no es válido o ha expirado."), 403

        return render_template(
            "portal.html",
            token=tok_limpio,
            identificador=inv.get("identificador")
        )

    @app.route("/salud", methods=["GET"])
    def salud():
        return jsonify({
            "estado": "ok",
            "servicio": "portal_captura_externa",
            "modo": "aislado_independiente"
        })

    @app.route("/api/estado", methods=["GET"])
    def api_estado():
        token = request.args.get("token", "").strip()
        if not token:
            return jsonify({"exito": False, "error": "Token de invitación no proporcionado."}), 400

        inv = gestor_inv.obtener_invitacion(token)
        if not inv or not inv.get("activo"):
            return jsonify({"exito": False, "error": "Invitación inválida o desactivada."}), 403

        id_sujeto = inv["identificador"]
        progreso = gestor_alm.obtener_progreso(id_sujeto)

        return jsonify({
            "exito": True,
            "identificador": id_sujeto,
            "grupo": inv.get("grupo", "desconocidos_prueba"),
            "consentimiento_registrado": inv.get("consentimiento_registrado", False),
            "consentimiento_timestamp": inv.get("consentimiento_timestamp"),
            "progreso": progreso
        })

    @app.route("/api/consentimiento", methods=["POST"])
    def api_consentimiento():
        data = request.get_json(silent=True) or {}
        token = str(
            data.get("token") or
            request.form.get("token") or
            request.args.get("token") or
            request.headers.get("X-Invitation-Token") or
            ""
        ).strip()

        if not token:
            return jsonify({"exito": False, "error": "Token de invitación requerido."}), 400

        inv = gestor_inv.obtener_invitacion(token)
        if not inv or not inv.get("activo"):
            return jsonify({"exito": False, "error": "Invitación inválida o desactivada."}), 403

        exito = gestor_inv.registrar_consentimiento(token)
        if exito:
            return jsonify({"exito": True, "mensaje": "Consentimiento informado registrado exitosamente."})
        return jsonify({"exito": False, "error": "No se pudo registrar el consentimiento."}), 500

    @app.route("/api/subir", methods=["POST"])
    def api_subir():
        # Puede recibirse como multipart/form-data o como JSON con base64
        token = ""
        pose = ""
        distancia = ""
        datos_imagen: Optional[bytes] = None

        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            token = str(data.get("token", "")).strip()
            pose = str(data.get("pose", "")).strip()
            distancia = str(data.get("distancia", "")).strip()
            b64_str = str(data.get("imagen_b64", "")).strip()
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                try:
                    datos_imagen = base64.b64decode(b64_str)
                except Exception:
                    return jsonify({"exito": False, "error": "Formato base64 de imagen corrupto o inválido."}), 400
        else:
            token = request.form.get("token", "").strip()
            pose = request.form.get("pose", "").strip()
            distancia = request.form.get("distancia", "").strip()
            archivo = request.files.get("imagen")
            if archivo:
                datos_imagen = archivo.read()

        if not token:
            return jsonify({"exito": False, "error": "Token de invitación obligatorio."}), 400

        inv = gestor_inv.obtener_invitacion(token)
        if not inv or not inv.get("activo"):
            return jsonify({"exito": False, "error": "Invitación inválida o revocada."}), 403

        if not inv.get("consentimiento_registrado"):
            return jsonify({
                "exito": False,
                "error": "Debe registrar explícitamente su consentimiento antes de enviar imágenes."
            }), 403

        if not datos_imagen:
            return jsonify({"exito": False, "error": "No se recibió ninguna fotografía."}), 400

        id_sujeto = inv["identificador"]
        exito, mensaje, progreso = gestor_alm.guardar_captura(
            identificador=id_sujeto,
            pose=pose,
            distancia=distancia,
            datos_imagen=datos_imagen,
            consentimiento_confirmado=True
        )

        if not exito:
            return jsonify({"exito": False, "error": mensaje}), 400

        # Respuesta segura: solo datos de avance, sin rutas ni enlaces de descarga
        return jsonify({
            "exito": True,
            "mensaje": mensaje,
            "identificador": id_sujeto,
            "pose": pose,
            "distancia": distancia,
            "progreso": progreso
        })

    return app
