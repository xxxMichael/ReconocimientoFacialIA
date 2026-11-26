"""
Aplicación Flask independiente para el portal de reconocimiento remoto.
No expone rutas administrativas, revisión de imágenes, datasets, entrenamiento
ni acceso a archivos locales del servidor.
"""

from pathlib import Path
from typing import Optional
from flask import Flask, jsonify, render_template, request

from .servicio import (
    CANDIDATO_PRINCIPAL_ID,
    ServicioPortalReconocimiento,
    ServidorOcupadoError,
)


def crear_portal_reconocimiento_app(
    raiz_proyecto: Optional[Path] = None,
    ruta_registro_intentos: Optional[Path] = None,
    max_inferencias_simultaneas: int = 4,
    indicador_leds: Optional[object] = None,
) -> Flask:
    """Crea la aplicación Flask aislada para reconocimiento remoto."""
    dir_base = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(dir_base / "templates"),
        static_folder=str(dir_base / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB máximo por fotograma

    servicio = ServicioPortalReconocimiento(
        raiz_proyecto=raiz_proyecto,
        ruta_registro_intentos=ruta_registro_intentos,
        max_inferencias_simultaneas=max_inferencias_simultaneas,
        indicador_leds=indicador_leds,
    )
    app.config["SERVICIO_PORTAL_RECONOCIMIENTO"] = servicio

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "portal_reconocimiento.html",
            candidatos=servicio.listar_candidatos(),
            candidato_principal_id=CANDIDATO_PRINCIPAL_ID,
        )

    @app.route("/api/salud", methods=["GET"])
    def api_salud():
        return jsonify({
            "status": "ok",
            "servicio": "portal_reconocimiento_remoto",
            "candidato_principal_id": CANDIDATO_PRINCIPAL_ID,
            "candidatos": servicio.listar_candidatos(),
        })

    @app.route("/api/reconocer", methods=["POST"])
    def api_reconocer():
        archivo = request.files.get("frame")
        if archivo is None:
            return jsonify({"status": "error", "mensaje": "Falta el fotograma 'frame'."}), 400

        candidato_id = (request.form.get("candidato_id") or CANDIDATO_PRINCIPAL_ID).strip()
        try:
            frame_seq = int(request.form.get("frame_seq") or 0)
        except ValueError:
            frame_seq = 0

        frame_bytes = archivo.read()
        try:
            res = servicio.procesar_fotograma_bytes(
                frame_bytes=frame_bytes,
                candidato_id=candidato_id,
                frame_seq=frame_seq,
            )
            return jsonify(res)
        except ServidorOcupadoError as e:
            return jsonify({"status": "ocupado", "codigo": "servidor_ocupado", "mensaje": str(e)}), 503
        except ValueError as e:
            return jsonify({"status": "error", "mensaje": str(e)}), 400
        except Exception:
            return jsonify({"status": "error", "mensaje": "Error interno al procesar el fotograma."}), 500

    @app.route("/api/intento", methods=["POST"])
    def api_intento():
        archivo = request.files.get("frame")
        if archivo is None:
            return jsonify({"status": "error", "mensaje": "Falta el fotograma 'frame'."}), 400

        candidato_id = (request.form.get("candidato_id") or CANDIDATO_PRINCIPAL_ID).strip()
        categoria_real = (request.form.get("categoria_real") or "").strip()
        participante_id = (request.form.get("participante_id") or "anonimo").strip()
        id_intento = (request.form.get("id_intento") or "").strip()

        frame_bytes = archivo.read()
        try:
            res = servicio.registrar_intento(
                frame_bytes=frame_bytes,
                candidato_id=candidato_id,
                categoria_real=categoria_real,
                participante_id=participante_id,
                id_intento=id_intento,
            )
            return jsonify(res)
        except ServidorOcupadoError as e:
            return jsonify({"status": "ocupado", "codigo": "servidor_ocupado", "mensaje": str(e)}), 503
        except ValueError as e:
            return jsonify({"status": "error", "mensaje": str(e)}), 400
        except Exception:
            return jsonify({"status": "error", "mensaje": "Error interno al registrar el intento."}), 500

    return app
