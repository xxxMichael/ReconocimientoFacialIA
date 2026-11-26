"""
Servidor WSGI Waitress independiente para el portal de reconocimiento remoto.
Escucha únicamente en 127.0.0.1 para ser publicado opcionalmente mediante Cloudflare Tunnel.
"""

from pathlib import Path
from typing import Optional

from .app import crear_portal_reconocimiento_app


def iniciar_servidor_portal_reconocimiento(
    host: str = "127.0.0.1",
    puerto: int = 5002,
    hilos: int = 4,
    modo_servidor: str = "waitress",
    raiz_proyecto: Optional[Path] = None,
    ruta_registro_intentos: Optional[Path] = None,
    indicador_leds: Optional[object] = None,
) -> None:
    """Inicia el portal remoto de reconocimiento facial en el puerto indicado."""
    app = crear_portal_reconocimiento_app(
        raiz_proyecto=raiz_proyecto,
        ruta_registro_intentos=ruta_registro_intentos,
        max_inferencias_simultaneas=hilos,
        indicador_leds=indicador_leds,
    )

    print("\n" + "=" * 68)
    print(" PORTAL INDEPENDIENTE DE RECONOCIMIENTO FACIAL REMOTO")
    print("=" * 68)
    print(f" URL Local:           http://{host}:{puerto}/")
    print(f" Servidor WSGI:       {modo_servidor.upper()}")
    print(f" Hilos / concurrencia:{hilos}")
    print(" Almacenamiento fotos:DESACTIVADO (procesamiento 100% en memoria)")
    print(" Web admin (5000):    Aislada y NO expuesta por este servicio")
    print("=" * 68)
    print("Presione Ctrl+C para detener el portal.\n")

    try:
        if modo_servidor.lower() == "waitress":
            try:
                import waitress
                waitress.serve(app, host=host, port=puerto, threads=hilos)
            except ImportError:
                print("[AVISO] Waitress no disponible. Iniciando con servidor interno de Flask...")
                app.run(host=host, port=puerto, debug=False, threaded=True)
        else:
            app.run(host=host, port=puerto, debug=False, threaded=True)
    finally:
        if indicador_leds is not None:
            indicador_leds.cerrar()  # apaga los LED y libera el puerto


if __name__ == "__main__":
    iniciar_servidor_portal_reconocimiento()
