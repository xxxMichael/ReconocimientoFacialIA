"""
Servidor WSGI de producción para el portal de captura de voluntarios externos.
Ejecuta la aplicación en un puerto independiente mediante Waitress (o servidor de desarrollo Flask),
aislado del panel administrativo.
"""

from pathlib import Path
from typing import Optional

from reconocimiento.portal.app import crear_portal_app


def iniciar_servidor_portal(
    host: str = "127.0.0.1",
    puerto: int = 5001,
    hilos: int = 4,
    modo_servidor: str = "waitress",
    dir_recepcion: Optional[Path] = None,
    ruta_invitaciones: Optional[Path] = None
) -> None:
    """Inicia el portal de captura de voluntarios en el puerto indicado."""
    app = crear_portal_app(dir_recepcion=dir_recepcion, ruta_invitaciones=ruta_invitaciones)

    print("\n" + "=" * 65)
    print(" PORTAL DE CAPTURA INDEPENDIENTE (VOLUNTARIOS EXTERNOS)")
    print("=" * 65)
    print(f" URL Local:        http://{host}:{puerto}/")
    print(f" Servidor WSGI:    {modo_servidor.upper()}")
    print(f" Hilos de atención:{hilos}")
    print(f" Directorio fotos: {Path(dir_recepcion) if dir_recepcion else 'data/recepcion_externa/fotos'}")
    print(f" Panel admin local: Conservado intacto (puerto 5000)")
    print("=" * 65)
    print("Presione Ctrl+C para detener el servicio.\n")

    if modo_servidor.lower() == "waitress":
        try:
            import waitress
            waitress.serve(app, host=host, port=puerto, threads=hilos)
        except ImportError:
            print("[AVISO] Waitress no disponible. Iniciando con servidor WSGI interno de Flask...")
            app.run(host=host, port=puerto, debug=False, threaded=True)
    else:
        app.run(host=host, port=puerto, debug=False, threaded=True)


if __name__ == "__main__":
    iniciar_servidor_portal()
