"""
Script de inicio directo para el portal independiente de reconocimiento facial remoto.
No expone el panel administrativo (5000) ni el portal de recopilación (5001).
"""

import argparse
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.portal_reconocimiento.servidor import iniciar_servidor_portal_reconocimiento


def main() -> int:
    parser = argparse.ArgumentParser(description="Iniciar portal independiente de reconocimiento remoto")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host local (default: 127.0.0.1)")
    parser.add_argument("--puerto", "--port", dest="puerto", type=int, default=5002, help="Puerto local (default: 5002)")
    parser.add_argument("--hilos", type=int, default=4, help="Hilos Waitress (default: 4)")
    parser.add_argument("--servidor", type=str, default="waitress", choices=["waitress", "flask"], help="Servidor WSGI")
    args = parser.parse_args()

    from reconocimiento.hardware.indicador_arduino import crear_indicador_auto
    iniciar_servidor_portal_reconocimiento(
        host=args.host,
        puerto=args.puerto,
        hilos=args.hilos,
        modo_servidor=args.servidor,
        raiz_proyecto=RAIZ,
        indicador_leds=crear_indicador_auto(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
