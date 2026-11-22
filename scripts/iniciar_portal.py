"""
Script ejecutable para iniciar el portal de captura de voluntarios externos.
Ejecuta la aplicación en un puerto independiente (por defecto 5001) usando Waitress.

Uso:
    python scripts/iniciar_portal.py [--puerto 5001] [--host 127.0.0.1]
"""

import argparse
import sys
from pathlib import Path

# Asegurar que 'src' esté en sys.path
RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconocimiento.portal.servidor import iniciar_servidor_portal


def main():
    parser = argparse.ArgumentParser(description="Iniciar el portal público de captura independiente.")
    parser.add_argument("--puerto", "--port", type=int, default=5001, help="Puerto de escucha (default: 5001)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host de escucha (default: 127.0.0.1)")
    parser.add_argument("--hilos", type=int, default=4, help="Número de hilos WSGI (default: 4)")
    parser.add_argument("--servidor", type=str, default="waitress", choices=["waitress", "flask"], help="Tipo de servidor (default: waitress)")

    args = parser.parse_args()
    iniciar_servidor_portal(
        host=args.host,
        puerto=args.puerto,
        hilos=args.hilos,
        modo_servidor=args.servidor
    )


if __name__ == "__main__":
    main()
