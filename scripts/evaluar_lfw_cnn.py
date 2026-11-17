#!/usr/bin/env python
"""
Script reproducible de evaluación externa para el modelo CNN en producción
utilizando 100 identidades independientes de Labeled Faces in the Wild (LFW).
"""
import argparse
import sys
from pathlib import Path

# Añadir src al path si se ejecuta directamente
raiz_proyecto = Path(__file__).resolve().parent.parent
src_dir = raiz_proyecto / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from reconocimiento.evaluacion.lfw import evaluar_lfw_cnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación externa de la CNN en producción con 100 identidades de LFW."
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=42,
        help="Semilla del generador de números aleatorios para muestreo reproducible (default: 42)",
    )
    parser.add_argument(
        "--n-identidades",
        type=int,
        default=100,
        help="Número de identidades externas a evaluar (default: 100)",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="resultados/evaluacion_externa_lfw_cnn",
        help="Directorio donde se guardarán los resultados, manifiestos y el informe (default: resultados/evaluacion_externa_lfw_cnn)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=================================================================")
    print(" EVALUACIÓN EXTERNA CON LABELED FACES IN THE WILD (LFW)")
    print(f" Semilla: {args.semilla} | Muestras: {args.n_identidades}")
    print(f" Directorio de salida: {args.salida}")
    print("=================================================================")

    try:
        evaluar_lfw_cnn(
            semilla=args.semilla,
            n_identidades=args.n_identidades,
            directorio_salida=Path(args.salida),
        )
        return 0
    except Exception as exc:
        print(f"\n[ERROR CRÍTICO] La evaluación externa no pudo completarse: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
