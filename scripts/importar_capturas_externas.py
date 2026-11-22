"""
Script para importar las capturas recibidas en el portal externo al manifiesto del proyecto.

Uso:
    python scripts/importar_capturas_externas.py [--manifiesto data/propio/manifiesto_local.csv] [--id externo_001]
"""

import argparse
import sys
from pathlib import Path

# Asegurar que 'src' esté en sys.path
RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconocimiento.portal.importador import importar_capturas_externas


def main():
    parser = argparse.ArgumentParser(description="Importar capturas externas al manifiesto del proyecto.")
    parser.add_argument("--recepcion", type=str, default=None, help="Directorio de recepción (default: data/recepcion_externa)")
    parser.add_argument("--manifiesto", type=str, default=None, help="Ruta al archivo de manifiesto destino (default: manifiesto_local.csv o manifiesto.csv)")
    parser.add_argument("--id", "--identificador", dest="identificador", default=None, help="Importar EXCLUSIVAMENTE este identificador de voluntario (ej. externo_001)")
    parser.add_argument("--excluir", type=str, default="", help="Lista de identificadores a excluir separados por coma (ej. externo_piloto,test_01)")
    parser.add_argument("--incluir-pilotos", action="store_true", default=False, help="Permitir importar identificadores con nombre 'piloto' o 'test' (por defecto se excluyen)")
    parser.add_argument("--permitir-excepcion", type=str, default="", help="Lista de identificadores piloto a incluir explícitamente como excepción (ej. externo_piloto)")

    args = parser.parse_args()

    dir_rec = Path(args.recepcion) if args.recepcion else None
    ruta_man = Path(args.manifiesto) if args.manifiesto else None
    excluir_lista = [x.strip() for x in args.excluir.split(",") if x.strip()]
    excepciones_lista = [x.strip() for x in args.permitir_excepcion.split(",") if x.strip()]

    print("\n" + "=" * 65)
    print(" IMPORTACIÓN DE CAPTURAS EXTERNAS AL MANIFIESTO")
    print("=" * 65)

    stats = importar_capturas_externas(
        dir_recepcion=dir_rec,
        ruta_manifiesto_destino=ruta_man,
        identificador_filtro=args.identificador,
        excluir_identificadores=excluir_lista,
        excluir_pilotos=not args.incluir_pilotos,
        permitir_excepciones_piloto=excepciones_lista,
    )

    print(f" Registros leídos en recepción: {stats['total_encontradas']}")
    print(f" Registros nuevos importados:   {stats['importadas_nuevas']}")
    print(f" Registros ya existentes:       {stats['omitidas_ya_existentes']}")
    print(f" Registros pilotos excluidos:   {stats['omitidas_piloto']}")
    print(f" Errores de validación:         {len(stats['errores'])}")
    print(f" Manifiesto actualizado:        {stats['ruta_manifiesto']}")

    if stats["errores"]:
        print("\nDetalle de incidencias:")
        for err in stats["errores"]:
            print(f"  - [{err['identificador']}]: {err['error']}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
