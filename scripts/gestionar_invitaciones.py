"""
Herramienta de gestión de invitaciones individuales para el portal de captura.
Permite emitir enlaces únicos vinculados a un identificador anónimo (ej. externo_001),
listar invitaciones activas y desactivar enlaces cuando sea necesario.

Uso:
    python scripts/gestionar_invitaciones.py crear --id externo_001 [--base-url https://xyz.trycloudflare.com]
    python scripts/gestionar_invitaciones.py listar [--base-url ...]
    python scripts/gestionar_invitaciones.py desactivar --token-o-id externo_001
    python scripts/gestionar_invitaciones.py activar --token-o-id externo_001
"""

import argparse
import sys
from pathlib import Path

# Asegurar que 'src' esté en sys.path
RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconocimiento.portal.gestor_invitaciones import GestorInvitaciones


def main():
    parser = argparse.ArgumentParser(description="Gestionar invitaciones del portal público de captura.")
    subparsers = parser.add_subparsers(dest="accion", required=True)

    # Subcomando crear
    parser_crear = subparsers.add_parser("crear", help="Crear una nueva invitación para un identificador")
    parser_crear.add_argument("--id", "--identificador", required=True, dest="identificador", help="Identificador anónimo (ej. externo_001)")
    parser_crear.add_argument("--notas", default="", help="Notas o descripción interna del voluntario")
    parser_crear.add_argument("--base-url", default=None, help="URL pública base (ej. https://xyz.trycloudflare.com o http://127.0.0.1:5001)")

    # Subcomando listar
    parser_listar = subparsers.add_parser("listar", help="Listar todas las invitaciones")
    parser_listar.add_argument("--base-url", default=None, help="URL pública base para formatear los enlaces")

    # Subcomando desactivar
    parser_desact = subparsers.add_parser("desactivar", help="Desactivar una invitación por token o identificador")
    parser_desact.add_argument("--token-o-id", required=True, help="Token o identificador del participante a desactivar")

    # Subcomando activar
    parser_act = subparsers.add_parser("activar", help="Reactivar una invitación previamente desactivada")
    parser_act.add_argument("--token-o-id", required=True, help="Token o identificador a reactivar")

    args = parser.parse_args()
    gestor = GestorInvitaciones()

    if args.accion == "crear":
        try:
            inv = gestor.crear_invitacion(
                identificador=args.identificador,
                notas=args.notas,
                base_url=args.base_url
            )
            print("\n✓ Invitación creada exitosamente:")
            print(f"  Identificador: {inv['identificador']}")
            print(f"  Token:         {inv['token']}")
            print(f"  Grupo:         {inv['grupo']} ({inv['particion']})")
            print(f"  Estado:        {'Activa' if inv['activo'] else 'Inactiva'}")
            print(f"  Enlace:        {inv['url']}\n")
        except Exception as e:
            print(f"\n[ERROR] No se pudo crear la invitación: {e}\n")
            sys.exit(1)

    elif args.accion == "listar":
        invs = gestor.listar_invitaciones(base_url=args.base_url)
        print("\n" + "=" * 75)
        print(" REGISTRO DE INVITACIONES DEL PORTAL DE CAPTURA")
        print("=" * 75)
        if not invs:
            print("No hay invitaciones registradas.")
        else:
            for i, inv in enumerate(invs, 1):
                estado_str = "ACTIVA" if inv.get("activo") else "DESACTIVADA"
                cons_str = "SÍ" if inv.get("consentimiento_registrado") else "NO"
                print(f"{i}. [{estado_str}] ID: {inv.get('identificador')} | Consentimiento: {cons_str}")
                print(f"   Token:  {inv.get('token')}")
                print(f"   Enlace: {inv.get('url')}")
                if inv.get("notas"):
                    print(f"   Notas:  {inv.get('notas')}")
                print("-" * 75)
        print()

    elif args.accion == "desactivar":
        exito = gestor.desactivar_invitacion(args.token_o_id)
        if exito:
            print(f"\n✓ Invitación '{args.token_o_id}' desactivada con éxito.\n")
        else:
            print(f"\n[ERROR] No se encontró ninguna invitación coincidente con '{args.token_o_id}'.\n")
            sys.exit(1)

    elif args.accion == "activar":
        exito = gestor.activar_invitacion(args.token_o_id)
        if exito:
            print(f"\n✓ Invitación '{args.token_o_id}' reactivada con éxito.\n")
        else:
            print(f"\n[ERROR] No se encontró ninguna invitación coincidente con '{args.token_o_id}'.\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
