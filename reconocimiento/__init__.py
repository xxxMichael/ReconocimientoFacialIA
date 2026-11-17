"""
Proxy para el paquete reconocimiento ubicado bajo src/reconocimiento.
Extiende __path__ para incluir src/reconocimiento permitiendo ejecuciones directas
desde la raíz del repositorio.
"""
from pathlib import Path

_SRC_PKG = Path(__file__).resolve().parent.parent / "src" / "reconocimiento"
if _SRC_PKG.exists():
    __path__.append(str(_SRC_PKG))
