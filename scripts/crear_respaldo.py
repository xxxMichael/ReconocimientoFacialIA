"""
Crea un respaldo fechado completo del sistema actual antes de implementar
soporte de orientación y alineación facial.

Incluye:
- Cinco modelos de 4 clases (resultados/desarrollo_4clases_corregido_20261119_123602/)
- Candidato congelado (configuracion_candidato_congelado*.json)
- Detector YuNet (modelos/)
- Artefactos de evaluación final independiente y modelo seleccionado
- Código completo (src/, scripts/, tests/, pyproject.toml, etc., sin __pycache__)
- Manifiestos CSV y anotaciones de revisión (sin copiar fotografías)

Genera inventario_sha256.json, verifica integridad copia vs origen y crea RESTAURAR.md.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple


RAIZ = Path(__file__).resolve().parent.parent
IGNORAR_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "env", "respaldos"}
EXTENSIONES_EXCLUIDAS_DATA = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def copiar_con_verificacion(origen: Path, destino: Path, inventario: List[Dict]) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)
    hash_orig = sha256_archivo(origen)
    hash_dest = sha256_archivo(destino)
    if hash_orig != hash_dest:
        raise RuntimeError(f"Fallo de verificación SHA-256 al copiar {origen} -> {destino}")
    inventario.append({
        "ruta_origen": str(origen.relative_to(RAIZ)).replace("\\", "/"),
        "ruta_respaldo": str(destino.relative_to(RAIZ)).replace("\\", "/"),
        "tamano_bytes": origen.stat().st_size,
        "sha256": hash_dest,
        "verificado": True,
    })


def copiar_arbol(
    dir_origen: Path,
    dir_destino: Path,
    inventario: List[Dict],
    excluir_imagenes: bool = False,
) -> None:
    if not dir_origen.exists():
        return
    for item in sorted(dir_origen.rglob("*")):
        if item.is_dir():
            continue
        if any(parte in IGNORAR_DIRS for parte in item.parts):
            continue
        if item.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if excluir_imagenes and item.suffix.lower() in EXTENSIONES_EXCLUIDAS_DATA:
            continue
        rel = item.relative_to(dir_origen)
        copiar_con_verificacion(item, dir_destino / rel, inventario)


def main() -> int:
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_respaldo = RAIZ / "respaldos" / f"respaldo_pre_alineacion_{marca}"
    dir_respaldo.mkdir(parents=True, exist_ok=True)
    inventario: List[Dict] = []

    print(f"Creando respaldo en: {dir_respaldo}")

    # 1. Archivos raíz relevantes (configs, pyproject, README, etc.)
    for patron in ("configuracion_candidato*.json", "pyproject.toml", "requirements*.txt", "README.md"):
        for archivo in sorted(RAIZ.glob(patron)):
            if archivo.is_file():
                copiar_con_verificacion(archivo, dir_respaldo / archivo.name, inventario)

    # 2. Código fuente, scripts y pruebas (incluye archivos no rastreados)
    for carpeta in ("src", "scripts", "tests"):
        copiar_arbol(RAIZ / carpeta, dir_respaldo / carpeta, inventario)

    # 3. Modelo YuNet y detector
    copiar_arbol(RAIZ / "modelos", dir_respaldo / "modelos", inventario)

    # 4. Cinco modelos actuales de cuatro clases y evaluación final
    carpetas_resultados = [
        "desarrollo_4clases_corregido_20261119_123602",
        "evaluacion_final_independiente_corregida_20261119_204348",
    ]
    for sub in carpetas_resultados:
        copiar_arbol(RAIZ / "resultados" / sub, dir_respaldo / "resultados" / sub, inventario)

    sel_json = RAIZ / "resultados" / "modelo_seleccionado.json"
    if sel_json.exists():
        copiar_con_verificacion(sel_json, dir_respaldo / "resultados" / "modelo_seleccionado.json", inventario)

    # 5. Manifiestos y anotaciones de revisión (sin fotografías)
    copiar_arbol(RAIZ / "data", dir_respaldo / "data", inventario, excluir_imagenes=True)

    # Guardar inventario
    ruta_inv = dir_respaldo / "inventario_sha256.json"
    datos_inv = {
        "fecha_creacion": datetime.now().isoformat(),
        "total_archivos": len(inventario),
        "todos_verificados": all(x["verificado"] for x in inventario),
        "archivos": inventario,
    }
    with open(ruta_inv, "w", encoding="utf-8") as f:
        json.dump(datos_inv, f, indent=2, ensure_ascii=False)

    # Crear RESTAURAR.md
    restaurar_md = f"""# Procedimiento de Restauración — Respaldo `{dir_respaldo.name}`

Fecha de creación: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total de archivos respaldados y verificados (SHA-256): **{len(inventario)}**

## Contenido del respaldo
- **Candidato congelado original**: `configuracion_candidato_congelado_corregida.json` (CNN 4 clases, umbral 0.46).
- **Cinco modelos de 4 clases**: `resultados/desarrollo_4clases_corregido_20261119_123602/` (`arbol`, `knn`, `svm`, `ann`, `cnn`, mapeo de clases, autorizados e informes).
- **Evaluación final independiente**: `resultados/evaluacion_final_independiente_corregida_20261119_204348/`.
- **Detector YuNet**: `modelos/detector/face_detection_yunet_2023mar.onnx`.
- **Código fuente, scripts y pruebas**: `src/`, `scripts/`, `tests/`, `pyproject.toml`.
- **Manifiestos y anotaciones**: `data/` (únicamente CSV, JSON y anotaciones; las fotografías originales no fueron modificadas ni duplicadas).
- **Inventario criptográfico**: `inventario_sha256.json`.

## Cómo verificar la integridad del respaldo
Ejecute desde la raíz del proyecto:
```powershell
python -c "import json, hashlib; from pathlib import Path; inv = json.load(open('{dir_respaldo.relative_to(RAIZ).as_posix()}/inventario_sha256.json')); ok = all(hashlib.sha256(open(x['ruta_respaldo'],'rb').read()).hexdigest() == x['sha256'] for x in inv['archivos']); print('Integridad OK:', ok, '| Archivos:', len(inv['archivos']))"
```

## Cómo restaurar la versión original completa
Desde la raíz del repositorio (`ReconocimientoFacialIA`):

1. **Restaurar código, configuraciones y modelos**:
```powershell
Copy-Item -Path "{dir_respaldo.relative_to(RAIZ).as_posix()}\\src\\*" -Destination "src\\" -Recurse -Force
Copy-Item -Path "{dir_respaldo.relative_to(RAIZ).as_posix()}\\scripts\\*" -Destination "scripts\\" -Recurse -Force
Copy-Item -Path "{dir_respaldo.relative_to(RAIZ).as_posix()}\\tests\\*" -Destination "tests\\" -Recurse -Force
Copy-Item -Path "{dir_respaldo.relative_to(RAIZ).as_posix()}\\configuracion_candidato*.json" -Destination "." -Force
Copy-Item -Path "{dir_respaldo.relative_to(RAIZ).as_posix()}\\resultados\\*" -Destination "resultados\\" -Recurse -Force
```

2. **Restaurar manifiestos (si se modificaron)**:
```powershell
Copy-Item -Path "{dir_respaldo.relative_to(RAIZ).as_posix()}\\data\\*" -Destination "data\\" -Recurse -Force
```

3. **Ejecutar el sistema original directamente sin restaurar código** (la configuración original permanece intacta):
```powershell
python -m reconocimiento web --config-candidato configuracion_candidato_congelado_corregida.json
```

4. **Validar tras restaurar**:
```powershell
python -m unittest discover tests
```
"""
    with open(dir_respaldo / "RESTAURAR.md", "w", encoding="utf-8") as f:
        f.write(restaurar_md)

    print(f"Respaldo completado: {len(inventario)} archivos copiados y verificados.")
    print(f"Inventario: {ruta_inv}")
    print(f"Guía de restauración: {dir_respaldo / 'RESTAURAR.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
