"""Prepara dos particiones de identidades LFW desconocidas sin procesar fotos."""

import argparse
import csv
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory


REPOSITORIO = Path(__file__).resolve().parents[1]
FUENTE_PREDETERMINADA = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"
SEMILLA = "lfw_desconocidos_v1"
PARTICIONES = ("calibracion", "reserva_prueba")
CANTIDAD = 100


def normalizar_identidad(valor):
    return " ".join(unicodedata.normalize("NFKC", valor).replace("_", " ").split()).casefold()


def leer_exclusiones(manifiestos):
    identidades = set()
    hashes = set()
    for manifiesto in manifiestos:
        with manifiesto.open(encoding="utf-8-sig", newline="") as entrada:
            filas = csv.DictReader(entrada)
            if "identidad_lfw" not in (filas.fieldnames or ()):
                raise ValueError(f"Falta identidad_lfw en {manifiesto}")
            for fila in filas:
                identidad = normalizar_identidad(fila["identidad_lfw"] or "")
                if not identidad:
                    raise ValueError(f"Identidad vacía en {manifiesto}, línea {filas.line_num}")
                identidades.add(identidad)
                if fila.get("sha256"):
                    hashes.add(fila["sha256"].strip().lower())
    return identidades, hashes


def seleccionar(fuente, identidades_excluidas, hashes_excluidos):
    def orden_identidad(carpeta):
        nombre = normalizar_identidad(carpeta.name)
        return hashlib.sha256(f"{SEMILLA}:{nombre}".encode("utf-8")).hexdigest(), carpeta.name

    identidades_usadas = identidades_excluidas.copy()
    hashes_usados = hashes_excluidos.copy()
    seleccion = []
    carpetas = sorted((ruta for ruta in fuente.iterdir() if ruta.is_dir()), key=orden_identidad)
    for carpeta in carpetas:
        identidad = normalizar_identidad(carpeta.name)
        if identidad in identidades_usadas:
            continue
        for foto in sorted(carpeta.iterdir()):
            if not foto.is_file() or foto.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            digest = hashlib.sha256(foto.read_bytes()).hexdigest()
            if digest in hashes_usados:
                continue
            seleccion.append((foto, identidad, digest))
            identidades_usadas.add(identidad)
            hashes_usados.add(digest)
            break
        if len(seleccion) == CANTIDAD * len(PARTICIONES):
            return seleccion
    raise ValueError(f"Solo se encontraron {len(seleccion)} identidades elegibles en {fuente}")


def preparar(fuente):
    destino = REPOSITORIO / "data" / "desconocidos_lfw"
    if destino.exists():
        raise FileExistsError(f"Ya existe {destino}")
    manifiestos = sorted((REPOSITORIO / "resultados").rglob("manifiesto_lfw*.csv"))
    if not manifiestos:
        raise ValueError("Se requiere al menos un manifiesto LFW histórico en resultados/")
    identidades_excluidas, hashes_excluidos = leer_exclusiones(manifiestos)
    seleccion = seleccionar(fuente, identidades_excluidas, hashes_excluidos)
    procedencia = {
        "dataset": "Labeled Faces in the Wild (LFW)",
        "variante": "lfw_funneled",
        "fuente_local": str(fuente),
        "caracteristicas": (
            "Fotografías prealineadas (funneled), de un dominio diferente al de una webcam. "
            "Son negativos de referencia y no sustituyen negativos reales de webcam."
        ),
        "uso": "Solo calibración y reserva de prueba. Excluir del entrenamiento y de las clases softmax.",
        "seleccion": "Orden SHA-256 de semilla e identidad normalizada; primera foto elegible por nombre.",
        "semilla": SEMILLA,
        "manifiestos_excluidos": [ruta.relative_to(REPOSITORIO).as_posix() for ruta in manifiestos],
        "identidades_historicas_excluidas": len(identidades_excluidas),
        "particiones": dict.fromkeys(PARTICIONES, CANTIDAD),
    }
    with TemporaryDirectory(prefix="desconocidos_lfw_", dir=destino.parent) as temporal:
        carpeta_temporal = Path(temporal) / destino.name
        for indice, particion in enumerate(PARTICIONES):
            carpeta = carpeta_temporal / particion
            carpeta.mkdir(parents=True)
            with (carpeta / "manifiesto.csv").open("w", encoding="utf-8", newline="") as salida:
                campos = ("identidad_lfw", "identidad_normalizada", "ruta_imagen", "ruta_fuente", "sha256")
                escritor = csv.DictWriter(salida, fieldnames=campos)
                escritor.writeheader()
                for foto, identidad, digest in seleccion[indice * CANTIDAD : (indice + 1) * CANTIDAD]:
                    copia = carpeta / foto.name
                    with foto.open("rb") as entrada, copia.open("xb") as archivo:
                        shutil.copyfileobj(entrada, archivo)
                    if hashlib.sha256(copia.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"La copia cambió el contenido esperado de {foto}")
                    escritor.writerow({
                        "identidad_lfw": foto.parent.name.replace("_", " "),
                        "identidad_normalizada": identidad,
                        "ruta_imagen": (destino / particion / foto.name).relative_to(REPOSITORIO).as_posix(),
                        "ruta_fuente": foto.relative_to(fuente).as_posix(),
                        "sha256": digest,
                    })
        (carpeta_temporal / "procedencia.json").write_text(
            json.dumps(procedencia, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        carpeta_temporal.rename(destino)
    print(f"{destino}: {CANTIDAD} identidades por partición, sin solapamiento de identidades ni SHA-256.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuente", type=Path, default=FUENTE_PREDETERMINADA, help="Carpeta local lfw_funneled")
    argumentos = parser.parse_args()
    try:
        preparar(argumentos.fuente.expanduser().resolve())
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
