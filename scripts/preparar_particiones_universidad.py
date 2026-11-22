"""
Prepara particiones reproducibles para el nuevo conjunto de desconocidos universitarios
ubicado en data/recepcion_externa/fotos_desc.

- Verifica legibilidad técnica y hashes SHA-256 sin modificar las fotos originales.
- Registra cualquier discrepancia concreta de inventario (ej. 34 archivos encontrados vs 33 declarados).
- Asigna identificadores anónimos (universidad_001 ... universidad_034).
- Permuta con semilla fija (SEMILLA = 20261120) antes de cualquier inferencia:
    * 18 entrenamiento
    * 7  validacion
    * 9  prueba_reservada (8 previstas + 1 excedente, todas fuera del desarrollo)
- Genera el manifiesto específico data/universidad_desc/manifiesto_universidad.csv.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DIR_FOTOS_DESC = RAIZ / "data" / "recepcion_externa" / "fotos_desc"
DIR_SALIDA = RAIZ / "data" / "universidad_desc"
SEMILLA_PARTICION = 20261120


def preparar_particiones() -> dict:
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    if not DIR_FOTOS_DESC.exists():
        raise FileNotFoundError(f"No existe el directorio de fotos universitarias: {DIR_FOTOS_DESC}")

    archivos = sorted([p for p in DIR_FOTOS_DESC.iterdir() if p.is_file()], key=lambda p: p.name)

    registros_crudos = []
    hashes_vistos = {}
    ilegibles = []
    duplicados = []

    for idx, p in enumerate(archivos, 1):
        contenido = p.read_bytes()
        sha256 = hashlib.sha256(contenido).hexdigest()
        if sha256 in hashes_vistos:
            duplicados.append((p.name, hashes_vistos[sha256]))
        else:
            hashes_vistos[sha256] = p.name

        img = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            ilegibles.append(p.name)
            alto, ancho = 0, 0
        else:
            alto, ancho = img.shape[:2]

        identificador = f"universidad_{idx:03d}"
        ruta_rel = str(p.relative_to(RAIZ)).replace("\\", "/")
        registros_crudos.append({
            "identificador": identificador,
            "ruta": ruta_rel,
            "hash_sha256": sha256,
            "procedencia": "archivo_universitario_con_permiso",
            "permiso_declarado": "autorizado_uso_academico_declarado_usuario_20261120",
            "tipo_sujeto": "desconocido",
            "ancho": ancho,
            "alto": alto,
        })

    if ilegibles:
        raise RuntimeError(f"Se encontraron archivos ilegibles en fotos_desc: {ilegibles}")
    if duplicados:
        raise RuntimeError(f"Se encontraron duplicados SHA-256 en fotos_desc: {duplicados}")

    n_total = len(registros_crudos)
    if n_total < 33:
        raise RuntimeError(f"Se esperaban al menos 33 archivos, encontrados: {n_total}")

    # Permutación reproducible con semilla fija
    rng = np.random.RandomState(SEMILLA_PARTICION)
    indices = rng.permutation(n_total)

    n_train = 18
    n_val = 7
    # El resto (9 si n_total == 34, 8 si n_total == 33) va a prueba_reservada
    idx_train = set(indices[:n_train].tolist())
    idx_val = set(indices[n_train:n_train + n_val].tolist())
    idx_test = set(indices[n_train + n_val:].tolist())

    for i, reg in enumerate(registros_crudos):
        if i in idx_train:
            reg["particion"] = "entrenamiento"
        elif i in idx_val:
            reg["particion"] = "validacion"
        else:
            reg["particion"] = "prueba_reservada"

    ruta_manifiesto = DIR_SALIDA / "manifiesto_universidad.csv"
    campos = [
        "identificador", "ruta", "hash_sha256", "procedencia",
        "permiso_declarado", "particion", "tipo_sujeto", "ancho", "alto"
    ]
    with open(ruta_manifiesto, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros_crudos)

    auditoria = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "directorio_origen": str(DIR_FOTOS_DESC),
        "semilla_permutacion": SEMILLA_PARTICION,
        "total_declarado_usuario": 33,
        "total_encontrado_real": n_total,
        "discrepancia_inventario": (
            f"Se encontraron {n_total} archivos PNG únicos y legibles (1 más de los 33 declarados, "
            f"debido a la coexistencia de image.png, image copy.png e image copy 2..33.png). "
            f"Se asignaron exactamente 18 a entrenamiento, 7 a validacion y {len(idx_test)} a prueba_reservada."
        ) if n_total != 33 else "Sin discrepancia (33 archivos).",
        "ilegibles": ilegibles,
        "duplicados": duplicados,
        "conteos_particion": {
            "entrenamiento": len(idx_train),
            "validacion": len(idx_val),
            "prueba_reservada": len(idx_test),
        },
        "identificadores_por_particion": {
            "entrenamiento": [registros_crudos[i]["identificador"] for i in sorted(idx_train)],
            "validacion": [registros_crudos[i]["identificador"] for i in sorted(idx_val)],
            "prueba_reservada": [registros_crudos[i]["identificador"] for i in sorted(idx_test)],
        },
        "ruta_manifiesto": str(ruta_manifiesto),
        "sha256_manifiesto": hashlib.sha256(ruta_manifiesto.read_bytes()).hexdigest(),
    }

    ruta_auditoria = DIR_SALIDA / "auditoria_inventario.json"
    ruta_auditoria.write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70)
    print(" AUDITORÍA Y PARTICIONADO: DESCONOCIDOS UNIVERSITARIOS")
    print("=" * 70)
    print(f" Total archivos reales:  {n_total} (legibles: {n_total}, duplicados: 0)")
    print(f" Discrepancia:           {auditoria['discrepancia_inventario']}")
    print(f" Semilla permutación:    {SEMILLA_PARTICION}")
    print(f" Particiones:            entrenamiento={len(idx_train)}, validacion={len(idx_val)}, prueba_reservada={len(idx_test)}")
    print(f" Manifiesto generado:    {ruta_manifiesto}")
    print("=" * 70)
    return auditoria


if __name__ == "__main__":
    preparar_particiones()
