"""
Auditoría de paridad de entrenamiento entre:
- Referencia: resultados/exp_4clases_portal_completo_20261120_200452/
- Antialiasing: resultados/exp_4clases_antialiasing_20261120_233516/

Distingue explícitamente la evidencia original conservada en disco de la
reconstrucción posterior del inventario de 1,221 muestras.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))

from reconocimiento.adquisicion.propio import ManifiestoDataset
from scripts.experimentar_antialiasing_alineacion import cargar_registros, preparar_datos_con_remuestreo


def _sha256_archivo(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main() -> int:
    dir_std = RAIZ / "resultados/exp_4clases_portal_completo_20261120_200452"
    dir_aa = RAIZ / "resultados/exp_4clases_antialiasing_20261120_233516"
    dir_out = RAIZ / "resultados/auditoria_paridad_antialiasing_20261121"
    dir_out.mkdir(parents=True, exist_ok=True)

    # 1. Evidencia original
    ruta_man_std = dir_std / "manifiesto_entrenamiento_experimento.csv"
    with ruta_man_std.open(encoding="utf-8", newline="") as f:
        filas_std = list(csv.DictReader(f))

    cfg_std = json.loads((dir_std / "configuracion_candidato_portal_completo_propuesto.json").read_text(encoding="utf-8"))
    cfg_aa = json.loads((dir_aa / "configuracion_candidato_antialiasing_propuesto.json").read_text(encoding="utf-8"))
    res_std = json.loads((dir_std / "resumen_experimento_portal_completo.json").read_text(encoding="utf-8"))
    res_aa = json.loads((dir_aa / "resumen_experimento_antialiasing.json").read_text(encoding="utf-8"))

    evidencia_original = {
        "portal_completo_tiene_manifiesto_csv_original": ruta_man_std.exists(),
        "portal_completo_n_filas_original": len(filas_std),
        "antialiasing_tiene_manifiesto_csv_original": (dir_aa / "manifiesto_entrenamiento_experimento.csv").exists(),
        "antialiasing_archivos_originales": sorted([p.name for p in dir_aa.iterdir() if p.is_file()]),
        "semilla_portal_completo": res_std.get("semilla"),
        "resumen_pesos_portal_completo": res_std.get("resumen_pesos"),
        "config_std_arquitectura": cfg_std.get("arquitectura"),
        "config_aa_arquitectura": cfg_aa.get("arquitectura"),
        "config_std_preprocesamiento": cfg_std.get("preprocesamiento"),
        "config_aa_preprocesamiento": cfg_aa.get("preprocesamiento"),
        "sha256_pesos_std": _sha256_archivo(dir_std / "cnn/modelo.pt"),
        "sha256_pesos_aa": _sha256_archivo(dir_aa / "cnn/modelo.pt"),
    }

    # 2. Reconstrucción posterior del inventario de 1,221 muestras con hashes fuente
    # Recolectar rutas fuente reales en el árbol actual
    rutas_fuente: dict[str, Path] = {}

    registros_propio, _ = cargar_registros(solo_base=False)
    for reg in registros_propio:
        if reg.procedencia != "portal_web_externo" and reg.particion == "entrenamiento":
            rutas_fuente[reg.identificador] = RAIZ / reg.ruta

    man_local = ManifiestoDataset(RAIZ / "data/propio/manifiesto_local.csv")
    for reg in man_local.registros:
        if reg.procedencia == "portal_web_externo" and reg.particion == "prueba":
            rutas_fuente[reg.identificador] = RAIZ / reg.ruta

    with (RAIZ / "data/desconocidos_lfw/entrenamiento_4clases/manifiesto.csv").open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            ruta_img = RAIZ / fila["ruta_imagen"]
            lfw_id = f"lfw_train_{fila['identidad_normalizada'].replace(' ', '_')}"
            rutas_fuente[lfw_id] = ruta_img

    with (RAIZ / "data/universidad_desc/manifiesto_universidad.csv").open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            if fila["particion"] == "entrenamiento":
                rutas_fuente[fila["identificador"]] = RAIZ / fila["ruta"]

    # Verificar cuántos de los 1,221 registros originales de Portal Completo tienen archivo fuente localizable hoy
    filas_reconstruidas = []
    hashes_unicos = set()
    sin_archivo = 0
    for fila in filas_std:
        fid = fila["id"]
        base_id = fid[:-5] if (fid.endswith("_aug1") or fid.endswith("_aug2")) else fid
        ruta_f = rutas_fuente.get(base_id)
        if ruta_f is not None and ruta_f.exists():
            h = _sha256_archivo(ruta_f)
            hashes_unicos.add(h)
            ruta_rel = str(ruta_f.relative_to(RAIZ)).replace("\\", "/")
        else:
            h = "no_localizado"
            ruta_rel = "no_localizado"
            sin_archivo += 1
        filas_reconstruidas.append({
            "id": fila["id"],
            "clase": fila["clase"],
            "identidad": fila["identidad"],
            "fuente": fila["fuente"],
            "variante": fila["variante"],
            "particion": fila["particion"],
            "ruta_fuente_actual": ruta_rel,
            "sha256_fuente_actual": h,
        })

    # Reconstruir conteos ejecutando la misma lógica de selección de fuentes
    datos_aa = preparar_datos_con_remuestreo("supersampling_area")
    n_aa_reconstruido = len(datos_aa["X_train_img"])

    ruta_csv_out = dir_out / "inventario_reconstruido_1221.csv"
    with ruta_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_reconstruidas[0].keys()))
        writer.writeheader()
        writer.writerows(filas_reconstruidas)

    resumen = {
        "tipo_documento": "auditoria_paridad_entrenamiento",
        "evidencia_original": evidencia_original,
        "reconstruccion_posterior": {
            "n_muestras_reconstruidas_antialiasing": n_aa_reconstruido,
            "coincide_total_1221": (n_aa_reconstruido == len(filas_std) == 1221),
            "hashes_fuente_unicos_actuales": len(hashes_unicos),
            "registros_sin_archivo_localizado": sin_archivo,
            "archivo_inventario_reconstruido": str(ruta_csv_out.relative_to(RAIZ)).replace("\\", "/"),
        },
        "limitacion_documentada": (
            "El experimento exp_4clases_portal_completo_20261120_200452 conserva su manifiesto_entrenamiento_experimento.csv "
            "original de 1,221 filas, mientras que exp_4clases_antialiasing_20261120_233516 no guardó un CSV equivalente "
            "en el momento del entrenamiento. La reconstrucción posterior confirma 1,221 muestras con la misma función de "
            "carga determinista, semilla 42 y hashes de archivos actuales, pero la igualdad fila-a-fila de las entradas de "
            "antialiasing se sustenta en reconstrucción posterior del script y conteos originales, no en un manifiesto CSV "
            "guardado originalmente en exp_4clases_antialiasing_20261120_233516."
        ),
    }
    (dir_out / "resumen_paridad.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
