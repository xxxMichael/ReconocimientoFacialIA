"""Selecciona y procesa 300 identidades LFW para entrenamiento de 4 clases."""

import argparse
import csv
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

REPOSITORIO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORIO / "src"))

import cv2
import numpy as np

from reconocimiento.preprocesamiento.detector import EstadoDeteccion
from reconocimiento.preprocesamiento.fabrica import crear_detector

REPOSITORIO = Path(__file__).resolve().parents[1]
FUENTE_PREDETERMINADA = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"
SEMILLA = "lfw_desconocidos_entrenamiento_4clases_v1"
CANTIDAD = 300


def normalizar_identidad(valor):
    return " ".join(unicodedata.normalize("NFKC", valor).replace("_", " ").split()).casefold()


def leer_todas_las_exclusiones():
    identidades = set()
    hashes = set()
    # Buscar todos los manifiestos lfw previos en resultados y data/desconocidos_lfw
    rutas_manifiestos = set()
    for ruta in (REPOSITORIO / "resultados").rglob("manifiesto_lfw*.csv"):
        rutas_manifiestos.add(ruta)
    for ruta in (REPOSITORIO / "data" / "desconocidos_lfw").rglob("*.csv"):
        rutas_manifiestos.add(ruta)

    manifiestos_leidos = []
    for manifiesto in sorted(rutas_manifiestos):
        manifiestos_leidos.append(manifiesto.relative_to(REPOSITORIO).as_posix())
        with manifiesto.open(encoding="utf-8-sig", newline="") as entrada:
            filas = csv.DictReader(entrada)
            for fila in filas:
                for k in ("identidad_lfw", "identidad_normalizada", "persona"):
                    val = fila.get(k)
                    if val:
                        identidades.add(normalizar_identidad(val))
                if fila.get("sha256"):
                    hashes.add(fila["sha256"].strip().lower())

    return identidades, hashes, manifiestos_leidos


def verificar_inventario(fuente: Path, identidades_excluidas: set):
    carpetas = [d for d in fuente.iterdir() if d.is_dir()]
    fotos = list(fuente.glob("*/*.jpg"))
    total_identidades = len(carpetas)
    total_fotos = len(fotos)

    carpetas_disponibles = 0
    fotos_disponibles = 0
    for d in carpetas:
        ident = normalizar_identidad(d.name)
        if ident not in identidades_excluidas:
            carpetas_disponibles += 1
            fotos_disponibles += len(list(d.glob("*.jpg")))

    return {
        "total_identidades": total_identidades,
        "total_fotos": total_fotos,
        "identidades_excluidas": len(identidades_excluidas),
        "identidades_disponibles": carpetas_disponibles,
        "fotos_disponibles": fotos_disponibles,
    }


def seleccionar_y_detectar(fuente: Path, identidades_excluidas: set, hashes_excluidos: set, detector):
    def orden_identidad(carpeta):
        nombre = normalizar_identidad(carpeta.name)
        return hashlib.sha256(f"{SEMILLA}:{nombre}".encode("utf-8")).hexdigest(), carpeta.name

    carpetas = sorted((ruta for ruta in fuente.iterdir() if ruta.is_dir()), key=orden_identidad)

    seleccionados = []
    rechazados = []
    identidades_usadas = identidades_excluidas.copy()
    hashes_usados = hashes_excluidos.copy()

    for carpeta in carpetas:
        identidad = normalizar_identidad(carpeta.name)
        if identidad in identidades_usadas:
            continue

        # Intentar con las fotos de la carpeta
        foto_aceptada = False
        for foto in sorted(carpeta.iterdir()):
            if not foto.is_file() or foto.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            digest = hashlib.sha256(foto.read_bytes()).hexdigest()
            if digest in hashes_usados:
                rechazados.append({
                    "identidad": identidad,
                    "foto": foto.name,
                    "causa": "hash_previamente_usado",
                    "sha256": digest,
                })
                continue

            contenido = foto.read_bytes()
            img_bgr = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                rechazados.append({
                    "identidad": identidad,
                    "foto": foto.name,
                    "causa": "error_decodificacion",
                    "sha256": digest,
                })
                continue

            deteccion = detector.detectar(img_bgr)
            if deteccion.estado != EstadoDeteccion.UN_ROSTRO or deteccion.recuadro is None:
                rechazados.append({
                    "identidad": identidad,
                    "foto": foto.name,
                    "causa": f"deteccion_{deteccion.estado.value}",
                    "sha256": digest,
                })
                continue

            # Evaluar si la caja es dudosa (aspect ratio extremo, tamaño desproporcionado)
            x, y, w, h = deteccion.recuadro
            h_img, w_img = img_bgr.shape[:2]
            dudosa = False
            motivo_dudosa = []
            if w < 30 or h < 30:
                dudosa = True
                motivo_dudosa.append("tamano_pequeno")
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                dudosa = True
                motivo_dudosa.append("aspect_ratio_atipico")
            if x < 0 or y < 0 or (x + w) > w_img or (y + h) > h_img:
                dudosa = True
                motivo_dudosa.append("fuera_de_limites")

            seleccionados.append({
                "foto_path": foto,
                "identidad_lfw": carpeta.name.replace("_", " "),
                "identidad_normalizada": identidad,
                "sha256": digest,
                "recuadro": deteccion.recuadro,
                "dudosa": dudosa,
                "motivo_dudosa": ";".join(motivo_dudosa) if dudosa else "ninguno",
            })
            identidades_usadas.add(identidad)
            hashes_usados.add(digest)
            foto_aceptada = True
            break  # Una foto por identidad

        if len(seleccionados) >= CANTIDAD:
            break

    if len(seleccionados) < CANTIDAD:
        raise ValueError(f"Solo se pudieron reunir {len(seleccionados)} de {CANTIDAD} requeridos.")

    return seleccionados, rechazados


def generar_lamina_inspeccion(seleccionados, ruta_salida: Path):
    """Genera una lámina visual de contacto con los recortes de las 300 fotos."""
    tam_celda = 64
    columnas = 20
    filas = (len(seleccionados) + columnas - 1) // columnas
    ancho_total = columnas * tam_celda
    alto_total = filas * tam_celda

    mosaico = np.zeros((alto_total, ancho_total, 3), dtype=np.uint8)

    for idx, item in enumerate(seleccionados):
        col = idx % columnas
        fil = idx // columnas
        x0 = col * tam_celda
        y0 = fil * tam_celda

        img = cv2.imread(str(item["foto_path"]))
        x, y, w, h = item["recuadro"]
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        recorte = img[y1:y2, x1:x2]
        if recorte.size > 0:
            recorte_res = cv2.resize(recorte, (tam_celda, tam_celda))
        else:
            recorte_res = np.zeros((tam_celda, tam_celda, 3), dtype=np.uint8)

        if item["dudosa"]:
            # Recuadro rojo para sospechosas
            cv2.rectangle(recorte_res, (0, 0), (tam_celda - 1, tam_celda - 1), (0, 0, 255), 2)
        else:
            # Borde sutil gris
            cv2.rectangle(recorte_res, (0, 0), (tam_celda - 1, tam_celda - 1), (50, 50, 50), 1)

        mosaico[y0 : y0 + tam_celda, x0 : x0 + tam_celda] = recorte_res

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), mosaico)
    print(f"Lámina de inspección guardada en {ruta_salida}")


def ejecutar():
    fuente = FUENTE_PREDETERMINADA.expanduser().resolve()
    if not fuente.exists():
        raise FileNotFoundError(f"No existe fuente LFW en {fuente}")

    print("Calculando exclusiones históricas...")
    identidades_excluidas, hashes_excluidos, manifiestos_excluidos = leer_todas_las_exclusiones()
    print(f"Identidades excluidas: {len(identidades_excluidas)} desde {len(manifiestos_excluidos)} manifiestos.")

    print("Verificando inventario LFW...")
    inventario = verificar_inventario(fuente, identidades_excluidas)
    print("Inventario:", inventario)

    print("Inicializando detector YuNet...")
    detector = crear_detector()

    print(f"Seleccionando y detectando {CANTIDAD} identidades bajo semilla {SEMILLA}...")
    seleccionados, rechazados = seleccionar_y_detectar(fuente, identidades_excluidas, hashes_excluidos, detector)
    print(f"Seleccionadas {len(seleccionados)} identidades. Rechazados durante el proceso: {len(rechazados)}.")

    dudosas = [s for s in seleccionados if s["dudosa"]]
    print(f"Cajas marcadas como dudosas: {len(dudosas)}")

    destino = REPOSITORIO / "data" / "desconocidos_lfw" / "entrenamiento_4clases"
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)

    # Copiar fotos y escribir manifiesto
    ruta_manifiesto = destino / "manifiesto.csv"
    with ruta_manifiesto.open("w", encoding="utf-8", newline="") as salida:
        campos = (
            "identidad_lfw",
            "identidad_normalizada",
            "ruta_imagen",
            "ruta_fuente",
            "sha256",
            "caja_yunet_x",
            "caja_yunet_y",
            "caja_yunet_w",
            "caja_yunet_h",
            "dudosa",
            "motivo_dudosa",
        )
        escritor = csv.DictWriter(salida, fieldnames=campos)
        escritor.writeheader()

        for item in seleccionados:
            foto_src = item["foto_path"]
            copia_dest = destino / foto_src.name
            shutil.copy2(foto_src, copia_dest)
            x, y, w, h = item["recuadro"]
            escritor.writerow({
                "identidad_lfw": item["identidad_lfw"],
                "identidad_normalizada": item["identidad_normalizada"],
                "ruta_imagen": copia_dest.relative_to(REPOSITORIO).as_posix(),
                "ruta_fuente": foto_src.relative_to(fuente).as_posix(),
                "sha256": item["sha256"],
                "caja_yunet_x": x,
                "caja_yunet_y": y,
                "caja_yunet_w": w,
                "caja_yunet_h": h,
                "dudosa": str(item["dudosa"]),
                "motivo_dudosa": item["motivo_dudosa"],
            })

    # Guardar registro de exclusión y rechazo
    registro_rechazo = {
        "semilla": SEMILLA,
        "candidatos_rechazados": rechazados,
        "total_rechazados": len(rechazados),
        "resumen_causas_rechazo": dict(
            Counter=[r["causa"] for r in rechazados]
        ),
    }
    with (destino / "rechazados.json").open("w", encoding="utf-8") as f:
        json.dump(registro_rechazo, f, ensure_ascii=False, indent=2)

    # Procedencia
    procedencia = {
        "dataset": "Labeled Faces in the Wild (LFW)",
        "variante": "lfw_funneled",
        "fuente_local": str(fuente),
        "uso": "Entrenamiento de la clase 'desconocido' en experimento controlado de 4 clases.",
        "semilla": SEMILLA,
        "total_imagenes": len(seleccionados),
        "total_identidades": len(seleccionados),
        "manifiestos_historicos_excluidos": manifiestos_excluidos,
        "inventario": inventario,
        "dudosas_detectadas": len(dudosas),
    }
    with (destino / "procedencia.json").open("w", encoding="utf-8") as f:
        json.dump(procedencia, f, ensure_ascii=False, indent=2)

    # Limitaciones del dataset
    limitaciones = (
        "# Limitaciones del Conjunto de Desconocidos LFW (4 Clases)\n\n"
        "1. **Discrepancia de Dominio (Domain Shift):**\n"
        "   LFW consiste en fotografías de celebridades y figuras públicas en interiores/exteriores con iluminación "
        "   heterogénea, sensores profesionales y posprocesamiento fotográfico de prensa, mientras que el sistema de "
        "   acceso opera con sensor webcam CMOS de baja resolución y rango dinámico limitado.\n\n"
        "2. **Prealineamiento funneled:**\n"
        "   Las imágenes originales de LFW funneled tienen rotaciones y escalas corregidas algorítmicamente, reduciendo "
        "   la variabilidad angular respecto a las poses libres de un sujeto frente a la webcam.\n\n"
        "3. **Representatividad Negativa Parcial:**\n"
        "   Aunque LFW provee 300 identidades para inducir una frontera de decisión para la clase 'desconocido', "
        "   no representa adecuadamente negativos bajo el entorno físico del sistema (mismo fondo de oficina/habitación, "
        "   misma iluminación o distancia del sensor).\n\n"
        "4. **Aislamiento Estricto:**\n"
        "   Se excluyeron todas las 300 identidades usadas en calibración o reserva de prueba, garantizando que "
        "   ningún sujeto de evaluación haya sido observado en este entrenamiento.\n"
    )
    with (destino / "limitaciones.md").open("w", encoding="utf-8") as f:
        f.write(limitaciones)

    # Generar lámina de inspección
    lamina_path = REPOSITORIO / "laminas" / "inspeccion_lfw_entrenamiento_4clases.png"
    generar_lamina_inspeccion(seleccionados, lamina_path)

    print(f"Finalizado con éxito en {destino}")


if __name__ == "__main__":
    ejecutar()
