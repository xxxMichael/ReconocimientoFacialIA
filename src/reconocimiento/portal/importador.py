"""
Herramienta de importación local explícita de capturas externas al manifiesto del proyecto.
Reutiliza estrictamente las validaciones del protocolo de ManifiestoDataset y RegistroCaptura,
respetando automáticamente la presencia de 'manifiesto_local.csv' cuando corresponda.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    calcular_hash_archivo,
)
from reconocimiento.portal.almacenamiento import GestorAlmacenamientoExterno


def importar_capturas_externas(
    dir_recepcion: Optional[Path] = None,
    ruta_manifiesto_destino: Optional[Path] = None,
    identificador_filtro: Optional[str] = None,
    excluir_identificadores: Optional[List[str]] = None,
    excluir_pilotos: bool = True,
    permitir_excepciones_piloto: Optional[List[str]] = None,
    raiz_proyecto: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Importa capturas desde la carpeta de recepción independiente hacia el manifiesto del proyecto.
    
    Retorna un diccionario con estadísticas de importación:
    - total_encontradas: registros leídos en recepción
    - importadas_nuevas: registros agregados y validados
    - omitidas_ya_existentes: registros que ya constaban en el manifiesto
    - omitidas_piloto: capturas de pruebas o ensayos pilotos excluidas
    - errores: lista de incidencias o fallos de validación
    - ruta_manifiesto: ruta del manifiesto actualizado
    """
    raiz = Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[3]

    gestor_alm = GestorAlmacenamientoExterno(dir_base=dir_recepcion, raiz_proyecto=raiz)
    registros_recepcion = gestor_alm.listar_todos_los_registros()

    lista_excluir = [x.strip().lower() for x in (excluir_identificadores or []) if x.strip()]
    excepciones_set = {x.strip().lower() for x in (permitir_excepciones_piloto or []) if x.strip()}

    if identificador_filtro:
        id_f = identificador_filtro.strip().lower()
        registros_recepcion = [r for r in registros_recepcion if r.get("identificador_sujeto") == id_f]

    manifiesto = ManifiestoDataset(ruta_archivo=ruta_manifiesto_destino)
    ids_existentes = manifiesto.identificadores_existentes()
    hashes_existentes = {r.hash_contenido for r in manifiesto.registros if r.hash_contenido}

    importadas = 0
    omitidas = 0
    omitidas_piloto = 0
    errores: List[Dict[str, str]] = []

    for reg in registros_recepcion:
        id_sujeto = reg.get("identificador_sujeto", "").strip().lower()
        pose = reg.get("pose", "")
        distancia = reg.get("distancia", "")
        rel_archivo = reg.get("archivo_relativo", "")
        sha256 = reg.get("sha256", "")
        fecha = reg.get("fecha_utc", "")

        id_captura = f"{id_sujeto}_{pose}_{distancia}"

        # 1. Filtro estricto de exclusión de pilotos / ensayos
        if id_sujeto in lista_excluir:
            omitidas_piloto += 1
            continue

        if id_sujeto not in excepciones_set:
            if excluir_pilotos and ("piloto" in id_sujeto or "test" in id_sujeto or "ensayo" in id_sujeto):
                omitidas_piloto += 1
                continue

        if id_captura in ids_existentes:
            omitidas += 1
            continue

        if sha256 and sha256 in hashes_existentes:
            omitidas += 1
            continue

        ruta_fisica = gestor_alm.dir_base / rel_archivo
        if not ruta_fisica.exists():
            errores.append({
                "identificador": id_captura,
                "error": f"Archivo físico no encontrado en disco: {ruta_fisica}"
            })
            continue

        # Validar integridad SHA-256
        hash_real = calcular_hash_archivo(ruta_fisica)
        if sha256 and hash_real != sha256:
            errores.append({
                "identificador": id_captura,
                "error": f"Discrepancia de integridad: hash esperado {sha256[:10]}... != real {hash_real[:10]}..."
            })
            continue

        # Ruta relativa normalizada hacia la raíz del proyecto
        try:
            ruta_para_manifiesto = str(ruta_fisica.relative_to(raiz)).replace("\\", "/")
        except ValueError:
            ruta_para_manifiesto = str(ruta_fisica).replace("\\", "/")

        anotacion = (
            f"destino=desconocidos_prueba;distancia={distancia};"
            f"origen=portal_web_externo;fecha={fecha};hash={hash_real}"
        )

        registro_nuevo = RegistroCaptura(
            identificador=id_captura,
            ruta=ruta_para_manifiesto,
            procedencia="portal_web_externo",
            persona=id_sujeto,
            tipo_sujeto="desconocido",
            sesion="sesion_externa",
            pose=pose,
            particion="prueba",
            consentimiento=True,
            anotaciones=anotacion,
            hash_contenido=hash_real,
        )

        try:
            manifiesto.agregar_registro(registro_nuevo)
            ids_existentes.add(id_captura)
            hashes_existentes.add(hash_real)
            importadas += 1
        except Exception as e:
            errores.append({
                "identificador": id_captura,
                "error": str(e)
            })

    if importadas > 0:
        manifiesto.guardar()

    return {
        "total_encontradas": len(registros_recepcion),
        "importadas_nuevas": importadas,
        "omitidas_ya_existentes": omitidas,
        "omitidas_piloto": omitidas_piloto,
        "errores": errores,
        "ruta_manifiesto": str(manifiesto.ruta_archivo),
    }
