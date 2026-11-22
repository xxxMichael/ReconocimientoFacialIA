"""
Evaluador offline de sujetos desconocidos reales adquiridos para control de acceso.

Reutiliza directamente el PipelineCandidatoCongelado con el modelo CNN de 4 clases
y el umbral calibrado 0.46 para evaluar el conjunto 'desconocidos_prueba'.
Exporta decisiones individuales y resumen estadístico sin modificar pesos ni umbrales.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reconocimiento.clasificacion.pipeline_candidato import (
    PipelineCandidatoCongelado,
    ErrorConfiguracionCandidato,
)
from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    DESTINOS_DESCONOCIDOS_VALIDOS,
)


def evaluar_capturas_desconocidos_reales(
    ruta_config: Path,
    ruta_manifiesto: Path,
    directorio_salida: Path,
    grupo_objetivo: str = "desconocidos_prueba",
) -> Dict[str, Any]:
    """
    Evalúa exhaustivamente las capturas de voluntarios desconocidos reales
    utilizando el pipeline del candidato congelado.
    """
    directorio_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" EVALUACIÓN OFFLINE DE SUJETOS DESCONOCIDOS REALES")
    print("=" * 65)
    print(f" Configuración candidata: {ruta_config}")
    print(f" Manifiesto evaluado:    {ruta_manifiesto}")
    print(f" Grupo objetivo:         {grupo_objetivo}")
    print(f" Directorio de salida:   {directorio_salida}")
    print("-" * 65)

    if not ruta_config.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración congelada: {ruta_config}")

    if not ruta_manifiesto.exists():
        print(f"[AVISO] No se encontró el archivo de manifiesto en: {ruta_manifiesto}")
        resumen_vacio = {
            "status": "sin_datos",
            "mensaje": f"El manifiesto {ruta_manifiesto} no existe. Aún no se han capturado fotos de desconocidos reales.",
            "total_imagenes": 0,
            "aperturas_indebidas_far": 0,
            "tasa_far_porcentaje": 0.0,
        }
        with open(directorio_salida / "resumen_evaluacion_desconocidos.json", "w", encoding="utf-8") as f:
            json.dump(resumen_vacio, f, indent=2, ensure_ascii=False)
        return resumen_vacio

    # Cargar pipeline congelado
    pipeline = PipelineCandidatoCongelado(ruta_config=ruta_config)
    print(f" Candidato cargado:      {pipeline.nombre}")
    print(f" Umbral de corte:        {pipeline.umbral_score}")
    print(f" Clases:                 {pipeline.orden_clases}")
    print(f" Autorizados:            {pipeline.personas_autorizadas}")
    print("-" * 65)

    manifiesto = ManifiestoDataset(ruta_manifiesto)
    registros_todos = manifiesto.registros

    # Filtrar únicamente registros pertenecientes al grupo objetivo de desconocidos
    registros_desconocidos = []
    for r in registros_todos:
        es_grupo = (r.particion == grupo_objetivo) or (r.destino_experimental == grupo_objetivo)
        es_tipo = (r.tipo_sujeto == "desconocido")
        if es_grupo or (es_tipo and r.particion in ("prueba", "desconocidos_prueba")):
            registros_desconocidos.append(r)

    print(f" Total de registros encontrados en '{grupo_objetivo}': {len(registros_desconocidos)}")

    if not registros_desconocidos:
        print("[AVISO] No se encontraron registros de desconocidos reales en el manifiesto.")
        resumen_sin_fotos = {
            "status": "sin_capturas",
            "mensaje": f"No se hallaron registros correspondientes al grupo '{grupo_objetivo}' en el manifiesto.",
            "total_registros_manifiesto": len(registros_todos),
            "total_imagenes_evaluadas": 0,
            "aperturas_indebidas_far": 0,
            "tasa_far_porcentaje": 0.0,
        }
        with open(directorio_salida / "resumen_evaluacion_desconocidos.json", "w", encoding="utf-8") as f:
            json.dump(resumen_sin_fotos, f, indent=2, ensure_ascii=False)
        return resumen_sin_fotos

    decisiones_export: List[Dict[str, Any]] = []
    total_evaluadas = 0
    total_con_deteccion = 0
    total_sin_deteccion = 0
    total_aperturas_indebidas = 0  # FAR: Puerta ABIERTA para un desconocido
    total_rechazos_correctos = 0   # Puerta CERRADA para un desconocido

    desglose_por_sujeto: Dict[str, Dict[str, int]] = {}
    desglose_por_distancia: Dict[str, Dict[str, int]] = {}
    desglose_por_pose: Dict[str, Dict[str, int]] = {}

    for reg in registros_desconocidos:
        ruta_img = Path(reg.ruta)
        if not ruta_img.is_absolute():
            ruta_img = Path.cwd() / ruta_img

        if not ruta_img.exists():
            print(f"[ALERTA] Archivo de imagen no encontrado en disco: {ruta_img}")
            continue

        img_bgr = cv2.imread(str(ruta_img))
        if img_bgr is None:
            print(f"[ALERTA] No se pudo decodificar la imagen: {ruta_img}")
            continue

        total_evaluadas += 1
        sujeto = reg.persona
        pose = reg.pose
        distancia = reg.distancia or "no_especificada"

        # Inicializar desgloses
        for d_map, clave in [(desglose_por_sujeto, sujeto), (desglose_por_distancia, distancia), (desglose_por_pose, pose)]:
            if clave not in d_map:
                d_map[clave] = {"total": 0, "aperturas_indebidas": 0, "bloqueos_correctos": 0}
            d_map[clave]["total"] += 1

        # Ejecutar pipeline canónico
        res_det, decision, probs = pipeline.evaluar_fotograma_completo(img_bgr, es_pre_recortada=False)

        if res_det.estado.value == "UN_ROSTRO":
            total_con_deteccion += 1
        else:
            total_sin_deteccion += 1

        apertura = (decision.estado_puerta == "ABIERTA")
        if apertura:
            total_aperturas_indebidas += 1
            for d_map, clave in [(desglose_por_sujeto, sujeto), (desglose_por_distancia, distancia), (desglose_por_pose, pose)]:
                d_map[clave]["aperturas_indebidas"] += 1
        else:
            total_rechazos_correctos += 1
            for d_map, clave in [(desglose_por_sujeto, sujeto), (desglose_por_distancia, distancia), (desglose_por_pose, pose)]:
                d_map[clave]["bloqueos_correctos"] += 1

        probs_dict = {}
        if probs is not None:
            for idx, cname in pipeline.mapeo_id_clase.items():
                if idx < len(probs):
                    probs_dict[cname] = round(float(probs[idx]), 4)

        fila_csv = {
            "identificador": reg.identificador,
            "persona": sujeto,
            "sesion": reg.sesion,
            "pose": pose,
            "distancia": distancia,
            "particion": reg.particion,
            "destino": reg.destino_experimental,
            "ruta": str(ruta_img),
            "estado_deteccion": res_det.estado.value,
            "clase_predicha": decision.persona_identificada,
            "es_autorizado": decision.es_autorizado,
            "score": round(decision.confianza, 4) if decision.confianza is not None else 0.0,
            "umbral": pipeline.umbral_score,
            "estado_puerta": decision.estado_puerta,
            "mensaje_acceso": decision.mensaje,
            "es_apertura_indebida": apertura,
            "es_rechazo_correcto": not apertura,
            "probabilidades_json": json.dumps(probs_dict),
        }
        decisiones_export.append(fila_csv)

    # Exportar CSV de decisiones individuales
    ruta_csv = directorio_salida / "decisiones_desconocidos_reales.csv"
    if decisiones_export:
        columnas = list(decisiones_export[0].keys())
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            writer.writerows(decisiones_export)
        print(f" Decisiones individuales guardadas en: {ruta_csv}")

    tasa_far = round((total_aperturas_indebidas / total_evaluadas) * 100.0, 2) if total_evaluadas > 0 else 0.0
    tasa_rechazo = round((total_rechazos_correctos / total_evaluadas) * 100.0, 2) if total_evaluadas > 0 else 0.0

    resumen_final = {
        "status": "ok",
        "candidato": pipeline.nombre,
        "umbral_evaluado": pipeline.umbral_score,
        "total_imagenes_evaluadas": total_evaluadas,
        "detecciones_exitosas": total_con_deteccion,
        "fallos_o_ausencia_deteccion": total_sin_deteccion,
        "aperturas_indebidas_far": total_aperturas_indebidas,
        "bloqueos_correctos": total_rechazos_correctos,
        "tasa_far_porcentaje": tasa_far,
        "tasa_rechazo_porcentaje": tasa_rechazo,
        "desglose_por_sujeto": desglose_por_sujeto,
        "desglose_por_distancia": desglose_por_distancia,
        "desglose_por_pose": desglose_por_pose,
        "ruta_detalle_csv": str(ruta_csv),
    }

    ruta_json = directorio_salida / "resumen_evaluacion_desconocidos.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen_final, f, indent=2, ensure_ascii=False)
    print(f" Resumen de métricas guardado en:      {ruta_json}")

    print("\n" + "=" * 65)
    print(f" RESULTADOS FINALES: Evaluadas: {total_evaluadas} | FAR: {tasa_far}% ({total_aperturas_indebidas}/{total_evaluadas}) | Rechazo: {tasa_rechazo}%")
    print("=" * 65 + "\n")

    return resumen_final


def main():
    parser = argparse.ArgumentParser(description="Evaluación de sujetos desconocidos reales con candidato congelado")
    parser.add_argument(
        "--config-candidato",
        type=str,
        default="configuracion_candidato_congelado_corregida.json",
        help="Ruta al archivo JSON de configuración del candidato congelado"
    )
    parser.add_argument(
        "--manifiesto",
        type=str,
        default="data/propio/manifiesto.csv",
        help="Ruta al manifiesto de capturas propio"
    )
    parser.add_argument(
        "--salida",
        type=str,
        default="resultados/evaluacion_desconocidos_reales",
        help="Directorio de salida para los resultados y reportes"
    )
    parser.add_argument(
        "--grupo",
        type=str,
        default="desconocidos_prueba",
        help="Grupo experimental a evaluar (default: desconocidos_prueba)"
    )

    args = parser.parse_args()
    evaluar_capturas_desconocidos_reales(
        ruta_config=Path(args.config_candidato),
        ruta_manifiesto=Path(args.manifiesto),
        directorio_salida=Path(args.salida),
        grupo_objetivo=args.grupo,
    )


if __name__ == "__main__":
    main()
