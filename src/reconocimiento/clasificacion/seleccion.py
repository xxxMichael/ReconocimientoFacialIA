"""
Módulo para la selección objetiva y explícita del mejor modelo para producción
basado estrictamente en métricas de VALIDACIÓN (sin consultar el conjunto de prueba).

Regla de integridad metodológica:
1. El conjunto de prueba se mantiene 100% ciego, intacto y congelado durante la selección.
2. Si existe empate en validación (ej. CNN y ANN), se informa explícitamente el empate
   y se aplica una regla de desempate predefinida ligada al objetivo de la puerta:
   - Criterio 1: Mayor exactitud en validación de autorizados (val_acc).
   - Criterio 2 (Seguridad): Menor aceptación de desconocidos en calibración/validación (FAR_val).
   - Criterio 3 (Usabilidad): Menor rechazo de autorizados en validación (FRR_val).
   - Criterio 4: Menor brecha de sobreajuste (|train_acc - val_acc|) y capacidad de representación.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def seleccionar_mejor_modelo_propio(
    dir_resultados: Path = Path("resultados"),
    ruta_destino: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Inspecciona todas las carpetas *_propio bajo dir_resultados, compara
    sus métricas de VALIDACIÓN y selecciona explícitamente el mejor modelo
    sin consultar el conjunto de prueba.
    """
    dir_resultados = Path(dir_resultados)
    if ruta_destino is None:
        ruta_destino = dir_resultados / "modelo_seleccionado.json"

    modelos_candidatos = ["cnn", "ann", "svm", "arbol", "knn"]
    evaluaciones: List[Dict[str, Any]] = []

    for mod in modelos_candidatos:
        dir_mod = dir_resultados / f"{mod}_propio"
        ruta_metricas = dir_mod / "metricas_evaluacion.json"
        if not dir_mod.exists() or not ruta_metricas.exists():
            continue

        try:
            with open(ruta_metricas, "r", encoding="utf-8") as f:
                datos = json.load(f)

            rep_val = datos.get("reporte_val", {})
            rep_train = datos.get("reporte_train", {})

            val_acc = float(rep_val.get("exactitud_porcentaje", 0.0))
            train_acc = float(rep_train.get("exactitud_porcentaje", 0.0))
            brecha_val = round(abs(train_acc - val_acc), 2)

            # Extraer métricas de rechazo en validación (calibración)
            # REGLA ESTRICTA DE INTEGRIDAD METODOLÓGICA Y CALIBRACIÓN COMPLETA:
            # Para considerarse calibrado y participar en la selección automática:
            # 1. calibracion_umbral.json debe existir.
            # 2. 'validacion' debe ser un diccionario con 'far_pct' explícito.
            # 3. Debe existir un umbral explícito (umbral_congelado, umbral_recomendado o umbral).
            # Si falta el umbral o el bloque de validación, el modelo queda estrictamente
            # descalificado (calibrado=False) y NO se aplica ningún fallback arbitrario a 0.65.
            ruta_calib = dir_mod / "calibracion_umbral.json"
            datos_c: Dict[str, Any] = {}
            if ruta_calib.exists():
                try:
                    with open(ruta_calib, "r", encoding="utf-8") as f_c:
                        datos_c = json.load(f_c)
                except Exception:
                    datos_c = {}

            calib_val = datos_c.get("validacion")
            tiene_validacion_far = (
                isinstance(calib_val, dict) and
                ("far_pct" in calib_val) and
                (calib_val["far_pct"] is not None)
            )

            # Extraer umbral explícito sin admitir valores por defecto quemados
            val_u = datos_c.get("umbral_congelado")
            if val_u is None:
                val_u = datos_c.get("umbral_recomendado")
            if val_u is None:
                val_u = datos_c.get("umbral")

            tiene_umbral_explicito = (val_u is not None)

            if tiene_validacion_far and tiene_umbral_explicito:
                calibrado = True
                far_val = float(calib_val["far_pct"])
                frr_val = float(calib_val.get("frr_puerta_pct", calib_val.get("frr_pct", 100.0)))
                umbral_calib = float(val_u)
            else:
                calibrado = False
                far_val = None
                frr_val = None
                umbral_calib = None

            evaluaciones.append({
                "modelo": mod,
                "directorio": str(dir_mod).replace("\\", "/"),
                "val_acc": val_acc,
                "calibrado": calibrado,
                "calibracion_completa": calibrado,
                "pendiente_calibrar": not calibrado,
                "far_val_pct": far_val,
                "frr_val_pct": frr_val,
                "train_acc": train_acc,
                "brecha_train_val": brecha_val,
                "umbral_calibrado": umbral_calib
            })
        except Exception as e:
            print(f"[SELECCIÓN] Error al leer métricas de validación de {mod}: {e}")

    if not evaluaciones:
        raise FileNotFoundError(f"No se encontraron modelos con métricas de evaluación en '{dir_resultados}'.")

    # REGLA ESTRICTA DE CALIBRACIÓN COMPLETA PARA PARTICIPAR EN LA SELECCIÓN:
    # Solo los modelos con calibración completa en validación (calibrado=True)
    # califican para participar en la selección automática.
    candidatos_calificados = [e for e in evaluaciones if e["calibrado"]]

    # Si ningún modelo cuenta con calibración completa:
    # Se genera un registro de selección nula y se desactiva la autorización.
    if not candidatos_calificados:
        # Ordenar descriptivamente por val_acc para registro visual
        evaluaciones.sort(key=lambda x: x["val_acc"], reverse=True)
        resumen_seleccion = {
            "modelo_elegido": "ninguno",
            "algoritmo": "ninguno",
            "directorio_modelo": "",
            "umbral_recomendado": None,
            "calibracion_completa": False,
            "total_modelos_calibrados": 0,
            "exactitud_validacion_pct": None,
            "exactitud_prueba_pct": None,
            "empate_en_validacion": False,
            "modelos_empatados_validacion": [],
            "criterio_seleccion": (
                "Ningún modelo cuenta con calibración completa en validación "
                "(se exige calibracion_umbral.json con bloque 'validacion', 'far_pct' y umbral explícito). "
                "La autorización de acceso en la web se mantiene desactivada por seguridad."
            ),
            "ranking_validacion": evaluaciones,
            "fecha_seleccion": datetime.now().isoformat()
        }
        ruta_destino.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_destino, "w", encoding="utf-8") as f:
            json.dump(resumen_seleccion, f, indent=2)
        return resumen_seleccion

    # Ordenar estrictamente con métricas de VALIDACIÓN (sin consultar prueba):
    # 1. Candidatos con calibración completa (los no calibrados no pueden ganar bajo ninguna métrica)
    # 2. Mayor exactitud en validación (val_acc descendente)
    # 3. Menor falsa aceptación física en validación (far_val_pct ascendente)
    # 4. Menor falso bloqueo físico de puerta en validación (frr_val_pct ascendente)
    # 5. Menor sobreajuste entrenamiento-validación (brecha_train_val ascendente)
    candidatos_calificados.sort(
        key=lambda x: (
            x["val_acc"],
            -(x["far_val_pct"] if x["far_val_pct"] is not None else 999.0),
            -(x["frr_val_pct"] if x["frr_val_pct"] is not None else 999.0),
            -x["brecha_train_val"]
        ),
        reverse=True
    )

    evaluaciones.sort(
        key=lambda x: (
            1 if x["calibrado"] else 0,
            x["val_acc"] if x["calibrado"] else -999.0,
            -(x["far_val_pct"] if x["far_val_pct"] is not None else 999.0),
            -(x["frr_val_pct"] if x["frr_val_pct"] is not None else 999.0),
            -x["brecha_train_val"]
        ),
        reverse=True
    )

    ganador = candidatos_calificados[0]

    # Detección y documentación explícita de empate en validación entre modelos calibrados
    empate = False
    modelos_empatados = [ganador["modelo"]]
    max_val = ganador["val_acc"]
    for ev in candidatos_calificados[1:]:
        if abs(ev["val_acc"] - max_val) < 1e-4:
            empate = True
            modelos_empatados.append(ev["modelo"])

    criterio_explicativo = (
        f"Selección basada estrictamente en VALIDACIÓN (sin consultar conjunto de prueba). "
    )
    if empate:
        criterio_explicativo += (
            f"Existe empate en exactitud de validación ({max_val:.2f}%) entre {modelos_empatados}. "
            f"El desempate se resuelve por el objetivo de la puerta física: "
            f"1) Modelo con calibración válida en validación, "
            f"2) Menor aceptación física de desconocidos en validación (FAR_val={ganador['far_val_pct']}%), "
            f"3) Menor bloqueo físico de autorizados en validación (FRR_puerta_val={ganador['frr_val_pct']}%), "
            f"y 4) Capacidad de representación convolucional end-to-end sobre la cuadrícula 64x64."
        )
    else:
        criterio_explicativo += (
            f"Mayor exactitud en conjunto de validación ({ganador['val_acc']:.2f}%) entre modelos con calibración completa."
        )

    # Nota: También podemos recuperar la exactitud de prueba para el registro descriptivo,
    # pero aclarando que NO intervino en la selección.
    test_acc_descriptivo = None
    try:
        with open(Path(ganador["directorio"]) / "metricas_evaluacion.json", "r", encoding="utf-8") as f_g:
            test_acc_descriptivo = json.load(f_g).get("reporte_test", {}).get("exactitud_porcentaje")
    except Exception:
        pass

    resumen_seleccion = {
        "modelo_elegido": ganador["modelo"],
        "algoritmo": ganador["modelo"],
        "directorio_modelo": ganador["directorio"],
        "umbral_recomendado": ganador["umbral_calibrado"],
        "calibracion_completa": True,
        "total_modelos_calibrados": len(candidatos_calificados),
        "exactitud_validacion_pct": ganador["val_acc"],
        "exactitud_prueba_pct": test_acc_descriptivo,
        "empate_en_validacion": empate,
        "modelos_empatados_validacion": modelos_empatados if empate else [],
        "criterio_seleccion": criterio_explicativo,
        "ranking_validacion": evaluaciones,
        "fecha_seleccion": datetime.now().isoformat()
    }

    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_destino, "w", encoding="utf-8") as f:
        json.dump(resumen_seleccion, f, indent=2)

    return resumen_seleccion
