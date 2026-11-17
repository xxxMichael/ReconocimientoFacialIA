"""
Interfaz de línea de comandos (CLI) para el sistema de reconocimiento facial.
Cumple con el contrato de ejecución y subcomandos especificados en ENCARGO_IMPLEMENTACION.md.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np

from reconocimiento.adquisicion.olivetti import cargar_olivetti
from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    generar_manifiesto_plantilla_300,
    POSES_VALIDAS,
    PARTICIONES_VALIDAS
)
from reconocimiento.adquisicion.camara import (
    sondear_camaras,
    ejecutar_vista_previa,
    ejecutar_modo_captura
)
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion
from reconocimiento.preprocesamiento.normalizador import (
    a_escala_de_grises,
    normalizar_intensidades,
    normalizar_resolucion
)
from reconocimiento.preprocesamiento.filtros import (
    generar_patron_sintetico_franjas,
    submuestrear,
    suavizado_gaussiano
)
from reconocimiento.preprocesamiento.mascara import (
    aislar_componente_mayor_8vecinos,
    generar_mascara_binaria,
    verificar_conectividad_8_vs_4
)
from reconocimiento.caracteristicas.textura import ExtractorHistogramasRegionales
from reconocimiento.clasificacion.base import guardar_artefactos_completos
from reconocimiento.clasificacion.knn import ClasificadorKNN
from reconocimiento.clasificacion.arbol import ClasificadorArbol
from reconocimiento.clasificacion.svm import ClasificadorSVM
from reconocimiento.clasificacion.ann import ClasificadorANN
from reconocimiento.clasificacion.cnn import ClasificadorCNN


def comando_ejecutar(args: argparse.Namespace) -> int:
    """
    Contrato: python -m reconocimiento ejecutar --modelo knn --dataset olivetti --salida resultados/knn
    """
    modelo_nombre = args.modelo.lower()
    dataset_nombre = args.dataset.lower()
    if args.salida == "resultados/knn" and (modelo_nombre != "knn" or dataset_nombre == "propio"):
        sufijo = "_propio" if dataset_nombre == "propio" else ""
        dir_salida = Path(f"resultados/{modelo_nombre}{sufijo}")
    else:
        dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" EJECUCIÓN PIPELINE: Modelo={modelo_nombre.upper()} | Dataset={dataset_nombre.upper()}")
    print(f" Directorio de salida: {dir_salida}")
    print(f"=======================================================\n")

    if modelo_nombre not in {"knn", "arbol", "svm", "ann", "cnn"}:
        print(f"Error: Modelo '{modelo_nombre}' no reconocido. Opciones: knn, arbol, svm, ann, cnn.")
        return 1

    if dataset_nombre not in {"olivetti", "propio"}:
        print(f"Error: Dataset '{dataset_nombre}' no reconocido para esta ejecución. Use 'olivetti' o 'propio'.")
        return 1

    if dataset_nombre == "olivetti":
        # Fase 1: Adquisición
        print("[1/4] Adquiriendo dataset Olivetti con partición 3/1/1 estratificada por sujeto...")
        dataset = cargar_olivetti(semilla=args.semilla)
        print(f"  - Sujetos (clases): {dataset.n_clases}")
        print(f"  - Muestras Entrenamiento: {len(dataset.y_train)} (60%)")
        print(f"  - Muestras Validación:    {len(dataset.y_val)} (20%)")
        print(f"  - Muestras Prueba:        {len(dataset.y_test)} (20%)")
        print(f"  - Dimensión original:     {dataset.dimensiones_originales}")
        print(f"  - Estado: imágenes ya recortadas y centradas (pre_recortada=True).")

        # Fase 2: Preprocesamiento
        print(f"\n[2/4] Preprocesando imágenes (normalización y suavizado opcional sigma={args.sigma})...")
        X_train_prep = np.array([suavizado_gaussiano(img, sigma=args.sigma) for img in dataset.X_train])
        X_val_prep = np.array([suavizado_gaussiano(img, sigma=args.sigma) for img in dataset.X_val])
        X_test_prep = np.array([suavizado_gaussiano(img, sigma=args.sigma) for img in dataset.X_test])
        print(f"  - Resolución normalizada: {X_train_prep.shape[1:]} píxeles")
        print(f"  - Rango de intensidades: [{X_train_prep.min():.2f}, {X_train_prep.max():.2f}]")

        # Fase 3: Extracción de características
        if modelo_nombre != "cnn":
            grid = (args.grid_y, args.grid_x)
            print(f"\n[3/4] Extrayendo histogramas de intensidad por regiones ({grid[0]}x{grid[1]}, bins={args.bins})...")
            extractor = ExtractorHistogramasRegionales(
                cuadricula=grid,
                n_bins=args.bins,
                rango_intensidad=(0.0, 1.0)
            )
            X_train_feat, nombres_feat = extractor.extraer_lote(X_train_prep)
            X_val_feat, _ = extractor.extraer_lote(X_val_prep)
            X_test_feat, _ = extractor.extraer_lote(X_test_prep)
            print(f"  - Dimensión del vector de características: {extractor.dimension_total}")
            print(f"  - Primeras 3 componentes: {nombres_feat[:3]}")
        else:
            print("\n[3/4] Representación espacial para CNN: Matrices 2D (64x64 gris) - Aprendizaje de Representación de extremo a extremo.")

        # Fase 4: Clasificación
        print(f"\n[4/4] Entrenando y evaluando clasificador {modelo_nombre.upper()}...")
        configuraciones_validadas = None
        scores_o_probs = None
        historia_train = None

        if modelo_nombre == "knn":
            knn = ClasificadorKNN(
                k=args.k,
                metrica=args.metrica,
                ponderacion=args.ponderacion
            )
            resumen = knn.ejecutar_experimento_completo(
                X_train=X_train_feat,
                y_train=dataset.y_train,
                ids_train=dataset.ids_train,
                X_val=X_val_feat,
                y_val=dataset.y_val,
                ids_val=dataset.ids_val,
                X_test=X_test_feat,
                y_test=dataset.y_test,
                ids_test=dataset.ids_test,
                nombre_dataset=dataset_nombre
            )
            y_test_pred = knn.predecir(X_test_feat)
            scores_o_probs = knn.predecir_probabilidades(X_test_feat)

        elif modelo_nombre == "arbol":
            modelo_arbol, configuraciones_validadas, mejor_cfg = ClasificadorArbol.seleccionar_y_entrenar(
                X_train=X_train_feat,
                y_train=dataset.y_train,
                X_val=X_val_feat,
                y_val=dataset.y_val,
                semilla=args.semilla
            )
            resumen = modelo_arbol.ejecutar_experimento_completo(
                X_train=X_train_feat,
                y_train=dataset.y_train,
                ids_train=dataset.ids_train,
                X_val=X_val_feat,
                y_val=dataset.y_val,
                ids_val=dataset.ids_val,
                X_test=X_test_feat,
                y_test=dataset.y_test,
                ids_test=dataset.ids_test,
                nombre_dataset=dataset_nombre
            )
            modelo_arbol.guardar(dir_salida / "modelo.joblib")
            y_test_pred = modelo_arbol.predecir(X_test_feat)
            scores_o_probs = modelo_arbol.predecir_probabilidades(X_test_feat)

        elif modelo_nombre == "svm":
            modelo_svm, configuraciones_validadas, mejor_cfg = ClasificadorSVM.seleccionar_y_entrenar(
                X_train=X_train_feat,
                y_train=dataset.y_train,
                X_val=X_val_feat,
                y_val=dataset.y_val,
                semilla=args.semilla
            )
            resumen = modelo_svm.ejecutar_experimento_completo(
                X_train=X_train_feat,
                y_train=dataset.y_train,
                ids_train=dataset.ids_train,
                X_val=X_val_feat,
                y_val=dataset.y_val,
                ids_val=dataset.ids_val,
                X_test=X_test_feat,
                y_test=dataset.y_test,
                ids_test=dataset.ids_test,
                nombre_dataset=dataset_nombre
            )
            modelo_svm.guardar(dir_salida / "modelo.joblib")
            y_test_pred = modelo_svm.predecir(X_test_feat)
            scores_o_probs = modelo_svm.predecir_probabilidades(X_test_feat)

        elif modelo_nombre == "ann":
            modelo_ann = ClasificadorANN(semilla=args.semilla)
            resumen = modelo_ann.ejecutar_experimento_completo(
                X_train=X_train_feat,
                y_train=dataset.y_train,
                ids_train=dataset.ids_train,
                X_val=X_val_feat,
                y_val=dataset.y_val,
                ids_val=dataset.ids_val,
                X_test=X_test_feat,
                y_test=dataset.y_test,
                ids_test=dataset.ids_test,
                nombre_dataset=dataset_nombre
            )
            modelo_ann.guardar(dir_salida / "modelo.pt")
            y_test_pred = modelo_ann.predecir(X_test_feat)
            scores_o_probs = modelo_ann.predecir_probabilidades(X_test_feat)
            historia_train = modelo_ann.historia_entrenamiento

        elif modelo_nombre == "cnn":
            modelo_cnn = ClasificadorCNN(semilla=args.semilla)
            resumen = modelo_cnn.ejecutar_experimento_completo(
                X_train=X_train_prep,
                y_train=dataset.y_train,
                ids_train=dataset.ids_train,
                X_val=X_val_prep,
                y_val=dataset.y_val,
                ids_val=dataset.ids_val,
                X_test=X_test_prep,
                y_test=dataset.y_test,
                ids_test=dataset.ids_test,
                nombre_dataset=dataset_nombre
            )
            modelo_cnn.guardar(dir_salida / "modelo.pt")
            y_test_pred = modelo_cnn.predecir(X_test_prep)
            scores_o_probs = modelo_cnn.predecir_probabilidades(X_test_prep)
            historia_train = modelo_cnn.historia_entrenamiento

        resumen.configuracion.update({
            "suavizado_sigma": args.sigma,
            "semilla_aleatoria": args.semilla
        })
        if modelo_nombre != "cnn":
            resumen.configuracion.update({
                "cuadricula_regiones": [args.grid_y, args.grid_x],
                "n_bins_histograma": args.bins,
                "dimension_caracteristicas": extractor.dimension_total
            })

        ids_map = {
            "entrenamiento": dataset.ids_train,
            "validacion": dataset.ids_val,
            "prueba": dataset.ids_test
        }

        # Guardar todos los artefactos completos de evaluación
        guardar_artefactos_completos(
            resumen=resumen,
            dir_salida=dir_salida,
            ids_map=ids_map,
            y_test_pred=y_test_pred,
            ids_test=dataset.ids_test,
            y_test_real=dataset.y_test,
            configuraciones_validadas=configuraciones_validadas,
            scores_o_probs=scores_o_probs,
            historia_entrenamiento=historia_train
        )

        with open(dir_salida / "semilla.txt", "w", encoding="utf-8") as f:
            f.write(str(args.semilla) + "\n")

        print("\n-------------------------------------------------------")
        print(f" REPORTE FINAL DE LA EJECUCIÓN ({modelo_nombre.upper()})")
        print("-------------------------------------------------------")
        print(f" Exactitud en Entrenamiento: {resumen.reporte_train.exactitud_porcentaje:.2f}% ({resumen.reporte_train.aciertos}/{resumen.reporte_train.total_muestras})")
        print(f" Exactitud en Validación:    {resumen.reporte_val.exactitud_porcentaje:.2f}% ({resumen.reporte_val.aciertos}/{resumen.reporte_val.total_muestras})")
        print(f" Exactitud en Prueba:        {resumen.reporte_test.exactitud_porcentaje:.2f}% ({resumen.reporte_test.aciertos}/{resumen.reporte_test.total_muestras})")
        print(f" Brecha de generalización:   {resumen.brecha_generalizacion_puntos:.2f} puntos porcentuales")
        print(f" Tiempo inferencia prueba:   {resumen.reporte_test.tiempo_promedio_por_muestra_ms:.3f} ms/muestra")
        print(f" Artefactos guardados en:     {dir_salida.resolve()}")
        print("=======================================================\n")
        return 0

    elif dataset_nombre == "propio":
        ruta_manifiesto = Path(getattr(args, "manifiesto", "data/propio/manifiesto.csv"))
        if not ruta_manifiesto.exists():
            print(f"Error: No se encontró el manifiesto de imágenes propias en: {ruta_manifiesto}")
            return 1

        print(f"[1/4] Adquiriendo dataset propio desde manifiesto: {ruta_manifiesto}")
        manifiesto = ManifiestoDataset(ruta_manifiesto)
        if len(manifiesto.registros) == 0:
            print("Error: El manifiesto no contiene registros.")
            return 1

        es_valido_ses, errores_ses = manifiesto.validar_aislamiento_sesiones()
        if not es_valido_ses:
            print("ERROR CRÍTICO: Violación de aislamiento de sesiones entre particiones:")
            for e in errores_ses:
                print(f"  - {e}")
            return 1

        es_valido_dup, errores_dup = manifiesto.validar_duplicados_entre_particiones()
        if not es_valido_dup:
            print("ERROR CRÍTICO: Violación de capturas duplicadas entre particiones:")
            for e in errores_dup:
                print(f"  - {e}")
            return 1

        conteo = manifiesto.resumen_conteo()
        print(f"  - Total registros en manifiesto: {conteo['total_muestras']}")
        print(f"  - Por partición: {conteo['por_particion']}")
        print(f"  - Por sujeto: {conteo['por_persona']}")
        print(f"  - Por pose: {conteo['por_pose']}")

        # Fase 2: Preprocesamiento y Detección Facial
        print(f"\n[2/4] Preprocesando y detectando rostros con Haar Cascade...")
        detector = DetectorFacial()
        grid = (args.grid_y, args.grid_x)

        datos_prep = {
            "entrenamiento": {"X": [], "y": [], "ids": [], "poses": []},
            "validacion": {"X": [], "y": [], "ids": [], "poses": []},
            "prueba": {"X": [], "y": [], "ids": [], "poses": []}
        }
        # Intentos de desconocidos particionados estrictamente para calibración vs prueba
        datos_desconocidos = {
            "validacion": {"X": [], "y": [], "ids": [], "poses": [], "total_intentos": 0, "fallos_deteccion": 0},
            "prueba": {"X": [], "y": [], "ids": [], "poses": [], "total_intentos": 0, "fallos_deteccion": 0}
        }
        # Conteo de intentos de autorizados por partición (para medir puerta física e2e)
        intentos_autorizados = {
            "entrenamiento": {"total_intentos": 0, "fallos_deteccion": 0, "exitos_deteccion": 0},
            "validacion": {"total_intentos": 0, "fallos_deteccion": 0, "exitos_deteccion": 0},
            "prueba": {"total_intentos": 0, "fallos_deteccion": 0, "exitos_deteccion": 0}
        }

        poses_totales = sorted(list(POSES_VALIDAS))
        stats_deteccion = {
            p: {"total": 0, "exitos": 0, "fallos": 0, "sin_rostro": 0, "varios_rostros": 0, "no_archivo": 0}
            for p in poses_totales
        }
        total_procesadas = 0
        detecciones_exitosas = 0
        fallos_deteccion = 0

        trazabilidad_muestras: Dict[str, Dict[str, any]] = {}

        for r in manifiesto.registros:
            total_procesadas += 1
            pose = r.pose if r.pose in stats_deteccion else "frontal"
            stats_deteccion[pose]["total"] += 1
            tipo_s = getattr(r, "tipo_sujeto", "autorizado")
            part_desc = r.particion if r.particion in ["validacion", "prueba"] else "prueba"

            p_img = Path(r.ruta)
            if not p_img.is_file():
                fallos_deteccion += 1
                stats_deteccion[pose]["fallos"] += 1
                stats_deteccion[pose]["no_archivo"] += 1
                if tipo_s == "desconocido":
                    datos_desconocidos[part_desc]["total_intentos"] += 1
                    datos_desconocidos[part_desc]["fallos_deteccion"] += 1
                elif r.particion in intentos_autorizados:
                    intentos_autorizados[r.particion]["total_intentos"] += 1
                    intentos_autorizados[r.particion]["fallos_deteccion"] += 1

                trazabilidad_muestras[r.identificador] = {
                    "identificador": r.identificador,
                    "persona_real": r.persona,
                    "persona_predicha": "N/A",
                    "correcta": "NO_EVALUADA",
                    "probabilidad": 0.0,
                    "pose": r.pose,
                    "particion": r.particion,
                    "estado_deteccion": "ARCHIVO_NO_ENCONTRADO"
                }
                continue

            img_bgr = cv2.imread(str(p_img))
            if img_bgr is None:
                fallos_deteccion += 1
                stats_deteccion[pose]["fallos"] += 1
                stats_deteccion[pose]["no_archivo"] += 1
                if tipo_s == "desconocido":
                    datos_desconocidos[part_desc]["total_intentos"] += 1
                    datos_desconocidos[part_desc]["fallos_deteccion"] += 1
                elif r.particion in intentos_autorizados:
                    intentos_autorizados[r.particion]["total_intentos"] += 1
                    intentos_autorizados[r.particion]["fallos_deteccion"] += 1

                trazabilidad_muestras[r.identificador] = {
                    "identificador": r.identificador,
                    "persona_real": r.persona,
                    "persona_predicha": "N/A",
                    "correcta": "NO_EVALUADA",
                    "probabilidad": 0.0,
                    "pose": r.pose,
                    "particion": r.particion,
                    "estado_deteccion": "ERROR_DECODIFICACION"
                }
                continue

            # REQUISITO 3:
            # - Para partición de entrenamiento: usar SOLO capturas aprobadas o corregidas con su caja aprobada/corregida.
            # - Para evaluación extremo a extremo (validación/prueba): ejecutar el detector automáticamente sobre el original.
            #   Una corrección manual sirve como referencia para medir el fallo, pero no otorga crédito automático.
            es_particion_entrenamiento = (r.particion == "entrenamiento")

            if es_particion_entrenamiento:
                decision_h = r.decision_humana
                if decision_h not in {"aprobada", "corregida"}:
                    # Captura pendiente o excluida: se omite estrictamente del conjunto de entrenamiento
                    continue

                caja_entrenamiento = r.caja_aprobada
                if caja_entrenamiento is None:
                    # Foto antigua aprobada sin caja registrada explícita: calcular con detector
                    res_temp = detector.detectar(img_bgr, es_pre_recortada=False)
                    caja_entrenamiento = res_temp.recuadro

                if caja_entrenamiento is None:
                    # Sin rostro localizable incluso para entrenamiento
                    continue

                recorte = detector.recortar_con_margen_seguro(img_bgr, caja_entrenamiento)
                gris = a_escala_de_grises(recorte) if len(recorte.shape) == 3 else recorte
                norm_res = normalizar_resolucion(gris, (64, 64))
                norm_int = normalizar_intensidades(norm_res, "float32_0_1")
                suavizada = suavizado_gaussiano(norm_int, sigma=args.sigma)

                detecciones_exitosas += 1
                stats_deteccion[pose]["exitos"] += 1
                intentos_autorizados["entrenamiento"]["total_intentos"] += 1
                intentos_autorizados["entrenamiento"]["exitos_deteccion"] += 1
                datos_prep["entrenamiento"]["X"].append(suavizada)
                datos_prep["entrenamiento"]["y"].append(r.persona)
                datos_prep["entrenamiento"]["ids"].append(r.identificador)
                datos_prep["entrenamiento"]["poses"].append(r.pose)

                trazabilidad_muestras[r.identificador] = {
                    "identificador": r.identificador,
                    "persona_real": r.persona,
                    "persona_predicha": "",
                    "correcta": "",
                    "probabilidad": 0.0,
                    "pose": r.pose,
                    "particion": r.particion,
                    "estado_deteccion": "UN_ROSTRO_REVISADO"
                }
                continue

            # Evaluación de extremo a extremo (validación y prueba): detector automático sobre el original
            res_det = detector.detectar(img_bgr, es_pre_recortada=False)

            # Verificar si detectó un rostro pero cayó fuera de la referencia humana (falso positivo sobre ropa/pelo)
            localizacion_valida = (res_det.estado == EstadoDeteccion.UN_ROSTRO)
            if localizacion_valida and r.caja_aprobada is not None:
                iou, iomin = detector.calcular_iou_y_iomin(res_det.recuadro, r.caja_aprobada)
                if iou < 0.5 and iomin < 0.5:
                    localizacion_valida = False

            if localizacion_valida:
                detecciones_exitosas += 1
                stats_deteccion[pose]["exitos"] += 1

                recorte = res_det.rostro_recortado
                gris = a_escala_de_grises(recorte) if len(recorte.shape) == 3 else recorte
                norm_res = normalizar_resolucion(gris, (64, 64))
                norm_int = normalizar_intensidades(norm_res, "float32_0_1")
                suavizada = suavizado_gaussiano(norm_int, sigma=args.sigma)

                if tipo_s == "desconocido":
                    datos_desconocidos[part_desc]["total_intentos"] += 1
                    datos_desconocidos[part_desc]["X"].append(suavizada)
                    datos_desconocidos[part_desc]["y"].append(r.persona)
                    datos_desconocidos[part_desc]["ids"].append(r.identificador)
                    datos_desconocidos[part_desc]["poses"].append(r.pose)
                elif r.particion in datos_prep:
                    intentos_autorizados[r.particion]["total_intentos"] += 1
                    intentos_autorizados[r.particion]["exitos_deteccion"] += 1
                    datos_prep[r.particion]["X"].append(suavizada)
                    datos_prep[r.particion]["y"].append(r.persona)
                    datos_prep[r.particion]["ids"].append(r.identificador)
                    datos_prep[r.particion]["poses"].append(r.pose)

                trazabilidad_muestras[r.identificador] = {
                    "identificador": r.identificador,
                    "persona_real": r.persona,
                    "persona_predicha": "",
                    "correcta": "",
                    "probabilidad": 0.0,
                    "pose": r.pose,
                    "particion": r.particion,
                    "estado_deteccion": "UN_ROSTRO"
                }
            else:
                fallos_deteccion += 1
                stats_deteccion[pose]["fallos"] += 1
                if res_det.estado == EstadoDeteccion.SIN_ROSTRO:
                    stats_deteccion[pose]["sin_rostro"] += 1
                elif res_det.estado == EstadoDeteccion.VARIOS_ROSTROS:
                    stats_deteccion[pose]["varios_rostros"] += 1
                elif res_det.estado == EstadoDeteccion.UN_ROSTRO and not localizacion_valida:
                    # Detectó caja pero sobre ropa o pelo (falso positivo de localización)
                    stats_deteccion[pose]["sin_rostro"] += 1

                if tipo_s == "desconocido":
                    datos_desconocidos[part_desc]["total_intentos"] += 1
                    datos_desconocidos[part_desc]["fallos_deteccion"] += 1
                elif r.particion in intentos_autorizados:
                    intentos_autorizados[r.particion]["total_intentos"] += 1
                    intentos_autorizados[r.particion]["fallos_deteccion"] += 1

                trazabilidad_muestras[r.identificador] = {
                    "identificador": r.identificador,
                    "persona_real": r.persona,
                    "persona_predicha": "N/A",
                    "correcta": "FALLO_DETECCION",
                    "probabilidad": 0.0,
                    "pose": r.pose,
                    "particion": r.particion,
                    "estado_deteccion": res_det.estado.value if localizacion_valida else "FALSO_POSITIVO_ROPA_PELO"
                }

        print(f"  - Total imágenes procesadas: {total_procesadas}")
        print(f"  - Rostros detectados (UN_ROSTRO): {detecciones_exitosas} ({(detecciones_exitosas/total_procesadas*100 if total_procesadas>0 else 0):.1f}%)")
        print(f"  - Fallos de detección: {fallos_deteccion} ({(fallos_deteccion/total_procesadas*100 if total_procesadas>0 else 0):.1f}%)")
        print("  - Registro de detección por pose:")
        for p in poses_totales:
            st = stats_deteccion[p]
            t = st["total"]
            f = st["fallos"]
            tasa = (f / t * 100) if t > 0 else 0.0
            print(f"    * Pose '{p:10s}': Total={t:2d} | Éxitos={st['exitos']:2d} | Fallos={f:2d} ({tasa:.1f}%) [Sin={st['sin_rostro']}, Varios={st['varios_rostros']}]")

        if len(datos_prep["entrenamiento"]["X"]) == 0:
            print("Error: No hay rostros detectados en la partición de entrenamiento. No se puede entrenar KNN.")
            return 1

        # REGLA ESTRICTA (Corrección 1):
        # Las clases autorizadas se definen ÚNICAMENTE con los autorizados de entrenamiento.
        personas_autorizadas = sorted(list(set(datos_prep["entrenamiento"]["y"])))
        persona_to_id = {p: i for i, p in enumerate(personas_autorizadas)}
        id_to_persona = {i: p for i, p in enumerate(personas_autorizadas)}

        # Separar cualquier persona no autorizada en validación o prueba hacia datos_desconocidos
        for part in ["validacion", "prueba"]:
            idx_aut = []
            for i, p in enumerate(datos_prep[part]["y"]):
                if p in persona_to_id:
                    idx_aut.append(i)
                else:
                    datos_desconocidos[part]["X"].append(datos_prep[part]["X"][i])
                    datos_desconocidos[part]["y"].append(p)
                    datos_desconocidos[part]["ids"].append(datos_prep[part]["ids"][i])
                    datos_desconocidos[part]["poses"].append(datos_prep[part]["poses"][i])
                    datos_desconocidos[part]["total_intentos"] += 1
                    intentos_autorizados[part]["total_intentos"] -= 1
                    intentos_autorizados[part]["exitos_deteccion"] -= 1

            if len(idx_aut) < len(datos_prep[part]["y"]):
                datos_prep[part]["X"] = [datos_prep[part]["X"][i] for i in idx_aut]
                datos_prep[part]["y"] = [datos_prep[part]["y"][i] for i in idx_aut]
                datos_prep[part]["ids"] = [datos_prep[part]["ids"][i] for i in idx_aut]
                datos_prep[part]["poses"] = [datos_prep[part]["poses"][i] for i in idx_aut]

        # Fase 3: Extracción de características
        grid = (args.grid_y, args.grid_x)
        extractor = ExtractorHistogramasRegionales(cuadricula=grid, n_bins=args.bins, rango_intensidad=(0.0, 1.0))
        
        if modelo_nombre != "cnn":
            print(f"\n[3/4] Extrayendo histogramas de intensidad por regiones ({grid[0]}x{grid[1]}, bins={args.bins})...")
            X_train_in, nombres_feat = extractor.extraer_lote(np.array(datos_prep["entrenamiento"]["X"]))
            X_val_in = extractor.extraer_lote(np.array(datos_prep["validacion"]["X"]))[0] if len(datos_prep["validacion"]["X"]) > 0 else np.empty((0, extractor.dimension_total))
            X_test_in = extractor.extraer_lote(np.array(datos_prep["prueba"]["X"]))[0] if len(datos_prep["prueba"]["X"]) > 0 else np.empty((0, extractor.dimension_total))
            print(f"  - Dimensión de características: {extractor.dimension_total}")
        else:
            print(f"\n[3/4] Preparando tensores espaciales 2D (64x64 gris) para CNN...")
            X_train_in = np.array(datos_prep["entrenamiento"]["X"], dtype=np.float32)
            X_val_in = np.array(datos_prep["validacion"]["X"], dtype=np.float32) if len(datos_prep["validacion"]["X"]) > 0 else np.empty((0, 64, 64), dtype=np.float32)
            X_test_in = np.array(datos_prep["prueba"]["X"], dtype=np.float32) if len(datos_prep["prueba"]["X"]) > 0 else np.empty((0, 64, 64), dtype=np.float32)

        tot_desc_val = len(datos_desconocidos["validacion"]["X"])
        tot_desc_test = len(datos_desconocidos["prueba"]["X"])
        print(f"  - Muestras Entrenamiento: {len(X_train_in)}")
        print(f"  - Muestras Validación:    {len(X_val_in)}")
        print(f"  - Muestras Prueba:        {len(X_test_in)}")
        print(f"  - Muestras Desconocidos:  {tot_desc_val + tot_desc_test} (Val: {tot_desc_val}, Prueba: {tot_desc_test})")

        # Fase 4: Clasificación con el modelo seleccionado (knn, arbol, svm, ann, cnn)
        print(f"\n[4/4] Entrenando y evaluando clasificador {modelo_nombre.upper()} en Dataset Propio...")
        print(f"  - Personas autorizadas de entrenamiento: {personas_autorizadas}")

        y_train_int = np.array([persona_to_id[p] for p in datos_prep["entrenamiento"]["y"]], dtype=np.int32)
        y_val_int = np.array([persona_to_id[p] for p in datos_prep["validacion"]["y"]], dtype=np.int32) if len(datos_prep["validacion"]["y"]) > 0 else np.array([], dtype=np.int32)
        y_test_int = np.array([persona_to_id[p] for p in datos_prep["prueba"]["y"]], dtype=np.int32) if len(datos_prep["prueba"]["y"]) > 0 else np.array([], dtype=np.int32)

        configuraciones_validadas = None
        scores_o_probs = None
        historia_train = None

        if modelo_nombre == "knn":
            k_knn = min(args.k, len(y_train_int))
            if k_knn < args.k:
                print(f"  * Advertencia: K ajustado a {k_knn} por cantidad de muestras de entrenamiento.")
            modelo_instancia = ClasificadorKNN(k=k_knn, metrica=args.metrica, ponderacion=args.ponderacion)
            resumen = modelo_instancia.ejecutar_experimento_completo(
                X_train=X_train_in,
                y_train=y_train_int,
                ids_train=datos_prep["entrenamiento"]["ids"],
                X_val=X_val_in,
                y_val=y_val_int,
                ids_val=datos_prep["validacion"]["ids"],
                X_test=X_test_in,
                y_test=y_test_int,
                ids_test=datos_prep["prueba"]["ids"],
                nombre_dataset=dataset_nombre
            )
            modelo_instancia.guardar(dir_salida / "modelo.joblib")

        elif modelo_nombre == "arbol":
            modelo_instancia, configuraciones_validadas, mejor_cfg = ClasificadorArbol.seleccionar_y_entrenar(
                X_train=X_train_in,
                y_train=y_train_int,
                X_val=X_val_in,
                y_val=y_val_int,
                semilla=args.semilla
            )
            resumen = modelo_instancia.ejecutar_experimento_completo(
                X_train=X_train_in,
                y_train=y_train_int,
                ids_train=datos_prep["entrenamiento"]["ids"],
                X_val=X_val_in,
                y_val=y_val_int,
                ids_val=datos_prep["validacion"]["ids"],
                X_test=X_test_in,
                y_test=y_test_int,
                ids_test=datos_prep["prueba"]["ids"],
                nombre_dataset=dataset_nombre
            )
            modelo_instancia.guardar(dir_salida / "modelo.joblib")

        elif modelo_nombre == "svm":
            modelo_instancia, configuraciones_validadas, mejor_cfg = ClasificadorSVM.seleccionar_y_entrenar(
                X_train=X_train_in,
                y_train=y_train_int,
                X_val=X_val_in,
                y_val=y_val_int,
                semilla=args.semilla
            )
            resumen = modelo_instancia.ejecutar_experimento_completo(
                X_train=X_train_in,
                y_train=y_train_int,
                ids_train=datos_prep["entrenamiento"]["ids"],
                X_val=X_val_in,
                y_val=y_val_int,
                ids_val=datos_prep["validacion"]["ids"],
                X_test=X_test_in,
                y_test=y_test_int,
                ids_test=datos_prep["prueba"]["ids"],
                nombre_dataset=dataset_nombre
            )
            modelo_instancia.guardar(dir_salida / "modelo.joblib")

        elif modelo_nombre == "ann":
            modelo_instancia = ClasificadorANN(
                n_entrada=extractor.dimension_total,
                n_clases=len(personas_autorizadas),
                epocas=120,
                semilla=args.semilla
            )
            resumen = modelo_instancia.ejecutar_experimento_completo(
                X_train=X_train_in,
                y_train=y_train_int,
                ids_train=datos_prep["entrenamiento"]["ids"],
                X_val=X_val_in,
                y_val=y_val_int,
                ids_val=datos_prep["validacion"]["ids"],
                X_test=X_test_in,
                y_test=y_test_int,
                ids_test=datos_prep["prueba"]["ids"],
                nombre_dataset=dataset_nombre
            )
            modelo_instancia.guardar(dir_salida / "modelo.pt")
            historia_train = modelo_instancia.historia_entrenamiento

        elif modelo_nombre == "cnn":
            modelo_instancia = ClasificadorCNN(
                n_clases=len(personas_autorizadas),
                epocas=80,
                semilla=args.semilla
            )
            resumen = modelo_instancia.ejecutar_experimento_completo(
                X_train=X_train_in,
                y_train=y_train_int,
                ids_train=datos_prep["entrenamiento"]["ids"],
                X_val=X_val_in,
                y_val=y_val_int,
                ids_val=datos_prep["validacion"]["ids"],
                X_test=X_test_in,
                y_test=y_test_int,
                ids_test=datos_prep["prueba"]["ids"],
                nombre_dataset=dataset_nombre
            )
            modelo_instancia.guardar(dir_salida / "modelo.pt")
            historia_train = modelo_instancia.historia_entrenamiento

        # Guardar mapeo de clases para inferencia y web
        mapeo_clases = {
            "persona_to_id": persona_to_id,
            "id_to_persona": {str(k): v for k, v in id_to_persona.items()},
            "personas_autorizadas": personas_autorizadas,
            "umbral_rechazo_recomendado": 0.65
        }
        with open(dir_salida / "clases_mapeo.json", "w", encoding="utf-8") as f:
            json.dump(mapeo_clases, f, indent=2)

        resumen.configuracion.update({
            "suavizado_sigma": args.sigma,
            "semilla_aleatoria": args.semilla,
            "estadisticas_deteccion": stats_deteccion,
            "personas": personas_autorizadas,
            "umbral_rechazo_desconocido": 0.65
        })
        if modelo_nombre != "cnn":
            resumen.configuracion.update({
                "cuadricula_regiones": [args.grid_y, args.grid_x],
                "n_bins_histograma": args.bins,
                "dimension_caracteristicas": extractor.dimension_total
            })

        # Completar trazabilidad con predicciones del modelo seleccionado
        y_test_pred_arr = np.array([], dtype=np.int32)
        y_test_probs_arr = None

        for part_nombre, X_in_p, y_arr, ids_list in [
            ("entrenamiento", X_train_in, y_train_int, datos_prep["entrenamiento"]["ids"]),
            ("validacion", X_val_in, y_val_int, datos_prep["validacion"]["ids"]),
            ("prueba", X_test_in, y_test_int, datos_prep["prueba"]["ids"])
        ]:
            if len(X_in_p) > 0:
                y_preds = modelo_instancia.predecir(X_in_p)
                probs = modelo_instancia.predecir_probabilidades(X_in_p)
                if part_nombre == "prueba":
                    y_test_pred_arr = y_preds
                    y_test_probs_arr = probs

                for mid, pred_id, prob_arr, y_real in zip(ids_list, y_preds, probs, y_arr):
                    p_pred = id_to_persona.get(int(pred_id), str(pred_id))
                    p_real = id_to_persona.get(int(y_real), str(y_real))
                    prob_max = round(float(np.max(prob_arr)), 4)
                    trazabilidad_muestras[mid]["persona_predicha"] = p_pred
                    trazabilidad_muestras[mid]["probabilidad"] = prob_max
                    trazabilidad_muestras[mid]["correcta"] = (p_pred == p_real)

        # Guardar artefactos completos estandarizados
        ids_map = {
            "entrenamiento": datos_prep["entrenamiento"]["ids"],
            "validacion": datos_prep["validacion"]["ids"],
            "prueba": datos_prep["prueba"]["ids"]
        }
        guardar_artefactos_completos(
            resumen=resumen,
            dir_salida=dir_salida,
            ids_map=ids_map,
            y_test_pred=y_test_pred_arr,
            ids_test=datos_prep["prueba"]["ids"],
            y_test_real=y_test_int,
            configuraciones_validadas=configuraciones_validadas,
            scores_o_probs=y_test_probs_arr,
            historia_entrenamiento=historia_train
        )

        with open(dir_salida / "semilla.txt", "w", encoding="utf-8") as f:
            f.write(str(args.semilla) + "\n")

        # Guardar predicciones.csv
        ruta_pred = dir_salida / "predicciones.csv"
        with open(ruta_pred, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "identificador", "persona_real", "persona_predicha", "correcta",
                "probabilidad", "pose", "particion", "estado_deteccion"
            ])
            writer.writeheader()
            for r in manifiesto.registros:
                info = trazabilidad_muestras.get(r.identificador, {})
                writer.writerow(info)

        # Guardar fallos_deteccion_pose.csv
        ruta_fallos = dir_salida / "fallos_deteccion_pose.csv"
        with open(ruta_fallos, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pose", "total", "exitos", "fallos", "tasa_fallo_pct", "sin_rostro", "varios_rostros"
            ])
            writer.writeheader()
            for p in poses_totales:
                st = stats_deteccion[p]
                tot = st["total"]
                fal = st["fallos"]
                tasa = round(fal / tot * 100, 2) if tot > 0 else 0.0
                writer.writerow({
                    "pose": p,
                    "total": tot,
                    "exitos": st["exitos"],
                    "fallos": fal,
                    "tasa_fallo_pct": tasa,
                    "sin_rostro": st["sin_rostro"],
                    "varios_rostros": st["varios_rostros"]
                })

        # Guardar metricas_por_pose.csv (para conjunto de prueba)
        ruta_metricas_pose = dir_salida / "metricas_por_pose.csv"
        with open(ruta_metricas_pose, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pose", "total_prueba", "rostros_detectados", "aciertos_knn",
                "exactitud_knn_pct", "exactitud_modelo_pct", "exactitud_end_to_end_pct"
            ])
            writer.writeheader()
            for p in poses_totales:
                items_p = [v for v in trazabilidad_muestras.values() if v["particion"] == "prueba" and v["pose"] == p]
                tot_p = len(items_p)
                det_ok = sum(1 for v in items_p if v["estado_deteccion"] == "UN_ROSTRO")
                aciertos = sum(1 for v in items_p if v["correcta"] is True)
                acc_knn = round(aciertos / det_ok * 100, 2) if det_ok > 0 else 0.0
                acc_e2e = round(aciertos / tot_p * 100, 2) if tot_p > 0 else 0.0
                writer.writerow({
                    "pose": p,
                    "total_prueba": tot_p,
                    "rostros_detectados": det_ok,
                    "aciertos_knn": aciertos,
                    "exactitud_knn_pct": acc_knn,
                    "exactitud_modelo_pct": acc_knn,
                    "exactitud_end_to_end_pct": acc_e2e
                })

        # =========================================================================
        # EVALUACIÓN DE ACCESO FÍSICO Y REGLA DE RECHAZO EN DOS ETAPAS ESTRICTAS:
        # Etapa 1: Calibración en Validación (Sintonizar umbral y congelar modelo y umbral)
        # Etapa 2: Prueba Final Reservada (Medición ciega con modelo y umbral congelados)
        # =========================================================================
        umbrales_eval = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
        
        # Cargar imágenes de control ajenas de Olivetti
        # Se dividen estrictamente: 10 para Validación (calibración) y 10 para Prueba (ciega)
        olivetti_extra = cargar_olivetti(semilla=args.semilla)
        imgs_extranas_val = list(olivetti_extra.X_val[:10])
        imgs_extranas_test = list(olivetti_extra.X_test[:10])

        # Muestras desconocidas para validación y prueba
        muestras_desc_val = list(imgs_extranas_val) + list(datos_desconocidos["validacion"]["X"])
        muestras_desc_test = list(imgs_extranas_test) + list(datos_desconocidos["prueba"]["X"])

        total_intentos_desc_val = len(imgs_extranas_val) + datos_desconocidos["validacion"]["total_intentos"]
        fallos_haar_desc_val = datos_desconocidos["validacion"]["fallos_deteccion"]

        total_intentos_desc_test = len(imgs_extranas_test) + datos_desconocidos["prueba"]["total_intentos"]
        fallos_haar_desc_test = datos_desconocidos["prueba"]["fallos_deteccion"]

        # Predicciones de probabilidad
        if modelo_nombre != "cnn":
            feats_desc_val = extractor.extraer_lote(muestras_desc_val)[0] if len(muestras_desc_val) > 0 else np.empty((0, extractor.dimension_total))
            feats_desc_test = extractor.extraer_lote(muestras_desc_test)[0] if len(muestras_desc_test) > 0 else np.empty((0, extractor.dimension_total))
            probs_desc_val = modelo_instancia.predecir_probabilidades(feats_desc_val) if len(feats_desc_val) > 0 else np.empty((0, len(personas_autorizadas)))
            probs_desc_test = modelo_instancia.predecir_probabilidades(feats_desc_test) if len(feats_desc_test) > 0 else np.empty((0, len(personas_autorizadas)))
            probs_val_aut = modelo_instancia.predecir_probabilidades(X_val_in) if len(X_val_in) > 0 else np.empty((0, len(personas_autorizadas)))
            probs_test_aut = modelo_instancia.predecir_probabilidades(X_test_in) if len(X_test_in) > 0 else np.empty((0, len(personas_autorizadas)))
        else:
            probs_desc_val = modelo_instancia.predecir_probabilidades(np.array(muestras_desc_val, dtype=np.float32)) if len(muestras_desc_val) > 0 else np.empty((0, len(personas_autorizadas)))
            probs_desc_test = modelo_instancia.predecir_probabilidades(np.array(muestras_desc_test, dtype=np.float32)) if len(muestras_desc_test) > 0 else np.empty((0, len(personas_autorizadas)))
            probs_val_aut = modelo_instancia.predecir_probabilidades(X_val_in) if len(X_val_in) > 0 else np.empty((0, len(personas_autorizadas)))
            probs_test_aut = modelo_instancia.predecir_probabilidades(X_test_in) if len(X_test_in) > 0 else np.empty((0, len(personas_autorizadas)))

        max_probs_desc_val = np.max(probs_desc_val, axis=1) if len(probs_desc_val) > 0 else np.array([])
        max_probs_desc_test = np.max(probs_desc_test, axis=1) if len(probs_desc_test) > 0 else np.array([])
        max_probs_val_aut = np.max(probs_val_aut, axis=1) if len(probs_val_aut) > 0 else np.array([])
        preds_val_aut = np.argmax(probs_val_aut, axis=1) if len(probs_val_aut) > 0 else np.array([])
        max_probs_test_aut = np.max(probs_test_aut, axis=1) if len(probs_test_aut) > 0 else np.array([])
        preds_test_aut = np.argmax(probs_test_aut, axis=1) if len(probs_test_aut) > 0 else np.array([])

        # --- ETAPA 1: CALIBRACIÓN EN VALIDACIÓN ---
        curva_calibracion_val = []
        tot_aut_val = intentos_autorizados["validacion"]["total_intentos"]
        fallo_haar_aut_val = intentos_autorizados["validacion"]["fallos_deteccion"]

        for u in umbrales_eval:
            # Desconocidos en validación (Control físico de puerta)
            intrusiones_val_u = int(np.sum(max_probs_desc_val >= u)) if len(max_probs_desc_val) > 0 else 0
            rech_seguros_val_u = total_intentos_desc_val - intrusiones_val_u
            far_val_u = (intrusiones_val_u / total_intentos_desc_val * 100.0) if total_intentos_desc_val > 0 else 0.0

            # Autorizados en validación:
            # A) Control Físico de Puerta:
            # La puerta permanece cerrada si Haar no detecta rostro o si la confianza es < u.
            rech_umbral_aut_val_u = int(np.sum(max_probs_val_aut < u)) if len(max_probs_val_aut) > 0 else 0
            bloqueos_puerta_val_u = fallo_haar_aut_val + rech_umbral_aut_val_u
            frr_puerta_val_u = (bloqueos_puerta_val_u / tot_aut_val * 100.0) if tot_aut_val > 0 else 0.0
            
            # La puerta SE ABRE físicamente cuando un rostro detectado alcanza confianza >= u
            # (incluso si la red confunde a una persona autorizada con otra autorizada, la web abre la puerta)
            aperturas_puerta_val_u = tot_aut_val - bloqueos_puerta_val_u
            tar_puerta_val_u = (aperturas_puerta_val_u / tot_aut_val * 100.0) if tot_aut_val > 0 else 0.0

            # B) Identificación Biométrica (Aciertos vs Confusiones entre autorizados):
            aciertos_id_val_u = int(np.sum((max_probs_val_aut >= u) & (preds_val_aut == y_val_int))) if len(preds_val_aut) > 0 else 0
            confusiones_id_val_u = int(np.sum((max_probs_val_aut >= u) & (preds_val_aut != y_val_int))) if len(preds_val_aut) > 0 else 0
            acc_id_val_u = (aciertos_id_val_u / tot_aut_val * 100.0) if tot_aut_val > 0 else 0.0
            conf_id_val_u = (confusiones_id_val_u / tot_aut_val * 100.0) if tot_aut_val > 0 else 0.0

            # Costo operacional de acceso físico y distancia de error al punto ideal (0, 0)
            costo_val_u = far_val_u + frr_puerta_val_u
            dist_val_u = np.sqrt(far_val_u**2 + frr_puerta_val_u**2)

            curva_calibracion_val.append({
                "umbral": u,
                "far_val_pct": round(far_val_u, 2),
                "frr_puerta_pct": round(frr_puerta_val_u, 2),
                "frr_val_pct": round(frr_puerta_val_u, 2),  # Compatibilidad
                "tar_puerta_pct": round(tar_puerta_val_u, 2),
                "tar_val_pct": round(tar_puerta_val_u, 2),  # Compatibilidad
                "intrusiones_val": intrusiones_val_u,
                "rechazos_seguros_val": rech_seguros_val_u,
                "aperturas_puerta_val": aperturas_puerta_val_u,
                "bloqueos_puerta_val": bloqueos_puerta_val_u,
                "rechazos_baja_confianza_val": rech_umbral_aut_val_u,
                "aciertos_identidad_val": aciertos_id_val_u,
                "exactitud_identidad_pct": round(acc_id_val_u, 2),
                "confusiones_identidad_val": confusiones_id_val_u,
                "confusion_identidad_pct": round(conf_id_val_u, 2),
                "costo_operacional_val": round(costo_val_u, 2),
                "distancia_error_val": round(dist_val_u, 2)
            })

        # REGLA OBJETIVA DE SELECCIÓN DE UMBRAL EN VALIDACIÓN:
        # Criterio de Acceso Físico con Garantía de Usabilidad:
        # 1. Una puerta de acceso física requiere que el bloqueo a autorizados sea tolerable
        #    (FRR_puerta <= 30.0%, garantizando al menos un 70% de aperturas a autorizados).
        # 2. Entre los umbrales que cumplen la cota de usabilidad (FRR_puerta <= 30%),
        #    se elige el de MENOR tasa de falsa aceptación (FAR_val) para maximizar la seguridad ante intrusos.
        # 3. Desempate formal:
        #    a) Menor tasa de bloqueo de puerta (FRR_puerta_val).
        #    b) Mayor tasa de aciertos de identidad biométrica (aciertos_identidad_val).
        #    c) Mayor umbral u (política conservadora).
        # Si ningún umbral cumple la cota (FRR <= 30%), se elige el que minimiza la distancia euclidiana al punto ideal (0,0).
        candidatos_viables = [c for c in curva_calibracion_val if c["frr_puerta_pct"] <= 30.0]
        if not candidatos_viables:
            candidatos_viables = sorted(curva_calibracion_val, key=lambda c: c["distancia_error_val"])
        else:
            candidatos_viables.sort(
                key=lambda c: (
                    c["far_val_pct"],
                    c["frr_puerta_pct"],
                    -c["aciertos_identidad_val"],
                    -c["umbral"]
                )
            )

        calib_optima = candidatos_viables[0]
        umbral_congelado = calib_optima["umbral"]

        res_calibracion = {
            "etapa": "calibracion_validacion",
            "regla_seleccion_umbral": "Minimizar FAR_val sujeto a cota de usabilidad (FRR_puerta <= 30%), desempatando por menor FRR_puerta, mayor exactitud de identidad y mayor umbral.",
            "umbral_congelado": umbral_congelado,
            "umbral_recomendado": umbral_congelado,
            "umbral": umbral_congelado,
            "validacion": {
                # Control de Acceso Físico - Desconocidos
                "total_intentos_desconocidos": total_intentos_desc_val,
                "desconocidos_control_olivetti": len(imgs_extranas_val),
                "desconocidos_manifiesto": len(datos_desconocidos["validacion"]["X"]),
                "fallos_deteccion_desconocidos_haar": fallos_haar_desc_val,
                "puerta_cerrada_desconocidos_segura": calib_optima["rechazos_seguros_val"],
                "rechazados_correctamente": calib_optima["rechazos_seguros_val"],
                "puerta_abierta_desconocidos_intrusiones": calib_optima["intrusiones_val"],
                "falsos_aceptados": calib_optima["intrusiones_val"],
                "far_pct": calib_optima["far_val_pct"],

                # Control de Acceso Físico - Autorizados
                "total_intentos_autorizados": tot_aut_val,
                "fallos_deteccion_autorizados_haar": fallo_haar_aut_val,
                "rechazados_por_baja_confianza": calib_optima["rechazos_baja_confianza_val"],
                "puerta_cerrada_autorizados_bloqueos": calib_optima["bloqueos_puerta_val"],
                "falsos_rechazos_autorizados": calib_optima["bloqueos_puerta_val"],
                "frr_puerta_pct": calib_optima["frr_puerta_pct"],
                "frr_pct": calib_optima["frr_puerta_pct"],
                "puerta_abierta_autorizados": calib_optima["aperturas_puerta_val"],
                "tar_puerta_pct": calib_optima["tar_puerta_pct"],
                "tar_pct": calib_optima["tar_puerta_pct"],

                # Identificación Biométrica de Identidad - Autorizados
                "identificaciones_correctas": calib_optima["aciertos_identidad_val"],
                "accesos_correctos_autorizados": calib_optima["aciertos_identidad_val"],
                "exactitud_identidad_pct": calib_optima["exactitud_identidad_pct"],
                "confusiones_entre_autorizados": calib_optima["confusiones_identidad_val"],
                "confusion_identidad_pct": calib_optima["confusion_identidad_pct"]
            },
            "curva_calibracion_validacion": curva_calibracion_val
        }
        with open(dir_salida / "calibracion_umbral.json", "w", encoding="utf-8") as f:
            json.dump(res_calibracion, f, indent=2)

        # --- ETAPA 2: PRUEBA FINAL RESERVADA (MEDICIÓN CIEGA CON MODELO Y UMBRAL CONGELADOS) ---
        # A) Desconocidos en conjunto de prueba
        intrusiones_test_def = int(np.sum(max_probs_desc_test >= umbral_congelado)) if len(max_probs_desc_test) > 0 else 0
        rech_test_umbral = int(np.sum(max_probs_desc_test < umbral_congelado)) if len(max_probs_desc_test) > 0 else 0
        rech_test_total = fallos_haar_desc_test + rech_test_umbral
        far_test_e2e = (intrusiones_test_def / total_intentos_desc_test * 100.0) if total_intentos_desc_test > 0 else 0.0
        trr_test_e2e = (rech_test_total / total_intentos_desc_test * 100.0) if total_intentos_desc_test > 0 else 100.0

        # B) Autorizados en conjunto de prueba
        tot_aut_test = intentos_autorizados["prueba"]["total_intentos"]
        fallo_haar_aut_test = intentos_autorizados["prueba"]["fallos_deteccion"]
        rech_u_aut_test = int(np.sum(max_probs_test_aut < umbral_congelado)) if len(max_probs_test_aut) > 0 else 0
        bloqueados_puerta_test = fallo_haar_aut_test + rech_u_aut_test
        aperturas_puerta_test = tot_aut_test - bloqueados_puerta_test
        frr_puerta_test_e2e = (bloqueados_puerta_test / tot_aut_test * 100.0) if tot_aut_test > 0 else 0.0
        tar_puerta_test_e2e = (aperturas_puerta_test / tot_aut_test * 100.0) if tot_aut_test > 0 else 0.0

        # C) Identificación Biométrica en prueba
        aciertos_id_test = int(np.sum((max_probs_test_aut >= umbral_congelado) & (preds_test_aut == y_test_int))) if len(preds_test_aut) > 0 else 0
        confusiones_id_test = int(np.sum((max_probs_test_aut >= umbral_congelado) & (preds_test_aut != y_test_int))) if len(preds_test_aut) > 0 else 0
        acc_id_test_e2e = (aciertos_id_test / tot_aut_test * 100.0) if tot_aut_test > 0 else 0.0
        conf_id_test_e2e = (confusiones_id_test / tot_aut_test * 100.0) if tot_aut_test > 0 else 0.0

        res_prueba_final = {
            "etapa": "prueba_final_reservada",
            "modelo_congelado": modelo_nombre,
            "umbral_congelado": umbral_congelado,
            "prueba": {
                # Control de Acceso Físico - Desconocidos
                "total_intentos_desconocidos": total_intentos_desc_test,
                "desconocidos_control_olivetti": len(imgs_extranas_test),
                "desconocidos_manifiesto": len(datos_desconocidos["prueba"]["X"]),
                "fallos_deteccion_desconocidos_haar": fallos_haar_desc_test,
                "rechazos_por_umbral_desconocidos": rech_test_umbral,
                "puerta_cerrada_desconocidos_segura": rech_test_total,
                "rechazos_seguros_totales": rech_test_total,
                "puerta_abierta_desconocidos_intrusiones": intrusiones_test_def,
                "falsos_aceptados_intrusiones": intrusiones_test_def,
                "far_end_to_end_pct": round(far_test_e2e, 2),
                "trr_end_to_end_pct": round(trr_test_e2e, 2),

                # Control de Acceso Físico - Autorizados
                "total_intentos_autorizados": tot_aut_test,
                "fallos_deteccion_autorizados_haar": fallo_haar_aut_test,
                "rechazados_por_baja_confianza_autorizados": rech_u_aut_test,
                "puerta_cerrada_autorizados_bloqueos": bloqueados_puerta_test,
                "total_autorizados_bloqueados": bloqueados_puerta_test,
                "frr_puerta_end_to_end_pct": round(frr_puerta_test_e2e, 2),
                "frr_end_to_end_pct": round(frr_puerta_test_e2e, 2),
                "puerta_abierta_autorizados": aperturas_puerta_test,
                "accesos_concedidos_totales": aperturas_puerta_test,
                "tar_puerta_end_to_end_pct": round(tar_puerta_test_e2e, 2),
                "tar_end_to_end_pct": round(tar_puerta_test_e2e, 2),

                # Identificación Biométrica de Identidad - Autorizados
                "identificaciones_correctas": aciertos_id_test,
                "accesos_correctos_concedidos": aciertos_id_test,
                "exactitud_identidad_end_to_end_pct": round(acc_id_test_e2e, 2),
                "confusiones_entre_autorizados": confusiones_id_test,
                "confusiones_identidad_autorizados": confusiones_id_test,
                "confusion_identidad_end_to_end_pct": round(conf_id_test_e2e, 2)
            }
        }
        with open(dir_salida / "prueba_final_acceso.json", "w", encoding="utf-8") as f:
            json.dump(res_prueba_final, f, indent=2)

        # Mantener evaluacion_rechazo.json para compatibilidad con interfaces previas
        res_rechazo = {
            "umbral": umbral_congelado,
            "total_desconocidos": total_intentos_desc_test,
            "desconocidos_control_olivetti": len(imgs_extranas_test),
            "desconocidos_manifiesto": len(datos_desconocidos["prueba"]["X"]),
            "rechazados_correctamente": rech_test_total,
            "falsos_aceptados": intrusiones_test_def,
            "far_pct": round(far_test_e2e, 2),
            "probabilidades_maximas": [round(float(p), 4) for p in max_probs_desc_test],
            "validacion": res_calibracion["validacion"],
            "prueba": res_prueba_final["prueba"],
            "curva_calibracion_validacion": curva_calibracion_val
        }
        with open(dir_salida / "evaluacion_rechazo.json", "w", encoding="utf-8") as f:
            json.dump(res_rechazo, f, indent=2)

        # Actualizar automáticamente la selección del mejor modelo en resultados/modelo_seleccionado.json
        try:
            from reconocimiento.clasificacion.seleccion import seleccionar_mejor_modelo_propio
            seleccionar_mejor_modelo_propio(dir_resultados=Path("resultados"))
        except Exception as e_sel:
            pass

        print("\n-------------------------------------------------------")
        print(f" REPORTE FINAL DE LA EJECUCIÓN ({modelo_nombre.upper()} - DATASET PROPIO)")
        print("-------------------------------------------------------")
        print(f" Detección Facial Global:         {detecciones_exitosas}/{total_procesadas} ({(detecciones_exitosas/total_procesadas*100 if total_procesadas>0 else 0):.1f}%)")
        print(f" Exactitud Entrenamiento:         {resumen.reporte_train.exactitud_porcentaje:.2f}% ({resumen.reporte_train.aciertos}/{resumen.reporte_train.total_muestras})")
        print(f" Exactitud Validación:            {resumen.reporte_val.exactitud_porcentaje:.2f}% ({resumen.reporte_val.aciertos}/{resumen.reporte_val.total_muestras})")
        print(f" Exactitud Prueba (Detectados):   {resumen.reporte_test.exactitud_porcentaje:.2f}% ({resumen.reporte_test.aciertos}/{resumen.reporte_test.total_muestras})")
        print(f" Brecha de generalización:        {resumen.brecha_generalizacion_puntos:.2f} puntos porcentuales")
        print(f" Calibración Validación (u={umbral_congelado}):")
        print(f"   * Seguridad Desconocidos:      FAR_val={calib_optima['far_val_pct']:.1f}% ({calib_optima['intrusiones_val']}/{total_intentos_desc_val} intrusiones)")
        print(f"   * Acceso Físico Autorizados:   Puerta Abierta={calib_optima['aperturas_puerta_val']}/{tot_aut_val} ({calib_optima['tar_puerta_pct']:.1f}%), Bloqueos={calib_optima['bloqueos_puerta_val']}/{tot_aut_val} (FRR_puerta={calib_optima['frr_puerta_pct']:.1f}%)")
        print(f"   * Identificación Biométrica:   Aciertos ID={calib_optima['aciertos_identidad_val']}/{tot_aut_val} ({calib_optima['exactitud_identidad_pct']:.1f}%), Confusiones ID={calib_optima['confusiones_identidad_val']}/{tot_aut_val} ({calib_optima['confusion_identidad_pct']:.1f}%)")
        print(f" Prueba Final Reservada (u={umbral_congelado}):")
        print(f"   * Seguridad Desconocidos:      FAR_e2e={far_test_e2e:.1f}% ({intrusiones_test_def}/{total_intentos_desc_test} intrusiones)")
        print(f"   * Acceso Físico Autorizados:   Puerta Abierta={aperturas_puerta_test}/{tot_aut_test} ({tar_puerta_test_e2e:.1f}%), Bloqueos={bloqueados_puerta_test}/{tot_aut_test} (FRR_puerta={frr_puerta_test_e2e:.1f}%)")
        print(f"   * Identificación Biométrica:   Aciertos ID={aciertos_id_test}/{tot_aut_test} ({acc_id_test_e2e:.1f}%), Confusiones ID={confusiones_id_test}/{tot_aut_test} ({conf_id_test_e2e:.1f}%)")
        print(f" Predicciones guardadas en:       {ruta_pred.resolve()}")
        print(f" Calibración guardada en:         {(dir_salida / 'calibracion_umbral.json').resolve()}")
        print(f" Prueba final de acceso en:       {(dir_salida / 'prueba_final_acceso.json').resolve()}")
        print("=======================================================\n")
        return 0


def comando_importar_olivetti(args: argparse.Namespace) -> int:
    """Descarga e importa el dataset Olivetti Faces."""
    dir_data = Path(args.data_dir)
    print(f"Importando Olivetti Faces en: {dir_data}...")
    dataset = cargar_olivetti(data_dir=dir_data, descargar=True)
    print(f"Importación exitosa:")
    print(f"  - Total muestras: {len(dataset.y_train) + len(dataset.y_val) + len(dataset.y_test)}")
    print(f"  - Sujetos: {dataset.n_clases}")
    print(f"  - Resolución por muestra: {dataset.dimensiones_originales}")
    return 0


def comando_registrar(args: argparse.Namespace) -> int:
    """Registra una captura propia con metadatos en el manifiesto."""
    if not args.consentimiento:
        print("Error: Se requiere consentimiento informado explícito (--consentimiento) para registrar la captura.")
        return 1

    ruta_manifiesto = Path(args.manifiesto)
    manifiesto = ManifiestoDataset(ruta_manifiesto)

    registro = RegistroCaptura(
        identificador=args.identificador,
        ruta=args.ruta,
        procedencia=args.procedencia,
        persona=args.persona,
        tipo_sujeto=args.tipo_sujeto,
        sesion=args.sesion,
        pose=args.pose,
        particion=args.particion,
        consentimiento=args.consentimiento,
        anotaciones=args.anotaciones or ""
    )

    try:
        manifiesto.agregar_registro(registro)
        # Validar aislamiento de sesiones
        es_valido_ses, errores_ses = manifiesto.validar_aislamiento_sesiones()
        if not es_valido_ses:
            print("ERROR CRÍTICO: La adición viola la regla de aislamiento de sesiones:")
            for e in errores_ses:
                print(f"  - {e}")
            return 1

        # Validar duplicados entre particiones
        es_valido_dup, errores_dup = manifiesto.validar_duplicados_entre_particiones()
        if not es_valido_dup:
            print("ERROR CRÍTICO: La adición viola la regla de capturas duplicadas entre particiones:")
            for e in errores_dup:
                print(f"  - {e}")
            return 1

        manifiesto.guardar()
        print(f"Captura registrada exitosamente en {ruta_manifiesto}: {registro.identificador}")
        return 0
    except Exception as e:
        print(f"Error al registrar captura: {e}")
        return 1


def comando_camaras(args: argparse.Namespace) -> int:
    """Sondea dispositivos de video disponibles mediante DirectShow."""
    print(f"\n=== SONDEANDO DISPOSITIVOS DE VIDEO (DirectShow, rango 0..{args.max_indices - 1}) ===")
    dispositivos = sondear_camaras(max_indices=args.max_indices)

    print(f"\n{'Índice':<8} | {'Backend':<22} | {'Estado':<14} | {'Resolución':<14} | {'FPS':<6}")
    print("-" * 72)
    camaras_disponibles = []
    for d in dispositivos:
        estado_str = "DISPONIBLE" if d["disponible"] else ("ABIERTO (sin frames)" if d["abierto"] else "NO DETECTADO")
        res_str = f"{d['ancho']}x{d['alto']}" if d["disponible"] else "N/A"
        fps_str = f"{d['fps_nominal']:.0f}" if d["disponible"] else "N/A"
        print(f"[{d['indice']}]      | {d['backend']:<22} | {estado_str:<14} | {res_str:<14} | {fps_str:<6}")
        if d["disponible"]:
            camaras_disponibles.append(d["indice"])

    print("-" * 72)
    if camaras_disponibles:
        print(f"\nCámaras listas para uso: índices {camaras_disponibles}")
        print(f"Sugerencia: use 'python -m reconocimiento vista-previa --indice {camaras_disponibles[0]}' para verificar.")
    else:
        print("\nNo se detectó ningún dispositivo con señal de video activa.")
    print()
    return 0


def comando_vista_previa(args: argparse.Namespace) -> int:
    """Inicia la vista previa en vivo con recuadros de detección y estados en pantalla."""
    try:
        ejecutar_vista_previa(indice=args.indice)
        return 0
    except Exception as e:
        print(f"\nError durante la vista previa en cámara índice {args.indice}: {e}")
        return 1


def comando_capturar(args: argparse.Namespace) -> int:
    """Ejecuta el modo de captura interactiva con guardado atómico."""
    if not args.consentimiento:
        print("Error: Se requiere consentimiento informado explícito (--consentimiento) para capturar.")
        return 1

    manifiesto_path = Path(args.manifiesto)
    carpeta_destino = Path(args.carpeta)
    if args.ensayo:
        if args.manifiesto == "data/propio/manifiesto.csv":
            manifiesto_path = Path("data/ensayos/manifiesto_ensayo.csv")
        if args.carpeta == "data/propio/imagenes":
            carpeta_destino = Path("data/ensayos/imagenes")

    try:
        capturas = ejecutar_modo_captura(
            indice=args.indice,
            persona=args.persona,
            sesion=args.sesion,
            pose=args.pose,
            particion=args.particion,
            consentimiento=args.consentimiento,
            ruta_manifiesto=manifiesto_path,
            carpeta_destino=carpeta_destino,
            tipo_sujeto=args.tipo_sujeto,
            num_fotos_objetivo=args.fotos,
            es_ensayo=args.ensayo
        )
        print(f"\nSe completaron {len(capturas)} capturas en {carpeta_destino}.")
        return 0
    except Exception as e:
        print(f"\nError en modo de captura: {e}")
        return 1


def comando_verificar_protocolo(args: argparse.Namespace) -> int:
    """Verifica el cumplimiento del protocolo formal 3/1/1 frente a sesiones piloto."""
    ruta_manifiesto = Path(args.manifiesto)
    if not ruta_manifiesto.exists():
        print(f"Error: Manifiesto no encontrado en {ruta_manifiesto}")
        return 1

    manifiesto = ManifiestoDataset(ruta_manifiesto)
    resultado = manifiesto.verificar_protocolo_completo()

    print("\n=======================================================")
    print(f" VERIFICACIÓN DE PROTOCOLO: {ruta_manifiesto}")
    print("=======================================================")
    print(f" Estado global del dataset:         {resultado['estado_dataset']}")
    print(f" Aislamiento de sesiones (3/1/1):   {'VÁLIDO' if resultado['aislamiento_sesiones_valido'] else 'VIOLADO'}")
    if not resultado['aislamiento_sesiones_valido']:
        for e in resultado['errores_aislamiento']:
            print(f"   - {e}")
    print(f" Control de duplicados partición:   {'VÁLIDO (sin fugas)' if resultado['sin_duplicados_entre_particiones'] else 'VIOLADO'}")
    if not resultado['sin_duplicados_entre_particiones']:
        for e in resultado['errores_duplicados']:
            print(f"   - {e}")

    print(f" Procedencia confirmada (cámara):  {'SÍ' if resultado['procedencia_camara_confirmada'] else 'NO (sintético/ensayo)'}")
    print(f" Procedencias detectadas:           {resultado['procedencias_detectadas']}")
    print(f" Total de capturas registradas:     {resultado['total_registros']}")
    print(f" Personas autorizadas evaluadas:    {resultado['personas_autorizadas_evaluadas']}")

    print("\n Desglose por persona autorizada:")
    for p, datos in resultado['desglose_por_persona'].items():
        cumple = "COMPLETO (100 fotos)" if datos['protocolo_persona_completo'] else "PARCIAL/INCOMPLETO"
        print(f"   * Sujeto '{p}': {cumple} ({datos['total_muestras']} fotos)")
        print(f"       Sesiones Train: {datos['sesiones_entrenamiento']} | Val: {datos['sesiones_validacion']} | Test: {datos['sesiones_prueba']}")
        print(f"       Poses ({len(datos['poses_registradas'])}/5): {datos['poses_registradas']}")

    print("=======================================================\n")
    return 0 if (resultado['aislamiento_sesiones_valido'] and resultado['sin_duplicados_entre_particiones']) else 1


def comando_web(args: argparse.Namespace) -> int:
    """Inicia la interfaz web local para adquisición de imágenes."""
    try:
        from reconocimiento.web import crear_app
    except ImportError as e:
        print(f"Error al importar dependencias web (Flask): {e}")
        print("Por favor instale Flask: pip install flask")
        return 1

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 5000)
    indice = getattr(args, "indice", 1)

    app = crear_app(indice_inicial=indice)

    url_local = f"http://{host}:{port}/"
    print("\n=======================================================")
    print(" SERVIDOR WEB DE ADQUISICIÓN FACIAL EN EJECUCIÓN")
    print("=======================================================")
    print(f" Dirección web local:   {url_local}")
    print(f" Cámara predeterminada: Índice {indice} (DirectShow)")
    print(" Presione [Ctrl + C] para detener el servidor y liberar la cámara.")
    print("=======================================================\n")

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
        return 0
    except KeyboardInterrupt:
        print("\nServidor web detenido por el usuario.")
        return 0
    except Exception as e:
        print(f"\nError en el servidor web: {e}")
        return 1


def comando_procesar_imagen(args: argparse.Namespace) -> int:
    """
    Ejecuta el pipeline de preprocesamiento completo sobre una imagen:
    detección de rostro, recorte seguro, normalización a 64x64,
    suavizado gaussiano y máscara binaria con 8-conectividad.
    Genera panel visual comparativo.
    """
    ruta_entrada = Path(args.entrada)
    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    if not ruta_entrada.exists():
        print(f"Error: Archivo de entrada no encontrado: {ruta_entrada}")
        return 1

    imagen_bgr = cv2.imread(str(ruta_entrada))
    if imagen_bgr is None:
        print(f"Error: No se pudo decodificar la imagen: {ruta_entrada}")
        return 1

    gris = a_escala_de_grises(imagen_bgr)
    detector = DetectorFacial()
    resultado_det = detector.detectar(gris, es_pre_recortada=args.pre_recortada)

    print(f"Estado de detección: {resultado_det.estado.value}")
    print(f"Mensaje: {resultado_det.mensaje}")
    print(f"Recuadro: {resultado_det.recuadro}")

    if resultado_det.estado != EstadoDeteccion.UN_ROSTRO:
        print("La imagen no contiene un rostro único detectable. Se detiene el procesamiento.")
        return 2

    recorte_original = resultado_det.rostro_recortado
    recorte_norm = normalizar_resolucion(recorte_original, (64, 64))
    recorte_suavizado = suavizado_gaussiano(recorte_norm, sigma=args.suavizado_sigma)
    mascara_bin, umbral = generar_mascara_binaria(recorte_suavizado, metodo=args.metodo_mascara)
    mascara_8vec, meta_comp = aislar_componente_mayor_8vecinos(mascara_bin)

    # Crear panel de 4 cuadrantes
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Imagen original con recuadro dibujado
    img_marcada = imagen_bgr.copy()
    if resultado_det.recuadro:
        x, y, w, h = resultado_det.recuadro
        cv2.rectangle(img_marcada, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_marcada_rgb = cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_marcada_rgb)
    axes[0].set_title(f"Original + BBox\n({img_marcada.shape[1]}x{img_marcada.shape[0]})")
    axes[0].axis("off")

    # 2. Recorte normalizado (64x64)
    axes[1].imshow(recorte_norm, cmap="gray")
    axes[1].set_title(f"Recorte Normalizado\n(64x64 píxeles)")
    axes[1].axis("off")

    # 3. Suavizado Gaussiano (sigma configurado)
    axes[2].imshow(recorte_suavizado, cmap="gray")
    axes[2].set_title(f"Suavizado Gaussiano\n(\\sigma={args.suavizado_sigma})")
    axes[2].axis("off")

    # 4. Máscara binaria con 8-conectividad
    axes[3].imshow(mascara_8vec, cmap="gray")
    axes[3].set_title(f"Máscara Otsu 8-vecinos\n(Umbral={umbral:.1f})")
    axes[3].axis("off")

    plt.tight_layout()
    ruta_panel = dir_salida / "panel_procesamiento.png"
    plt.savefig(ruta_panel, dpi=150)
    plt.close()

    # Guardar también recortes individuales
    cv2.imwrite(str(dir_salida / "recorte_64x64.png"), recorte_norm)
    cv2.imwrite(str(dir_salida / "mascara_8vecinos.png"), mascara_8vec)

    print(f"Panel visual guardado en: {ruta_panel}")
    return 0


def comando_demo_aliasing(args: argparse.Namespace) -> int:
    """
    Compara el submuestreo directo (con aliasing) frente al submuestreo con antialiasing
    tanto en un patrón sintético de franjas de alta frecuencia como en una muestra facial.
    Vinculado a Nyquist (0.25 ciclos/píxel para factor=2).
    """
    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    print("\nGenerando demostración comparativa de Aliasing vs Antialiasing...")
    factor = args.factor
    print(f"  Factor de reducción: {factor}x")
    print(f"  Nuevo límite de Nyquist: {0.5 / factor:.3f} ciclos/píxel original")

    # 1. Patrón sintético de franjas
    patron = generar_patron_sintetico_franjas(dimensiones=(256, 256))
    patron_directo = submuestrear(patron, factor=factor, aplicar_antialiasing=False)
    patron_antialiased = submuestrear(patron, factor=factor, aplicar_antialiasing=True)

    # 2. Cargar una muestra facial de Olivetti para comprobar en rostros reales
    dataset = cargar_olivetti()
    cara_orig = (dataset.X_train[0] * 255.0).astype(np.uint8)
    cara_directo = submuestrear(cara_orig, factor=factor, aplicar_antialiasing=False)
    cara_antialiased = submuestrear(cara_orig, factor=factor, aplicar_antialiasing=True)

    # Generar panel comparativo de franjas sintéticas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(patron, cmap="gray")
    axes[0].set_title(f"Patrón Original (256x256)\nFrecuencia espacial creciente")
    axes[0].axis("off")

    axes[1].imshow(patron_directo, cmap="gray")
    axes[1].set_title(f"Reducción directa {factor}x (128x128)\n¡ALÍASING / FALSOS PATRONES!")
    axes[1].axis("off")

    axes[2].imshow(patron_antialiased, cmap="gray")
    axes[2].set_title(f"Reducción con Antialiasing {factor}x (128x128)\nFiltro pasa-bajos gaussiano previo")
    axes[2].axis("off")

    plt.tight_layout()
    ruta_franjas = dir_salida / "comparacion_aliasing_franjas.png"
    plt.savefig(ruta_franjas, dpi=150)
    plt.close()

    # Generar panel comparativo en rostro
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cara_orig, cmap="gray")
    axes[0].set_title(f"Rostro Original (64x64)")
    axes[0].axis("off")

    axes[1].imshow(cara_directo, cmap="gray")
    axes[1].set_title(f"Rostro Reducción Directa {factor}x (32x32)\nArtefactos de pixelado en bordes")
    axes[1].axis("off")

    axes[2].imshow(cara_antialiased, cmap="gray")
    axes[2].set_title(f"Rostro con Antialiasing {factor}x (32x32)\nTransiciones tonales preservadas")
    axes[2].axis("off")

    plt.tight_layout()
    ruta_rostro = dir_salida / "comparacion_aliasing_rostro.png"
    plt.savefig(ruta_rostro, dpi=150)
    plt.close()

    print(f"Demostración completada:")
    print(f"  - Panel franjas sintéticas: {ruta_franjas}")
    print(f"  - Panel rostro real:       {ruta_rostro}")
    return 0


def comando_verificar_conectividad(args: argparse.Namespace) -> int:
    """Verifica formalmente la conectividad 8 vs 4 en máscara sintética."""
    resultado = verificar_conectividad_8_vs_4()
    print("\n=== VERIFICACIÓN DE CONECTIVIDAD: 8-VECINOS VS 4-VECINOS ===")
    print("Máscara sintética 3x3:")
    for fila in resultado["mascara_sintetica"]:
        print(f"  {fila}")
    print("\nResultados:")
    print(f"  - Conectividad 4: {resultado['conectividad_4']['num_componentes_primer_plano']} componentes de primer plano. ¿Mismo componente?: {resultado['conectividad_4']['comparten_componente']}")
    print(f"  - Conectividad 8: {resultado['conectividad_8']['num_componentes_primer_plano']} componentes de primer plano. ¿Mismo componente?: {resultado['conectividad_8']['comparten_componente']}")
    print(f"\nConclusión: {resultado['conclusion']}\n")
    return 0


def comando_test_controlado(args: argparse.Namespace) -> int:
    """
    Ejecuta pruebas con entradas controladas:
    1. Sin rostro -> SIN_ROSTRO
    2. Rostro normal centrado -> UN_ROSTRO
    3. Rostro en el borde -> recorte con padding seguro
    4. Varios rostros simulados -> VARIOS_ROSTROS
    Falla con código 1 si algún resultado es incorrecto.
    """
    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    print("\n=== EJECUTANDO PRUEBAS CONTROLADAS DE DETECCIÓN Y BORDES ===")

    detector = DetectorFacial()
    errores = []

    # Caso 1: Imagen sin rostro (ruido o fondo sintético)
    sin_rostro = np.full((200, 200), 128, dtype=np.uint8)
    res_sin = detector.detectar(sin_rostro)
    caso1_ok = (res_sin.estado == EstadoDeteccion.SIN_ROSTRO and res_sin.total_rostros_detectados == 0)
    print(f"1. Caso sin rostro: Estado={res_sin.estado.value} | Total={res_sin.total_rostros_detectados} | Correcto={caso1_ok}")
    if not caso1_ok:
        errores.append(f"Caso 1: se esperaba SIN_ROSTRO (total=0), pero se obtuvo {res_sin.estado.value} ({res_sin.total_rostros_detectados})")

    # Caso 2: Rostro centrado de prueba (usando una imagen de Olivetti escalada en lienzo)
    dataset = cargar_olivetti()
    cara_base = (dataset.X_train[0] * 255).astype(np.uint8)
    lienzo = np.full((300, 300), 200, dtype=np.uint8)
    lienzo[100:164, 100:164] = cara_base

    res_centro = detector.detectar(cara_base, es_pre_recortada=True)
    caso2_ok = (res_centro.estado == EstadoDeteccion.UN_ROSTRO and res_centro.rostro_recortado is not None)
    print(f"2. Caso rostro normal: Estado={res_centro.estado.value} | Recorte={res_centro.rostro_recortado.shape if res_centro.rostro_recortado is not None else None} | Correcto={caso2_ok}")
    if not caso2_ok:
        errores.append(f"Caso 2: se esperaba UN_ROSTRO con recorte válido, pero se obtuvo {res_centro.estado.value}")

    # Caso 3: Rostro en el borde (clipping en y=0, x=0 con padding seguro)
    recuadro_borde = (-10, -10, 60, 60)
    recorte_seguro = detector.recortar_con_margen_seguro(cara_base, recuadro_borde)
    caso3_ok = (recorte_seguro is not None and recorte_seguro.shape[0] > 0 and recorte_seguro.shape[0] == recorte_seguro.shape[1])
    print(f"3. Caso recuadro en borde: Recorte seguro generado={caso3_ok}, forma={recorte_seguro.shape if recorte_seguro is not None else None}")
    if not caso3_ok:
        errores.append("Caso 3: el recorte con margen seguro en borde no es válido o no conserva relación cuadrada.")

    # Caso 4: Varios rostros (simulados con dos recuadros)
    dos_recuadros = [(20, 20, 60, 60), (120, 20, 60, 60)]
    lienzo_dos = np.full((200, 250), 128, dtype=np.uint8)
    res_varios = detector.detectar(lienzo_dos, recuadros_simulados=dos_recuadros)
    caso4_ok = (res_varios.estado == EstadoDeteccion.VARIOS_ROSTROS and res_varios.total_rostros_detectados == 2)
    print(f"4. Caso varios rostros (dos recuadros simulados): Estado={res_varios.estado.value} | Total={res_varios.total_rostros_detectados} | Correcto={caso4_ok}")
    if not caso4_ok:
        errores.append(f"Caso 4: se esperaba VARIOS_ROSTROS con total=2, pero se obtuvo {res_varios.estado.value} ({res_varios.total_rostros_detectados})")

    if errores:
        print("\n[ERROR] Fallaron las siguientes pruebas controladas:")
        for err in errores:
            print(f"  - {err}")
        return 1

    print("\nTodas las pruebas controladas de lógica pasaron con éxito (0 fallos).\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reconocimiento",
        description="Sistema de Reconocimiento Facial e Inteligencia Artificial (Entrega 1)"
    )
    subparsers = parser.add_subparsers(dest="subcomando", help="Subcomando a ejecutar")

    # Subcomando 'ejecutar'
    parser_ej = subparsers.add_parser("ejecutar", help="Ejecutar el pipeline de un modelo")
    parser_ej.add_argument("--modelo", type=str, required=True, choices=["knn", "arbol", "ann", "svm", "cnn"], help="Identificador del algoritmo")
    parser_ej.add_argument("--dataset", type=str, required=True, choices=["olivetti", "propio"], help="Dataset a utilizar")
    parser_ej.add_argument("--manifiesto", type=str, default="data/propio/manifiesto.csv", help="Ruta al manifiesto CSV para dataset propio (default: data/propio/manifiesto.csv)")
    parser_ej.add_argument("--salida", type=str, default="resultados/knn", help="Directorio para guardar resultados")
    parser_ej.add_argument("--k", type=int, default=5, help="Número de vecinos para KNN (default: 5)")
    parser_ej.add_argument("--metrica", type=str, default="euclidean", help="Métrica de distancia (default: euclidean)")
    parser_ej.add_argument("--ponderacion", type=str, default="uniform", choices=["uniform", "distance"], help="Ponderación de votos")
    parser_ej.add_argument("--grid-y", type=int, default=4, help="Número de divisiones verticales de la cuadrícula (default: 4)")
    parser_ej.add_argument("--grid-x", type=int, default=4, help="Número de divisiones horizontales de la cuadrícula (default: 4)")
    parser_ej.add_argument("--bins", type=int, default=16, help="Número de bins por histograma regional (default: 16)")
    parser_ej.add_argument("--sigma", type=float, default=0.5, help="Desviación estándar de suavizado gaussiano previo (default: 0.5)")
    parser_ej.add_argument("--semilla", type=int, default=42, help="Semilla pseudoaleatoria (default: 42)")

    # Subcomando 'importar-olivetti'
    parser_imp = subparsers.add_parser("importar-olivetti", help="Descargar y verificar Olivetti Faces")
    parser_imp.add_argument("--data-dir", type=str, default="data/olivetti", help="Directorio donde almacenar el dataset")

    # Subcomando 'registrar'
    parser_reg = subparsers.add_parser("registrar", help="Registrar imagen propia en el manifiesto")
    parser_reg.add_argument("--manifiesto", type=str, default="data/propio/manifiesto.csv")
    parser_reg.add_argument("--identificador", type=str, required=True)
    parser_reg.add_argument("--ruta", type=str, required=True)
    parser_reg.add_argument("--procedencia", type=str, default="camara")
    parser_reg.add_argument("--persona", type=str, required=True)
    parser_reg.add_argument("--tipo-sujeto", type=str, default="autorizado", choices=["autorizado", "desconocido"])
    parser_reg.add_argument("--sesion", type=str, required=True)
    parser_reg.add_argument("--pose", type=str, required=True, choices=list(POSES_VALIDAS))
    parser_reg.add_argument("--particion", type=str, required=True, choices=list(PARTICIONES_VALIDAS))
    parser_reg.add_argument("--consentimiento", action="store_true", default=False, help="Consentimiento informado explícito del voluntario (obligatorio)")
    parser_reg.add_argument("--anotaciones", type=str, default="")

    # Subcomando 'procesar-imagen'
    parser_proc = subparsers.add_parser("procesar-imagen", help="Preprocesar una imagen individual y generar panel")
    parser_proc.add_argument("--entrada", type=str, required=True)
    parser_proc.add_argument("--salida", type=str, default="resultados/ejemplos")
    parser_proc.add_argument("--pre-recortada", action="store_true", default=False)
    parser_proc.add_argument("--suavizado-sigma", type=float, default=1.0)
    parser_proc.add_argument("--metodo-mascara", type=str, default="otsu")

    # Subcomando 'demo-aliasing'
    parser_alias = subparsers.add_parser("demo-aliasing", help="Demostración de aliasing vs antialiasing")
    parser_alias.add_argument("--salida", type=str, default="resultados/aliasing")
    parser_alias.add_argument("--factor", type=int, default=2)

    # Subcomando 'verificar-conectividad'
    subparsers.add_parser("verificar-conectividad", help="Verificar componentes conexos 8 vs 4 en máscara sintética")

    # Subcomando 'test-controlado'
    parser_test = subparsers.add_parser("test-controlado", help="Pruebas controladas de detector, bordes y lógica")
    parser_test.add_argument("--salida", type=str, default="resultados/controlados")

    # Subcomando 'camaras'
    parser_cam = subparsers.add_parser("camaras", help="Sondear dispositivos de video disponibles (DirectShow)")
    parser_cam.add_argument("--max-indices", type=int, default=5, help="Rango de índices a sondear (default: 5)")

    # Subcomando 'vista-previa'
    parser_prev = subparsers.add_parser("vista-previa", help="Vista previa en vivo con recuadro y estados del detector")
    parser_prev.add_argument("--indice", type=int, default=1, help="Índice de la cámara DirectShow (default: 1)")

    # Subcomando 'capturar'
    parser_cap = subparsers.add_parser("capturar", help="Modo de captura manual con metadatos y guardado atómico")
    parser_cap.add_argument("--indice", type=int, default=1, help="Índice de la cámara DirectShow (default: 1)")
    parser_cap.add_argument("--persona", type=str, required=True, help="Identificador del sujeto (ej. persona_1)")
    parser_cap.add_argument("--tipo-sujeto", type=str, default="autorizado", choices=["autorizado", "desconocido"])
    parser_cap.add_argument("--sesion", type=str, required=True, help="Sesión de captura (ej. sesion_01)")
    parser_cap.add_argument("--pose", type=str, required=True, choices=list(POSES_VALIDAS), help="Pose (perspectiva del sujeto)")
    parser_cap.add_argument("--particion", type=str, required=True, choices=list(PARTICIONES_VALIDAS), help="Partición (entrenamiento, validacion, prueba)")
    parser_cap.add_argument("--consentimiento", action="store_true", default=False, help="Consentimiento informado explícito (obligatorio)")
    parser_cap.add_argument("--manifiesto", type=str, default="data/propio/manifiesto.csv", help="Ruta al manifiesto")
    parser_cap.add_argument("--carpeta", type=str, default="data/propio/imagenes", help="Carpeta base de imágenes")
    parser_cap.add_argument("--fotos", type=int, default=1, help="Número de fotos a tomar en la sesión (default: 1)")
    parser_cap.add_argument("--ensayo", action="store_true", default=False, help="Guardar como ensayo técnico en data/ensayos/")

    # Subcomando 'verificar-protocolo'
    parser_prot = subparsers.add_parser("verificar-protocolo", help="Verificar protocolo 3/1/1 frente a sesiones piloto")
    parser_prot.add_argument("--manifiesto", type=str, default="data/propio/manifiesto.csv", help="Ruta al manifiesto a auditar")

    # Subcomando 'web'
    parser_web = subparsers.add_parser("web", help="Iniciar interfaz web local para adquisición facial")
    parser_web.add_argument("--indice", type=int, default=1, help="Índice inicial de la cámara DirectShow (default: 1)")
    parser_web.add_argument("--host", type=str, default="127.0.0.1", help="Host donde escuchar (default: 127.0.0.1)")
    parser_web.add_argument("--port", type=int, default=5000, help="Puerto donde escuchar (default: 5000)")

    # Subcomando 'evaluar-lfw'
    parser_lfw = subparsers.add_parser("evaluar-lfw", help="Evaluación externa de la CNN en producción con LFW (100 identidades)")
    parser_lfw.add_argument("--semilla", type=int, default=42, help="Semilla aleatoria reproducible (default: 42)")
    parser_lfw.add_argument("--n-identidades", type=int, default=100, help="Número de identidades a evaluar (default: 100)")
    parser_lfw.add_argument("--salida", type=str, default="resultados/evaluacion_externa_lfw_cnn", help="Directorio de salida")

    args = parser.parse_args(argv)

    if args.subcomando == "ejecutar":
        return comando_ejecutar(args)
    elif args.subcomando == "importar-olivetti":
        return comando_importar_olivetti(args)
    elif args.subcomando == "registrar":
        return comando_registrar(args)
    elif args.subcomando == "procesar-imagen":
        return comando_procesar_imagen(args)
    elif args.subcomando == "demo-aliasing":
        return comando_demo_aliasing(args)
    elif args.subcomando == "verificar-conectividad":
        return comando_verificar_conectividad(args)
    elif args.subcomando == "test-controlado":
        return comando_test_controlado(args)
    elif args.subcomando == "camaras":
        return comando_camaras(args)
    elif args.subcomando == "vista-previa":
        return comando_vista_previa(args)
    elif args.subcomando == "capturar":
        return comando_capturar(args)
    elif args.subcomando == "verificar-protocolo":
        return comando_verificar_protocolo(args)
    elif args.subcomando == "web":
        return comando_web(args)
    elif args.subcomando == "evaluar-lfw":
        from reconocimiento.evaluacion.lfw import evaluar_lfw_cnn
        try:
            evaluar_lfw_cnn(
                semilla=args.semilla,
                n_identidades=args.n_identidades,
                directorio_salida=Path(args.salida),
            )
            return 0
        except Exception as e:
            print(f"Error durante evaluación externa LFW: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
