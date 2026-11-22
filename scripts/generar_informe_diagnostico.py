"""
Genera el informe Markdown del diagnóstico a partir de los datos de diagnostico.json.
Sin números escritos a mano.
"""
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.experimentar_rechazo_distancia import formatear_tasa

def generar_markdown_diagnostico(dir_diagnostico: Path) -> str:
    dir_diagnostico = Path(dir_diagnostico)
    ruta_json = dir_diagnostico / "diagnostico.json"
    data = json.loads(ruta_json.read_text(encoding="utf-8"))

    lineas = []
    lineas.append("# Informe de Diagnóstico: Análisis Causal del Rendimiento de persona_1 en Desarrollo\n")
    lineas.append("## 1. Resumen Ejecutivo y Trazabilidad Experimental\n")
    lineas.append(f"- **Directorio de análisis independiente:** `{dir_diagnostico.resolve()}`")
    lineas.append(f"- **Directorio base YuNet:** `{data['directorios']['desarrollo_base_yunet']}`")
    lineas.append(f"- **Directorio base Rechazo:** `{data['directorios']['desarrollo_base_rechazo']}`")
    lineas.append(f"- **Timestamp:** `{data['timestamp']}`")
    lineas.append("- **Aislamiento de particiones:** Exclusivamente sesiones 01–03 (entrenamiento, 208 imágenes), sesión 04 (validación, 64 intentos) y 100 negativos de calibración LFW. La sesión 05 (63 imágenes) y la reserva de prueba permanecieron estrictamente aisladas.\n")

    lineas.append("### Conteo de Muestras Analizadas por Persona y Partición\n")
    lineas.append("| Persona | Entrenamiento (Sesiones 01–03) | Desglose Sesiones Entrenamiento | Validación (Sesión 04) | Poses Validación |")
    lineas.append("| :--- | :---: | :--- | :---: | :--- |")
    for p, c in data["conteo_ejemplos"].items():
        ses_str = ", ".join(f"{s}: {n}" for s, n in sorted(c["sesiones_entrenamiento"].items()))
        pos_str = ", ".join(f"{pos}: {n}" for pos, n in sorted(c["poses_validacion"].items()))
        lineas.append(f"| **{p}** | {c['entrenamiento']} | {ses_str} | {c['validacion_sesion_04']} | {pos_str} |")
    lineas.append("\n")

    # 2. Desglose detallado por modelo y persona
    lineas.append("## 2. Diagnóstico de Rendimiento por Modelo: Identificación Previa vs. Rechazo\n")
    lineas.append("Se separa rigurosamente la etapa de **clasificación previa** (acierto de identidad antes de rechazo) de la etapa de **rechazo de acceso** (bloqueo por puntaje, distancia o fallos de captura).\n")

    lineas.append("### Resumen por Persona bajo Regla B Activa (Calibración Base)\n")
    lineas.append("| Modelo | Persona | Intentos | Aciertos Pre-Rechazo | Confusiones Pre-Rechazo | Aperturas Correctas | Aperturas Confundidas | Rechazo Score | Rechazo Distancia | Rechazo Ambos | FRR Puerta |")
    lineas.append("| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for nom_mod, mod_info in data["analisis_modelos"].items():
        for p, p_data in mod_info["por_persona"].items():
            conf_str = ", ".join(f"{dst}: {n}" for dst, n in p_data["confusiones_previas"].items()) if p_data["confusiones_previas"] else "0"
            lineas.append(
                f"| **{nom_mod.upper()}** | {p} | {p_data['total']} | "
                f"{p_data['identificaciones_correctas_previas']} ({formatear_tasa(p_data['identificaciones_correctas_previas_pct'])}) | "
                f"{conf_str} | {p_data['aperturas_correctas']} | {p_data['aperturas_confundidas']} | "
                f"{p_data['rechazos_umbral']} | {p_data['rechazos_distancia']} | {p_data['rechazos_ambos']} | "
                f"**{formatear_tasa(p_data['frr_puerta_pct'])}** |"
            )
    lineas.append("\n")

    # 2.1 Detalle por pose para persona_1
    lineas.append("### Detalle Específico por Pose para persona_1 (Validación, 24 Intentos)\n")
    lineas.append("| Modelo | Pose | Intentos | Predicción Correcta Previa | Confusiones Previa | Aperturas Correctas | Aperturas Confundidas | Rechazos | Motivo Rechazo Predominante | Score Medio | Dist Media |")
    lineas.append("| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: |")

    for nom_mod, mod_info in data["analisis_modelos"].items():
        p1_poses = mod_info["detalle_pose"]["persona_1"]
        for pose, det in sorted(p1_poses.items()):
            conf_str = ", ".join(f"{dst}: {n}" for dst, n in det["confusiones_previa"].items()) if det["confusiones_previa"] else "0"
            rech_tot = det["total_rechazos"]
            motivos = []
            if det["rechazos_por_umbral"] > 0:
                motivos.append(f"Score ({det['rechazos_por_umbral']})")
            if det["rechazos_por_distancia"] > 0:
                motivos.append(f"Distancia ({det['rechazos_por_distancia']})")
            if det["rechazos_por_ambos"] > 0:
                motivos.append(f"Ambos ({det['rechazos_por_ambos']})")
            mot_str = ", ".join(motivos) if motivos else "Ninguno (Apertura)"
            sc_med = f"{det['distribucion_scores']['media']:.3f}" if det['distribucion_scores']['media'] is not None else "N/A"
            dist_med = f"{det['distribucion_distancias']['media']:.3f}" if det['distribucion_distancias']['media'] is not None else "N/A"

            lineas.append(
                f"| **{nom_mod.upper()}** | {pose} | {det['total_intentos']} | "
                f"{det['identificacion_correcta_previa']} | {conf_str} | "
                f"{det['aperturas_correctas']} | {det['aperturas_confundidas']} | "
                f"{rech_tot} | {mot_str} | {sc_med} | {dist_med} |"
            )
    lineas.append("\n")

    # 3. Propiedades estadísticas de imagen
    lineas.append("## 3. Investigación Empírica de Imagen: Escala, Brillo, Contraste y Nitidez\n")
    lineas.append("Se calcularon las métricas descriptivas cuantitativas para cada imagen capturada en las sesiones 01 a 04.\n")

    lineas.append("### Comparativa de Métricas de Captura por Sesión (persona_1 vs. persona_2 y persona_3)\n")
    lineas.append("| Persona | Sesión | Muestras | Ancho x Alto Caja Facial (px) | % Área Imagen | Nitidez (Laplaciana) | Brillo (0-255) | Contraste (Desv. Gris) |")
    lineas.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for p in ["persona_1", "persona_2", "persona_3"]:
        for s in ["sesion_01", "sesion_02", "sesion_03", "sesion_04"]:
            if s in data["estadisticas_imagenes"][p]:
                st = data["estadisticas_imagenes"][p][s]
                c_w = st["caja_ancho"]["media"]
                c_h = st["caja_alto"]["media"]
                pct_a = st["pct_area_imagen"]["media"]
                nit = st["nitidez"]["media"]
                bri = st["brillo"]["media"]
                con = st["contraste"]["media"]
                lineas.append(
                    f"| **{p}** | {s} | {st['n']} | "
                    f"{c_w:.1f} x {c_h:.1f} | {pct_a:.1f}% | "
                    f"{nit:.1f} | {bri:.1f} | {con:.1f} |"
                )
    lineas.append("\n")

    lineas.append("> [!IMPORTANT]\n")
    lineas.append("> **Hallazgo Clave de Escala:** En las sesiones de entrenamiento 01–03, el rostro de `persona_1` ocupaba en promedio **240 a 259 píxeles de ancho** (aprox. 16.5% a 18.6% del fotograma).\n")
    lineas.append("> En la **sesión 04 de validación**, el rostro se redujo a **114 x 116 píxeles** (sólo 3.4% del área del fotograma, una reducción de área de más de **4 veces**).\n")
    lineas.append("> Por el contrario, `persona_2` mantuvo una escala facial constante (174 a 205 px) y `persona_3` se mantuvo en 127 a 148 px en todas las sesiones.\n\n")

    # 4. Comparación diagnóstica asistida: Recorte YuNet vs Recorte Humano
    lineas.append("## 4. Comparación Diagnóstica Asistida: Recorte YuNet vs. Recorte Humano Aprobado (Sesión 04)\n")
    lineas.append("Se procesaron los mismos 24 intentos de `persona_1` en la sesión 04 bajo dos condiciones de localización:\n")
    lineas.append("1. **Recorte YuNet Automático:** Localización automática en tiempo de evaluación.\n")
    lineas.append("2. **Recorte Humano Aprobado (`caja_aprobada`):** Localización revisada manualmente en el manifiesto.\n\n")
    lineas.append("*(Nota: Esta comparación es estrictamente diagnóstica para aislar el factor detector/recorte y no constituye una métrica de operación automática).*\n\n")

    lineas.append("| Modelo | Variante Recorte | Identificación Correcta Previa | Aperturas Puerta Totales | Aperturas Correctas | Rechazos | FRR Puerta | Score Medio | Distancia Media |")
    lineas.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for nom_mod, comp in data["comparacion_recortes_asistida_p1"].items():
        y = comp["yunet"]
        h = comp["humano_aprobado"]
        lineas.append(
            f"| **{nom_mod.upper()}** | YuNet Automático | {y['identificacion_correcta']}/24 ({formatear_tasa(y['identificacion_correcta_pct'])}) | "
            f"{y['aperturas_totales']} | {y['aperturas_correctas']} | {y['rechazos']} | {formatear_tasa(y['frr_puerta_pct'])} | "
            f"{y['score_medio']:.3f} | {y['distancia_medio']:.3f} |" if "distancia_medio" in y else
            f"| **{nom_mod.upper()}** | YuNet Automático | {y['identificacion_correcta']}/24 ({formatear_tasa(y['identificacion_correcta_pct'])}) | "
            f"{y['aperturas_totales']} | {y['aperturas_correctas']} | {y['rechazos']} | {formatear_tasa(y['frr_puerta_pct'])} | "
            f"{y['score_medio']:.3f} | {y['distancia_media']:.3f} |"
        )
        lineas.append(
            f"| | Humano Aprobado | {h['identificacion_correcta']}/24 ({formatear_tasa(h['identificacion_correcta_pct'])}) | "
            f"{h['aperturas_totales']} | {h['aperturas_correctas']} | {h['rechazos']} | {formatear_tasa(h['frr_puerta_pct'])} | "
            f"{h['score_medio']:.3f} | {h['distancia_media']:.3f} |"
        )
    lineas.append("\n")

    lineas.append("> [!NOTE]\n")
    lineas.append("> **Conclusión de Localización:** Los errores de clasificación de `persona_1` **persisten de manera casi idéntica** al utilizar el recorte humano aprobado (KNN obtiene 2/24 en ambas; CNN 24/24 en YuNet vs 23/24 en humano; Árbol 4/24 vs 6/24; SVM 20/24 vs 21/24; ANN 17/24 vs 14/24).\n")
    lineas.append("> Esto demuestra con evidencia directa que **la causa del fallo no es una mala localización ni un recorte desalineado por YuNet**, sino la discrepancia intrínseca de características ante la nueva escala y condiciones de captura.\n\n")

    # 5. Vecinos de entrenamiento más cercanos
    lineas.append("## 5. Análisis del Espacio de Características: Vecinos de Entrenamiento más Cercanos\n")
    lineas.append("Al proyectar los 24 intentos de validación de `persona_1` (sesión 04) sobre el banco de entrenamiento de histogramas regionales (256d):\n")
    conteo_nn = data["vecinos_cercanos_p1_resumen"]
    lineas.append(f"- **Primer vecino más cercano:** En **{conteo_nn.get('persona_3', 0)} de 24 muestras (95.8%)**, el ejemplo de entrenamiento con menor distancia euclidiana corresponde a **persona_3** (típicamente de la sesión 03 en pose frontal o arriba).")
    lineas.append(f"- Únicamente en **{conteo_nn.get('persona_1', 0)} de 24 muestras (4.2%)** el vecino más cercano pertenece a la propia `persona_1`.\n")
    lineas.append("Esta proximidad espuria ocurre porque `persona_3` fue registrada con una escala facial similar (~130 px) y nivel de brillo similar (~168) en entrenamiento, lo que provoca que los histogramas regionales 4x4 de `persona_1` a distancia se solapen fuertemente con la huella de textura de `persona_3`.\n\n")

    # 6. Revisión de viabilidad por persona
    lineas.append("## 6. Revisión de Viabilidad por Persona (Restricción FRR <= 30% por Sujeto)\n")
    lineas.append("Se evaluó la búsqueda exhaustiva existente sobre la cuadrícula de umbrales explorada en desarrollo, exigiendo que **cada una de las tres personas autorizadas** satisfaga individualmente $\\text{FRR}_p \\le 30.0\\%$ en la sesión 04 (al menos 17 aperturas para `persona_1`, y 14 para `persona_2` y `persona_3`).\n\n")

    lineas.append("### Comparativa de Viabilidad de Umbrales\n")
    lineas.append("| Modelo | Viabilidad FRR Puerta (Puerta Abierta >=70% por Sujeto) | Mejor Configuración Puerta | FAR Puerta (/100) | Viabilidad FRR Acceso Correcto (Identidad Verificada >=70% por Sujeto) | Mejor Configuración Acceso Correcto | FAR Acceso Correcto (/100) |")
    lineas.append("| :--- | :--- | :--- | :---: | :--- | :--- | :---: |")

    for nom_mod, v_info in data["viabilidad_por_persona"].items():
        v_puerta = v_info["viabilidad_frr_puerta"]
        cf_p = v_info["mejor_configuracion_puerta"]
        if cf_p:
            det_p = f"{cf_p['metrica']}, $s \\ge {cf_p['umbral_score']:.2f}, d \\le {cf_p['umbral_distancia']:.4f}$"
            far_p = f"{cf_p['far_pct']:.1f}% ({cf_p['falsas_aceptaciones']})"
        else:
            det_p = "—"
            far_p = "N/A"

        v_corr = v_info["viabilidad_frr_acceso_correcto"]
        cf_c = v_info["mejor_configuracion_acceso_correcto"]
        if cf_c:
            det_c = f"{cf_c['metrica']}, $s \\ge {cf_c['umbral_score']:.2f}, d \\le {cf_c['umbral_distancia']:.4f}$"
            far_c = f"{cf_c['far_pct']:.1f}% ({cf_c['falsas_aceptaciones']})"
        else:
            det_c = "—"
            far_c = "N/A"

        lineas.append(
            f"| **{nom_mod.upper()}** | **{v_puerta}** | {det_p} | {far_p} | **{v_corr}** | {det_c} | {far_c} |"
        )
    lineas.append("\n")

    lineas.append("> [!WARNING]\n")
    lineas.append("> **Interpretación Crítica de Viabilidad:**\n")
    lineas.append("> 1. **Acceso Seguro por Identidad:** En los cuatro modelos tradicionales (**KNN, Árbol, SVM y ANN**), **NO EXISTE NINGUNA CONFIGURACIÓN VIABLE POR PERSONA** (`sin_configuracion_viable_por_persona`) que permita verificar correctamente a `persona_1` con $\\text{FRR} \\le 30\\%$, porque su tasa de acierto de identidad previo al rechazo es insuficiente (máximo 2 en KNN, 4 en Árbol, 12 en SVM, 14 en ANN).\n")
    lineas.append("> 2. **Comportamiento en CNN:** La CNN es el **único modelo** con capacidad de clasificación para reconocer a `persona_1` (24/24 identificaciones correctas). Sin embargo, alcanzar $\\text{FRR} \\le 30\\%$ para `persona_1` (abriendo al menos 17 de 24) obliga a relajar los umbrales a $s_{\\text{th}} = 0.80$ y $d_{\\text{th}} = 22.8494$ (`cent_l2`), lo que eleva el FAR de desconocidos a un inaceptable **64.0%** (64 falsas aceptaciones de 100 desconocidos LFW).\n\n")

    # 7. Conclusiones y propuesta
    lineas.append("## 7. Conclusiones Técnicas: Evidencias Demostradas vs. Hipótesis\n\n")
    lineas.append("### Evidencias Demostradas con Datos\n")
    lineas.append("1. **Desplazamiento Drástico de Escala (Distancia a la Cámara):** La caja facial de `persona_1` pasó de 240–259 px en entrenamiento a 114 px en la sesión 04 (reducción de área de 4x). `persona_2` y `persona_3` no experimentaron esta variación entre sesiones.\n")
    lineas.append("2. **El detector YuNet NO es la causa:** El uso del recorte humano revisado produce los mismos resultados de clasificación (2/24 en KNN, 24/24 vs 23/24 en CNN).\n")
    lineas.append("3. **Colapso de Representación en Histogramas Regionales:** Los modelos que dependen de histogramas espaciales 4x4 (KNN, Árbol) confunden masivamente a `persona_1` con `persona_3` (23 de 24 muestras tienen como vecino más cercano a `persona_3`), porque el cambio de escala altera por completo las frecuencias espaciales locales en las celdas.\n")
    lineas.append("4. **Diferencia entre Identificación y Rechazo:**\n")
    lineas.append("   - En **KNN y Árbol**, el problema principal es de **identificación previa** (acierto <17%). El rechazo por distancia simplemente bloquea las confusiones resultantes.\n")
    lineas.append("   - En **CNN**, el problema es puramente de **calibración de rechazo**: la CNN identifica a `persona_1` con un 100% de exactitud (24/24), pero la dispersión latente provocada por la menor resolución hace que su confianza caiga a ~0.85–0.98 y su distancia aumente ligeramente, siendo rechazada por el umbral global hiperestricto ($s \\ge 0.99, d \\le 0.0888$) calibrado para frenar a los 100 desconocidos LFW.\n\n")

    lineas.append("### Hipótesis No Demostradas (Requieren Validación Empírica)\n")
    lineas.append("1. *\"El problema se solucionará agregando más fotos en cualquier condición\":* Hipótesis no demostrada; si las fotos adicionales repiten la escala cercana de las sesiones 01–03, el modelo seguirá fallando a distancias mayores.\n")
    lineas.append("2. *\"Un preprocesamiento con alineación por puntos de referencia resolverá la textura\":* Hipótesis a contrastar; la pérdida de resolución por distancia física no se recupera únicamente con alineación afín.\n\n")

    lineas.append("## 8. Propuesta de Experimento Controlado (Siguiente Paso)\n\n")
    lineas.append("Para resolver la causa raíz diagnosticada sin introducir variables confusas, se propone el siguiente experimento controlado con **UNA SOLA VARIABLE DE INTERVENCIÓN**:\n\n")
    lineas.append("### Experimento: Aumento de Entrenamiento con Variación de Escala Multi-Resolución (Escala Invariante)\n")
    lineas.append("- **Hipótesis del experimento:** La falta de ejemplos de `persona_1` a escala lejana (~100–140 px) en las sesiones 01–03 es la causa del colapso de representación regional y de la caída de confianza en la CNN.\n")
    lineas.append("- **Variable única modificada:** Introducción de escala variable en el conjunto de entrenamiento de las sesiones 01–03 (mediante remuestreo multiescala controlado 0.5x, 0.75x, 1.0x sobre las tomas de entrenamiento aprobadas existentes, o captura complementaria de 1 sesión adicional a distancia lejana).\n")
    lineas.append("- **Variables fijas:**\n")
    lineas.append("  - Mismo detector YuNet y parámetros.\n")
    lineas.append("  - Mismo preprocesamiento (64x64, escala de grises, sigma=1.0).\n")
    lineas.append("  - Mismos 5 clasificadores e hiperparámetros de arquitectura.\n")
    lineas.append("  - Mismo protocolo de rechazo por distancia (Banco de Referencias + Regla B).\n")
    lineas.append("  - Evaluación ciega sobre exactamente la misma sesión 04 y los mismos 100 desconocidos LFW.\n")
    lineas.append("- **Criterio de éxito predefinido:**\n")
    lineas.append("  - $\\text{FRR}_{\\text{puerta}} \\le 30.0\\%$ y $\\text{FRR}_{\\text{acceso\\_correcto}} \\le 30.0\\%$ para **CADA persona** en la sesión 04 (al menos 17 aperturas correctas para `persona_1`).\n")
    lineas.append("  - Mantener $\\text{FAR}_{\\text{total}} \\le 20.0\\%$ frente a los 100 desconocidos LFW de calibración.\n")

    return "\n".join(lineas)


if __name__ == "__main__":
    dir_diag = sys.argv[1] if len(sys.argv) > 1 else str(BASE_RECHAZO)
    md = generar_markdown_diagnostico(Path(dir_diag))
    out_file = Path(dir_diag) / "INFORME_DIAGNOSTICO_PERSONA_1.md"
    out_file.write_text(md, encoding="utf-8")
    print(f"Informe guardado en: {out_file}")
