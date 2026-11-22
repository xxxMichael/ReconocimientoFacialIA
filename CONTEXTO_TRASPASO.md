# Traspaso de sesión — ReconocimientoFacialIA

Actualizado: 20 de noviembre de 2026. Este documento resume hechos verificados durante la conversación y decisiones pendientes. No sustituye los artefactos ni prueba que el árbol de trabajo no haya cambiado después.

## 1. Objetivo y forma de trabajar

Proyecto universitario en Python: reconocer tres personas autorizadas y simular apertura de una puerta en pantalla. Comparar cinco clasificadores: Árbol, KNN, SVM, ANN densa y CNN. Mantener identificables adquisición, preprocesamiento, extracción de características y clasificación. Respetar enseñanzas de clase documentadas en Markdown convertidos de PDF y ejercicios; localizar esos documentos antes de cambiar algoritmos académicos. El trabajo es en pareja y cada algoritmo debe poder recorrer las cuatro fases de forma independiente.

El asistente actúa principalmente como director y revisor del agente ejecutor: inspecciona código y resultados, explica brevemente y redacta encargos concretos. No asumir que el resumen del ejecutor demuestra que algo funciona. El usuario prefiere español, baja verbosidad y pocas actualizaciones. Prioriza funcionamiento y resultados académicos sobre endurecimiento de seguridad o ciclos interminables de pruebas. No cambiar modelos activos ni reentrenar cuando solo se solicita revisar o redactar un prompt. Se han autorizado subagentes económicos cuando sean necesarios; no son necesarios para cada revisión.

Último estado: ya se redactó y entregó al usuario el prompt para que el agente ejecutor implemente la iteración con las 75 capturas del portal. El asistente revisor no ha ejecutado ese entrenamiento ni ha recibido todavía sus resultados. El usuario pidió actualizar este traspaso para facilitar la revisión posterior. El encargo concreto y los criterios de aceptación están en el apartado 14.

## 2. Entorno y navegación

- Raíz: `C:/Dev/7-semestre/AI2/ReconocimientoFacialIA`.
- Windows, PowerShell. `python` funciona; no presuponer `.venv/Scripts/python.exe` (no existía en revisiones anteriores).
- Hay numerosos cambios locales y archivos sin seguimiento (60 entradas de git status en esta revisión). No resetear, limpiar, sobrescribir ni publicar automáticamente.
- Graphify: `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.json`, `graphify-out/graph.html`. Reporte fechado 2026-11-20: 90 archivos, 1420 nodos, 3699 aristas; 11% de relaciones inferidas. Usarlo para navegación, no como evidencia definitiva ni asumir que refleja todos los cambios recientes.
- El usuario conserva fotografías privadas y respaldos. No subirlas a Git ni publicarlas por iniciativa propia. Hubo solicitudes históricas sobre Git; consultar intención actual antes de publicar datos.

## 3. Arquitectura

- YuNet ONNX localiza rostro y puntos faciales. No identifica personas.
- Los clasificadores entrenados identifican `persona_1`, `persona_2`, `persona_3` y, en experimentos actuales, `desconocido`.
- Pipeline original: recorte con margen, gris, 64x64, float32 [0,1], gaussiano sigma 0.5 (kernel efectivo 5x5). Histogramas regionales 4x4x16 para modelos de características; CNN sobre imagen.
- Pipeline alineado: detección multiorientación 0/90/180/270, alineación de similitud de cinco puntos, aumentos solo en entrenamiento. No confundir rotación de imagen con perfiles que ocultan rasgos.
- Archivos clave: `src/reconocimiento/clasificacion/pipeline_candidato.py`, `regla_acceso.py`, `src/reconocimiento/preprocesamiento/orientacion.py`, `alineador.py`, `detector_yunet.py`, `src/reconocimiento/web/streamer.py`, `web/app.py`.
- Web local Flask, normalmente puerto 5000, cámara por OpenCV/DirectShow (DroidCam o webcam). La cámara del servidor no es la cámara del visitante remoto.
- No hay detección de vivacidad. No incorporarla salvo petición. Las pruebas sobre pantallas no equivalen a presencia física.

## 4. Candidatos y respaldo

1. CNN original cuatro clases: `configuracion_candidato_congelado_corregida.json`, umbral 0.46. Pesos en `resultados/desarrollo_4clases_corregido_20261119_123602/cnn/modelo.pt`. SHA256 histórico: `773f354628ebf09e6050e0129503113f4bbe5b7187b7eb7d3461f9bd85aaaec9`.
2. CNN alineada anterior: `configuracion_candidato_alineado.json`, umbral 0.84. Pesos en `resultados/desarrollo_4clases_alineado_20261119_223850/cnn/modelo.pt`. SHA256 histórico: `eefeaed013ce6bd4c590321720967c4ce4368dcac21f163384618e1b610326fb`.
3. CNN universitaria nueva propuesta: configuración dentro de `resultados/exp_4clases_universidad_20261120_192345/configuracion_candidato_universidad_propuesto.json`, umbral 0.50. No promover todavía a web.

Respaldos registrados: `respaldos/respaldo_pre_alineacion_20261119_221439/` (inventario y RESTAURAR.md; ejecutor reportó 130 archivos), y `respaldos/respaldo_pre_universidad_20261120_191500/`. Verificar antes de restaurar; no ejecutar restauración durante revisión.

Comando para candidata alineada: `python -m reconocimiento web --indice 1 --config-candidato configuracion_candidato_alineado.json`.

Incidente importante: arrancar sin configuración cargaba la CNN antigua de `resultados/cnn_propio` con umbral 0.65 mediante `resultados/modelo_seleccionado.json`; el selector genérico también cargaba modelos viejos. Se verificó vía API en una sesión. Nunca inferir modelo activo solo por decir CNN: comprobar nombre, configuración, umbral y pesos. Estado actual de ese selector requiere inspección, no asumir resuelto.

## 5. Datos y protocolo

Autorizados: sesiones 01–03 entrenamiento; 04 desarrollo/validación; 05 prueba histórica ya evaluada. Persona_1 agregó 60 imágenes en sesiones cercana/media/lejana de entrenamiento. Composición: 268 originales entrenamiento (148 p1, 60 p2, 60 p3), 64 validación (24/20/20), 63 prueba histórica (22/21/20).

Manifiesto efectivo suele ser `data/propio/manifiesto_local.csv`; `ManifiestoDataset` puede priorizarlo frente al principal. Hay revisión humana y recortes corregidos. Evaluación automática debe usar fotograma original y no reemplazar detección por recortes humanos para conceder aciertos.

LFW: 300 identidades de entrenamiento de cuatro clases; 100 calibración; 100 reserva histórica; además 100 identidades de una auditoría previa. Conjuntos separados por identidad. La reserva LFW y sesión05 ya se evaluaron históricamente: no presentarlas como nuevas pruebas intactas.

Portal: cinco personas `david`, `externo_piloto`, `paul`, `sonia`, `stevenl`, 15 fotos cada una (5 poses × 3 distancias). `externo_piloto` es una persona desconocida real distinta del usuario: excepción explícita autorizada. Estas cinco identidades ya se usan en entrenamiento de la nueva versión, por tanto ninguna de sus fotos es una prueba independiente de personas nuevas.

Archivo universitario: `data/recepcion_externa/fotos_desc/`. Usuario confirmó permiso, una persona por imagen, sin autorizados ni identidades anteriores. Aunque inicialmente dijo 33, inventario del ejecutor encontró 34 PNG legibles/únicos. Partición semilla 20261120: 18 train, 7 validación, 9 reserva. Manifiesto `data/universidad_desc/manifiesto_universidad.csv`. Las NUEVE identidades reservadas siguen pendientes de evaluación según el flujo inspeccionado. No abrirlas para desarrollar.

## 6. Portal remoto

Servicio independiente en `src/reconocimiento/portal/`: invitaciones, almacenamiento, importador, app y servidor Waitress. Sin galería ni descarga pública de fotos; no exponer el panel administrativo.

- Inicio: `python scripts/iniciar_portal.py --puerto 5001`.
- Túnel temporal: `cloudflared tunnel --url http://localhost:5001`.
- Invitación: `python scripts/gestionar_invitaciones.py crear --id externo_001 --base-url https://SUBDOMINIO.trycloudflare.com`.
- Importación explícita: `python scripts/importar_capturas_externas.py` (ver opciones de excepción para piloto y filtros).
- Fotos: `data/recepcion_externa/fotos/`; metadatos en recepción. Navegador usa getUserMedia, solo envía foto elegida; permite selector nativo.
- Pose/distancia se congelan con la captura y se conservan al reintentar. Se verificó código y 13 pruebas del portal. El test de cambio de selectores solo enviaba etiquetas por API, NO ejecutaba JavaScript; no afirmar que hubo prueba móvil completa.
- Identificadores con piloto/test/ensayo se excluyen por defecto; existe excepción explícita para externo_piloto real.

## 7. Resultados verificados y significado

CNN original cuatro clases, prueba histórica: 55/63 identificaciones correctas, 8 confusiones entre autorizados, 0 bloqueos; LFW 3/100 falsas aceptaciones (3/80 sobre clasificados), 20 fallos de detección. No era 100% de identificación ni ~7% FAR. Directorio `resultados/evaluacion_final_independiente_20261119_130219/`; versión corregida `evaluacion_final_independiente_corregida_20261119_204348/` conservó conteos.

Evaluación externa real original (cinco participantes independientes entonces): `resultados/evaluacion_desconocidos_reales_20261120_182810/`. Recalculado desde CSV:
- CNN original .46: 35/75 FA, 33 desconocidos explícitos, 2 rechazo score, 5 fallo detección.
- CNN alineada .84: 23/75 FA, 39 desconocidos explícitos, 10 rechazo score, 3 fallo detección. 17 FA asignadas p1 y 6 p3.
- Esto muestra peor generalización a esa fuente. Sensor/iluminación como causa concreta NO demostrados; dispositivo no documentado. Tokens distintos no son prueba biométrica de identidades distintas: se acepta declaración del usuario.

## 8. Experimento universitario actual

Script `scripts/experimentar_4clases_universidad.py`; resultados `resultados/exp_4clases_universidad_20261120_192345/`. Tres tests específicos `test_experimento_universidad.py` pasaron en revisión.

Entrenamiento construido:
- Autorizados 268 originales + 536 aumentos = 804.
- LFW 300 imágenes.
- Portal SOLO 15 fotos: frontal/media, izquierda/media, derecha/media por persona. Omitió 60 fotos para limitar contribución por identidad.
- Universidad: 15 de 18 detectadas, cada una con dos aumentos =>45 muestras; 3 fallos detectados omitidos de entrenamiento.
- Total1164: p1=444,p2=180,p3=180,desconocido=360.
- Validación64autorizados +100LFW (86detectados) +7universidad (5detectados).

Validación (correctas/confusiones/bloqueos autorizados; FA de107 desconocidos):
- Árbol sin umbral viable: Regla A27/24/13,10FA. No confundir métricas A con B habilitada.
- KNN .58:45/12/7,3FA.
- SVM .52:43/4/17,3FA.
- ANN .54:52/1/11,4FA.
- CNN nueva .50:63/0/1,0FA.
- CNN alineada anterior .84:63/0/1,0FA.

Ambas CNN EMPATAN en validación. Cero FA universitarias significa 5 rechazos por clasificación y 2 fallos del detector; no siete aciertos del clasificador. Bajar el umbral entre modelos no demuestra mejora.

## 9. Último diagnóstico revisado (estado inmediato)

Directorio `resultados/diagnostico_portal_universidad_20261120_194830/`:
- `decisiones_individuales_portal.csv`
- `diagnostico_portal_resumen.json`
- `INFORME_DIAGNOSTICO_PORTAL.md`

Se recalcularon conteos desde150filas (75 por modelo), sin reentrenar:
- Total75: anterior23FA; nueva15FA.
- 15fotos entrenadas: anterior4FA; nueva0FA.
- 60restantes de las MISMAS identidades: anterior19FA; nueva15FA.
- Pareado:51cierran ambos,9abre solo anterior,1abre solo nueva,14abren ambos.
- Nueva por pose: arriba7,abajo2,derecha4,frontal1,izquierda1.
- Nueva por distancia: cercana7/25,media3/25,lejana5/25.
- Nueva clases de falsas aperturas: p1=10,p3=5 (son15FA totales, no solo14compartidas).
- Anterior también7FA arriba: sin mejora en esa pose.

Interpretación: mejora parcial y aprendizaje de fotos vistas, no prueba de generalización a personas nuevas. La limitación a tres combinaciones dejó sin entrenar condiciones relevantes. No se ha demostrado que sea la única causa.

## 10. Siguiente paso recomendado, aún pendiente

Preparar otro experimento que utilice las75capturas del portal sin eliminar sistemáticamente poses/distancias. Conservar candidatos previos y no cambiar la web.

Controlar contribución por identidad DENTRO de desconocidos y balance entre clases por separado. No aplicar 1/n indiscriminadamente a todas las identidades: tres autorizados frente a cientos de desconocidos puede distorsionar la pérdida.
- Árbol/SVM: pesos por muestra compatibles con implementación.
- ANN/CNN: pérdida por muestra ponderada o muestreo equilibrado (no basta agregar sample_weight si la implementación PyTorch no lo usa).
- KNN: estrategia compatible sin descartar nuevamente condiciones enteras; documentar limitaciones. Pesos por distancia no son pesos de clase.

Seleccionar checkpoints, hiperparámetros y umbrales únicamente con desarrollo: sesión04 +LFWcalibración +7universidad. Las75del portal servirán solo para diagnosticar aprendizaje; una vez entrenadas todas, sus métricas son entrenamiento.

Después de decidir candidato y congelar configuración, evaluar las9identidades universitarias reservadas UNA vez. No iterar entrenamiento/umbrales con esa prueba. Su tamaño es pequeño y solo evalúa desconocidos, no FRR autorizados. Nuevas pruebas de autorizados requerirán nuevas capturas independientes si se desea conclusión final de esa versión.

## 11. Reglas de métricas y revisión

- Apertura autorizada con identidad equivocada = confusión, NO bloqueo de puerta.
- Intentos autorizados = bloqueos + aperturas correctas + confusiones.
- FRR puerta = bloqueos/intentos autorizados, con restricción por persona <=30%.
- FAR total = aperturas de desconocidos/todos sus intentos. Reportar también condicionado a clasificación y separar fallos de detección/procesamiento.
- Cero denominador = null/no calculable, no0%.
- Calibración: menor FAR, menor FRR, más identidades correctas, mayor umbral como último desempate. Sin candidato viable =>null y regla deshabilitada, nunca respaldo automático.
- No atribuir diferencias causalmente a alineación cuando también cambiaron pesos, entrenamiento y umbral.
- Registrar decisiones individuales y calcular tablas desde artefactos, no cifras escritas directamente.

## 12. Antecedentes técnicos a no perder

Hubo fallos de tests por cámara física abierta automáticamente y singleton/hilos. Se cambió crear_app(indice_inicial=-1) en tests y lectura del buffer. Históricamente detener no esperaba el hilo; no asumir resuelto sin revisar. El usuario pidió no seguir ampliando pruebas marginales si no afectan funcionamiento.

Problemas anteriores de NaN y configuración duplicada motivaron pipeline compartido. Persistían detalles de validación/configuración, pero no son prioridad salvo efecto funcional. La revisión no debe convertirse en auditoría de seguridad comercial.

Historial de pruebas verificadas:157, luego167, luego180 globales pasaron en distintas revisiones; después13del portal y3universidad pasaron específicamente. No afirmar que se acaba de ejecutar la suite global actual sin ejecutarla. No afirmar pruebas en teléfono/cámara física realizadas por el asistente si solo hubo fixtures.

## 13. Arranque recomendado de la nueva sesión

1. Leer este archivo y Graphify para ubicar módulos; git status antes de cambios.
2. Confirmar el último pedido del usuario: normalmente redactar prompt o revisar resultados, no modificar automáticamente.
3. Usar el diagnóstico del apartado9 como punto de partida; no repetir investigación desde cero.
4. El encargo de la iteración del apartado 10 YA FUE REDACTADO. Consultar el apartado 14 antes de revisar la respuesta del ejecutor; no volver a emitirlo ni asumir que se completó.

## 14. Último prompt entregado: cobertura completa del portal y balance

### Estado y objetivo

Encargo entregado después de revisar el diagnóstico de 23 a 15 falsas aceptaciones. Pendiente de recibir y auditar la ejecución. Se busca aprovechar poses y distancias antes omitidas sin deteriorar el reconocimiento de autorizados. La mejora buscada es en validación y, posteriormente, en identidades nuevas; memorizar las 75 capturas no basta.

El prompt solicita implementar UNA iteración acotada de los cinco algoritmos, no seguir experimentando automáticamente hasta obtener cero errores. No activar el candidato en la web ni evaluar la reserva en esta entrega.

### Composición exigida

- Entrenamiento: autorizados actuales incluidas sesiones multidistancia; 300 identidades LFW de entrenamiento; 18 identidades universitarias de entrenamiento; las 75 capturas del portal.
- Procesar todas las combinaciones disponibles del portal, sin restringir nuevamente a distancia media o a tres poses. Si alguna imagen no produce entrada válida, registrar su fallo y excluirla explícitamente del entrenamiento, sin inventar recortes ni afirmar que se entrenó con ella.
- Manifiesto específico nuevo; preservar fotografías, registros históricos, modelos y resultados anteriores.
- Validación exclusiva: sesión 04, 100 LFW de calibración y siete identidades universitarias de validación.
- Mantener fuera de este experimento las nueve identidades universitarias reservadas, sesión 05 y reserva LFW.

### Balance y entrenamiento exigidos

- Controlar contribución por identidad dentro de desconocido y balance entre las cuatro clases por separado.
- No aplicar 1/n_fotos indiscriminadamente a todas las identidades: podría infraponderar a los tres autorizados frente a cientos de desconocidos.
- Árbol/SVM: pesos por muestra realmente usados; comprobar interacción con class_weight para no duplicar compensaciones.
- ANN/CNN: pérdida individual ponderada o muestreo equilibrado aplicado en el bucle de entrenamiento, no un argumento decorativo.
- KNN: mantener cobertura de condiciones; justificar cualquier muestreo/prototipo y compatibilidad académica. weights='distance' no equivale a balanceo de clases.
- Registrar pesos efectivos por clase, fuente e identidad. Los aumentos no deben incrementar accidentalmente el peso total de una identidad.
- Mantener arquitecturas y pipeline alineado; aumentos solo en entrenamiento; escaladores ajustados solo con entrenamiento. Sin embeddings externos, rechazo métrico ni vivacidad en este encargo.

### Selección, métricas y diagnóstico exigidos

- Seleccionar checkpoints, parámetros y umbrales solo con validación. Nunca con las 75 capturas del portal.
- Recalibrar cada modelo; no copiar automáticamente 0.50 ni 0.84.
- Mantener restricción FRR por autorizado <=30% y desempate canónico del apartado 11. Ausencia de candidato viable => null y regla B deshabilitada.
- Comparar la nueva iteración con ambas CNN anteriores sobre las mismas muestras de validación, con sus configuraciones respectivas.
- Reportar identificación, confusiones, bloqueos, FRR por persona, FAR por fuente, rechazos explícitos/por score, fallos y latencia completa.
- Evaluar después las 75 del portal únicamente como diagnóstico de ENTRENAMIENTO. Mantener desglose histórico de 15 originales y 60 incorporadas, aclarando que ahora ambas partes son elegibles para entrenamiento.
- Informar expresamente qué sucede en arriba/abajo y cercana/lejana. Una caída del error ahí confirma aprendizaje de esas condiciones, no generalización a personas nuevas.

### Evidencia que debe entregar el ejecutor

1. Directorio nuevo y fechado, modelos/configuraciones separados y respaldo previo comprobado.
2. Estrategia de balance por algoritmo y cantidades realmente procesadas/excluidas.
3. Manifiestos, hashes, semillas, historiales y decisiones individuales.
4. Tabla de validación y diagnóstico del portal con conteos y desgloses coherentes.
5. Configuración candidata propuesta aparte, sin promoción automática.
6. Pruebas de separación de particiones, cobertura, pesos efectivos, aumentos solo train, métricas y carga; suite completa con resultado real.
7. Recomendación de congelar la nueva versión o conservar una anterior; declaración de que web y reserva permanecieron intactas.

### Cómo revisar el resultado en la nueva sesión

- Leer el resumen del ejecutor, pero verificar código y artefactos reales antes de confirmar cifras.
- Comprobar que las 60 fotos adicionales no se filtraron nuevamente por pose, distancia o scores del modelo. Distinguir fallos legítimos del detector de selección por conveniencia.
- Revisar pesos efectivos y la función de pérdida/muestreo, no solo documentación. Comprobar especialmente doble ponderación y aumentos.
- Recalcular conteos desde predicciones; contrastar curvas y selección de umbral con la validación declarada.
- Confirmar que mejoría en portal no sea presentada como evaluación independiente y que las nueve identidades sigan sin evaluar.
- Si ambas CNN vuelven a empatar en validación, decirlo. No declarar ganadora a la nueva únicamente por memorizar más fotos ni por tener otro umbral. Documentar criterio de selección antes de usar reserva.
- Si el candidato está listo, el siguiente encargo podrá congelarlo y ejecutar una única evaluación final sobre las nueve identidades nuevas. No hacerlo por iniciativa durante la revisión.
- Si no hay mejora, informar el resultado y proponer una decisión acotada; evitar encadenar reentrenamientos indefinidos con el mismo conjunto.


## 15. Iteración controlada con revisiones manuales de `fotos_desc` (2026-11-20)

- **Auditoría de `data/universidad_desc/revision_universidad.json`:** Las 34 imágenes tienen decisión `aprobada` dentro de los límites de imagen.
  - **Entrenamiento (18):** Se rescataron **3 identidades** que antes quedaban fuera por `VARIOS_ROSTROS` (`universidad_014`, `021`, `028`). Al no haber 5 puntos válidos de YuNet, se aplicó `preprocesar_rostro_alineado` con `puntos_faciales=None` y `recuadro=caja_aprobada` (`fallback_recorte_seguro`), sin inventar landmarks. En las otras 15 (incluidas `001` y `022` con caja aprobada distinta), los 5 puntos de YuNet caían dentro de la caja aprobada y conservaron `similitud_5puntos`.
  - **Validación (7):** `007` y `012` fallan en YuNet automático (`VARIOS_ROSTROS`), pero se mantuvieron en automático para calibrar la operación real en vivo.
  - **Prueba reservada (9):** Ya evaluada previamente en `evaluacion_final_universidad_20261120_203322`; **no** se reevaluó ni se usó para ajustar parámetros.
- **Resultado (`resultados/exp_4clases_univ_corregida_20261120_213750/`):**
  - Entrenamiento total: `1,230` muestras (`+9` por las 3 identidades rescatadas × 3).
  - **Validación automática (`64` Auth Sesión 04 + `100` LFW + `7` Univ):**
    - **Referencia Portal Completo (`0.61`, candidato principal recomendado):** `63` identificaciones correctas, `0` confusiones, `1` bloqueo (`FRR_puerta = 1.56%`); `FAR = 0/107` (`91` rechazos explícitos como `desconocido` + `16` fallos de detección: `14` LFW y `2` Univ).
    - **Nueva CNN corregida (`0.57`, alternativa experimental):** `63` identificaciones correctas, `1` confusión, `0` bloqueos (`64` aperturas totales, `FRR_puerta = 0.00%`, pero **no** aumentaron las identificaciones correctas); `FAR = 0/107` (`91` rechazos explícitos + `16` fallos de detección).
    - **Detalle de la confusión:** `persona_3_sesion_04_frontal_02` (real: `persona_3`, pose: `frontal`) fue clasificada por la nueva CNN (`0.57`) como `persona_2` (`0.7026`, `ABIERTA`), mientras la referencia (`0.61`) la clasificó como `desconocido` (`0.9868`, `CERRADA`).
  - **Auditoría reconstruida sin reentrenamiento:** `resultados/exp_4clases_univ_corregida_20261120_213750/auditoria_reconstruida/` (decisiones individuales, 4 probabilidades y curva de umbrales con concordancia 100%).
  - **Recomendación:** Se conserva **Portal Completo (`0.61`)** como candidato principal recomendado por tener `0` confusiones en validación; la nueva CNN (`0.57`) queda como alternativa experimental. No se promueve automáticamente a la web.

## 16. Experimento acotado de remuestreo con antialiasing en alineación (2026-11-20)

- **Implementación:** Se añadió el parámetro versionado `metodo_remuestreo = "supersampling_area"` (por defecto `"estandar"` para preservar compatibilidad total con configuraciones previas) en `alinear_rostro`, `preprocesar_rostro_alineado`, `DetectorYuNet` y `PipelineCandidatoCongelado`.
  - Cuando `escala < 0.9`, aplica sobremuestreo $k = \text{clamp}(\text{round}(1 / \text{escala}), 2, 16)$ con desplazamiento sub-píxel $(k-1)/2$ antes de reducir a `64×64` mediante `cv2.resize(..., INTER_AREA)`. Cuando `escala >= 0.9` (ampliación), no desenfoca.
- **Verificación sintética:** En un patrón sintético de alta frecuencia (`512×512 → 64×64`), redujo la desviación estándar residual de `104.14` a `3.21` (`96.92%` de reducción en ese patrón sintético; esto mide atenuación de variación residual en patrón de prueba, no garantiza el mismo efecto en todas las fotografías reales ni demuestra por sí solo mejor reconocimiento). Latencia media de detección+alineación prácticamente idéntica (`17.08 ms` vs `17.49 ms`).
- **Comparación controlada en validación automática (`resultados/exp_4clases_antialiasing_20261120_233516/`):**
  - **Referencia Portal Completo (`0.61`):** `63/64` correctas, `0` confusiones, `1` bloqueo (`FRR = 1.56%`), `FAR = 0/107` (`91` rechazos explícitos + `16` fallos de detección).
  - **CNN Antialiasing (`0.52`):** `63/64` correctas, `0` confusiones, `1` bloqueo (`FRR = 1.56%`), `FAR = 0/107` (`91` rechazos explícitos + `16` fallos de detección).
- **Conclusión:** Empate exacto en validación automática.

## 17. Preparación de CNN Antialiasing — 0.52 como versión principal y evaluación en Sesión 05 (2026-11-21)

- **Configuraciones congeladas estables (sin sobrescribir experimentos ni `resultados/modelo_seleccionado.json`):**
  - **Principal:** `configuracion_candidato_antialiasing_052.json` (`CNN Antialiasing — 0.52`, `metodo_remuestreo="supersampling_area"`, umbral `0.52`, SHA-256 pesos `581b805b649be08577e8157fe26fc30d8d8bfe90a3b4819379f42051bc613f7f`).
  - **Comparación / reversión:** `configuracion_candidato_estandar_portal_061.json` (`CNN Estándar Portal Completo — 0.61`, `metodo_remuestreo="estandar"`, umbral `0.61`, SHA-256 pesos `24d3a1ed14cd8750863ff5bf464185c22eaef02bf4313044b64747929bd636f9`).
- **Paridad verificada:** Ambas variantes entrenaron sobre exactamente las mismas `1,221` muestras (`semilla=42`, pesos de clase `305.25`, arquitectura idéntica). Equivalencia numérica (`atol=1e-6`) comprobada entre `preprocesar_rostro_alineado` y `PipelineCandidatoCongelado` + `inspeccionar_fotograma_pipeline` en múltiples escalas y rotaciones.
- **Evaluación en Sesión 05 (`resultados/evaluacion_sesion05_antialiasing_20261121_115108/resumen_sesion05.json`, 63 intentos autorizados; sin desconocidos, por lo que no mide FAR):**
  - **CNN Estándar Portal Completo — 0.61:** `58/63` (`92.06%`) correctas, `5` confusiones (`persona_1→persona_2`: 2, `persona_2→persona_1`: 3), `0` bloqueos (`FRR=0.0%`), latencia media total `19.14 ms`.
  - **CNN Antialiasing — 0.52:** `59/63` (`93.65%`) correctas, `4` confusiones (`persona_1→persona_2`: 1, `persona_2→persona_1`: 3), `0` bloqueos (`FRR=0.0%`), latencia media total `19.54 ms`.
  - **Comparación pareada:** `58` ambas correctas, `1` solo Antialiasing (`persona_1_sesion_05_izquierda_01`), `0` solo Estándar, `4` ninguna.
- **Interfaz web y reversión:** `python -m reconocimiento web` carga por defecto `configuracion_candidato_antialiasing_052.json`. El selector `#select-modelo-recon` ofrece explícitamente `"CNN Antialiasing — 0.52 (Principal)"` y `"CNN Estándar Portal Completo — 0.61 (Comparación)"` (sin selector genérico `"CNN"` antiguo). Para volver a la referencia estándar en CLI: `python -m reconocimiento web --config-candidato configuracion_candidato_estandar_portal_061.json`.

## 18. Corrección de pendientes y portal independiente de reconocimiento remoto (2026-11-21)

- **Concurrencia en `GestorCamaraWeb` (`src/reconocimiento/web/streamer.py`):** Se añadió `lock_modelo` y `lock_inferencia` con instantánea atómica (`_obtener_instantanea_modelo`) para garantizar que cada fotograma utilice detector, preprocesamiento, clasificador y umbral del mismo candidato sin mezclas durante cambios de modelo.
- **Medición de latencia en `scripts/evaluar_sesion05_comparativa.py`:** Se eliminó la llamada redundante a `preprocesar_recorte` antes de `evaluar_recorte` (que ya preprocesa internamente). Los resultados históricos en `resultados/evaluacion_sesion05_antialiasing_20261121_115108/` se conservaron intactos.
- **Auditoría de paridad de entrenamiento (`scripts/auditar_paridad_entrenamiento.py` → `resultados/auditoria_paridad_antialiasing_20261121/`):**
  - *Evidencia original:* `exp_4clases_portal_completo_20261120_200452/manifiesto_entrenamiento_experimento.csv` (`1,221` filas), `resumen_experimento_portal_completo.json` (`semilla=42`, pesos de clase `305.25`) y `resumen_experimento_antialiasing.json` / `configuracion_candidato_antialiasing_propuesto.json`.
  - *Reconstrucción posterior:* `inventario_reconstruido_1221.csv` y `resumen_paridad.json` (`1,221` muestras reconstruidas, `655` hashes SHA-256 únicos de archivos fuente actuales, `0` archivos faltantes).
  - *Limitación documentada:* `exp_4clases_antialiasing_20261120_233516/` no guardó su propio `manifiesto_entrenamiento_experimento.csv` al entrenar, por lo que la igualdad fila-a-fila se sustenta en reconstrucción posterior del script determinista y conteos originales.
- **Portal independiente de reconocimiento remoto (`src/reconocimiento/portal_reconocimiento/`):**
  - Aislado de la web administrativa (`5000`) y del portal de captura (`5001`). Reutiliza `PipelineCandidatoCongelado` y carga una sola vez `cnn_antialiasing_052` (`0.52`) y `cnn_estandar_portal_061` (`0.61`) mediante lista cerrada.
  - Procesa fotogramas de `getUserMedia` estrictamente en memoria (~2 req/s adaptativo, máx. 1 solicitud pendiente por cliente, descarte por `frame_seq`/epoch, sin almacenar fotos ni base64).
  - Registra intentos explícitos con `id_intento` deduplicado en `resultados/portal_reconocimiento/intentos.jsonl` (etiquetas autodeclaradas por participantes, sin influir en la inferencia). Ante pérdida de respuesta, timeout o reinicio del servidor, `servicio.py` recupera el `registro` original por `id_intento` sin reevaluar ni duplicar; el cliente (`portal_reconocimiento.js`) conserva el intento pendiente en memoria (sin guardarlo en `localStorage` ni disco) para **Reintentar intento** o **Descartar intento pendiente**, usa `AbortController` con timeout explícito, protege `getUserMedia` pendiente y pistas `ended`, y coordina vista previa y registro bajo un bloqueo único por visitante.
  - Pruebas ejecutables del frontend en `tests/test_portal_reconocimiento_frontend.mjs` (9 escenarios, incluyendo timeout sobre `resp.json()` atascado, limpieza de puerta abierta ante `503`/timeout de registro y botón `Detener` habilitado durante `getUserMedia` pendiente) integradas en `tests/test_portal_reconocimiento.py`.
  - Inicio local en `127.0.0.1:5002` con Waitress (`python scripts/iniciar_portal_reconocimiento.py --puerto 5002` o `python -m reconocimiento portal-reconocimiento --port 5002`) y publicación HTTPS documentada en `GUIA_PORTAL_RECONOCIMIENTO.md` mediante `cloudflared tunnel --url http://127.0.0.1:5002`.
- **Cambio de modelo predeterminado (2026-11-21):** Tras validar en el piloto remoto que `CNN Estándar Portal Completo — 0.61` rechazó correctamente a los participantes desconocidos que fallaban con `AA 0.52`, se fijó `CNN Estándar Portal Completo — 0.61` (`configuracion_candidato_estandar_portal_061.json`) como candidato principal por defecto en el portal remoto (`CANDIDATO_PRINCIPAL_ID`), en `python -m reconocimiento web` y en el selector web (`#select-modelo-recon`), conservando `CNN Antialiasing — 0.52` para comparación. Diagnóstico completo previo guardado en `resultados/diagnostico_piloto_remoto_20261121_142849/`.
- **Pruebas en vivo reportadas por el usuario con `CNN Estándar Portal Completo — 0.61` (2026-11-21):**
  - `persona_1`: puerta `ABIERTA` (`97.3%–99%`).
  - `persona_2`: puerta `ABIERTA` (`93.5%`).
  - Desconocidos re-testeados: rechazados correctamente como `desconocido`.
  - `persona_3`: **pendiente** de prueba en vivo.
- **Fase 1 — Compuerta opcional de calidad sin reentrenar (2026-11-21):**
  - Implementada en `src/reconocimiento/preprocesamiento/calidad.py` e integrada opcionalmente en `PipelineCandidatoCongelado` (solo cuando `filtro_calidad.habilitado = true`, manteniendo intactos `configuracion_candidato_estandar_portal_061.json` y `configuracion_candidato_antialiasing_052.json`).
  - Nuevo candidato seleccionable en el portal remoto: `configuracion_candidato_estandar_portal_061_calidad.json` (`CNN Estándar 0.61 + Filtro Calidad (Fase 1)`).
  - Evaluación offline (`scripts/evaluar_fase1_calidad.py`): en portal `75` redujo falsas aceptaciones de `22.67% → 0.0%` (`oscura`), `22.67% → 0.0%` (`oscura_ruido`) y `21.33% → 0.0%` (`oclusion_inferior`), manteniendo `0.0%` en `limpia`, `60/64` en `sesion_04` limpia y `57/63` en `sesion_05` limpia (`0` confusiones adicionales).
- **Fase 2 — Iteración controlada de entrenamiento con robustez (`scripts/experimentar_fase2_robustez.py`, 2026-11-21):**
  - Entrenada en directorio aparte (`resultados/exp_4clases_fase2_robustez_20261121_155939/`, `2,131` muestras con aumentos de oscuridad/contraluz/oclusión y reponderación de `desconocido_portal` al `30%` de la clase `desconocido`).
  - Configuración congelada aparte en `configuracion_candidato_fase2_robustez.json` (`umbral=0.62`, SHA-256 `94dca14969e9383f6abecc7085afbab86f9672bd7817c660651b60fd08418a82`), seleccionable en el portal remoto sin reemplazar a `CNN Estándar Portal Completo — 0.61`.
  - Comparativa en desconocidos (`75`): `oscura` `22.67% → 0.0%`, `oscura_ruido` `22.67% → 0.0%`, `contraluz` `13.33% → 0.0%`, `oclusion_inferior` `21.33% → 4.0%`, `oclusion_lateral` `20.0% → 10.67%`; trade-off: `portal limpia` `0.0% → 6.67%` (`5/75`), `sesion_04` `62/64` (`FRR 3.12%`) y `sesion_05` `56/63` (`3` confusiones, `4` bloqueos).
