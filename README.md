# Reconocimiento facial

Interfaz web local para capturar y revisar fotos de un proyecto universitario. Cada participante guarda las imágenes en su propia computadora.

## Primera instalación en Windows

Abre PowerShell en la carpeta del proyecto. Con Python instalado, ejecuta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Abrir la interfaz web

Conecta la cámara. Si usas DroidCam, inicia primero la conexión con el celular. Desde la carpeta del proyecto ejecuta:

```powershell
.\.venv\Scripts\python.exe -m reconocimiento web --indice 1
```

Abre [la interfaz local](http://127.0.0.1:5000) en tu navegador. Si aparece otra cámara, pulsa **Sondear**, elige el dispositivo y pulsa **Iniciar cámara**. Para cerrar el servidor, usa **Ctrl+C** en PowerShell.

## Registrar una sesión

Selecciona **Sesión Real**, tu identificador asignado, la sesión y la pose. Comprueba la partición antes de guardar: sesiones 01–03 para entrenamiento, 04 para validación y 05 para prueba. Confirma que el recuadro cubre el rostro y usa **Revisar Capturas** para corregirlo si hace falta.

Git admite las fotos de las sesiones reales en `data/propio/imagenes/` y sus registros en `data/propio/manifiesto.csv`. Quienes tengan acceso al repositorio podrán ver las fotos que compartas. Los ensayos en `data/ensayos/` quedan fuera de Git. Si dos participantes modifican el manifiesto, conserva las filas de ambos al combinar sus cambios.
