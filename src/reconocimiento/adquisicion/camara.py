"""
Módulo de adquisición en vivo con cámara (DirectShow / DroidCam).
Implementa sondeo de dispositivos, vista previa interactiva con detección facial,
y modo de captura manual/guiada con preservación de fotogramas originales
y consistencia atómica de manifiesto.
"""
import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from reconocimiento.adquisicion.propio import (
    ManifiestoDataset,
    RegistroCaptura,
    POSES_VALIDAS,
    PARTICIONES_VALIDAS,
    calcular_hash_archivo,
    validar_coherencia_sesion_particion_propio,
    es_manifiesto_propio
)
from reconocimiento.preprocesamiento.fabrica import crear_detector
from reconocimiento.preprocesamiento.detector import DetectorFacial, EstadoDeteccion

_LOCK_GUARDADO = threading.Lock()


def sanitizar_componente_ruta(valor: str) -> str:
    """
    Valida y sanitiza nombres de persona, sesión, pose o partición para evitar path traversal.
    Solo permite caracteres alfanuméricos, guiones y guiones bajos.
    """
    texto = str(valor).strip()
    if not texto or not re.match(r"^[a-zA-Z0-9_\-]+$", texto):
        raise ValueError(
            f"Componente de ruta inválido '{valor}'. Solo se permiten caracteres alfanuméricos, guiones y guiones bajos."
        )
    return texto


def sondear_camaras(
    max_indices: int = 5,
    backend: int = cv2.CAP_DSHOW,
    omitir: Optional[Dict[int, Tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Sondea un rango pequeño de índices de video usando DirectShow (Windows),
    obtiene dimensiones reales y backend, y libera inmediatamente cada dispositivo.

    `omitir` mapea índices en uso (ya abiertos por la aplicación) a su (ancho, alto):
    no se vuelven a abrir, porque abrir y liberar un dispositivo activo corta su flujo.
    """
    omitir = omitir or {}
    dispositivos = []
    for idx in range(max_indices):
        if idx in omitir:
            ancho_o, alto_o = omitir[idx]
            dispositivos.append({
                "indice": idx,
                "backend": "CAP_DSHOW (DirectShow)",
                "disponible": True,
                "abierto": True,
                "en_uso": True,
                "ancho": ancho_o,
                "alto": alto_o,
                "canales": 3,
                "fps_nominal": 30.0
            })
            continue
        cap = cv2.VideoCapture(idx, backend)
        abierto = cap.isOpened()
        if abierto:
            # Intentar lectura de prueba con hasta 3 intentos de arranque
            exito_lectura = False
            ancho, alto, canales = 0, 0, 0
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    exito_lectura = True
                    alto, ancho = frame.shape[:2]
                    canales = frame.shape[2] if len(frame.shape) > 2 else 1
                    break
                time.sleep(0.05)

            fps = cap.get(cv2.CAP_PROP_FPS)
            dispositivos.append({
                "indice": idx,
                "backend": "CAP_DSHOW (DirectShow)",
                "disponible": exito_lectura,
                "abierto": True,
                "ancho": ancho,
                "alto": alto,
                "canales": canales,
                "fps_nominal": fps if fps > 0 else 30.0
            })
            cap.release()
        else:
            dispositivos.append({
                "indice": idx,
                "backend": "CAP_DSHOW (DirectShow)",
                "disponible": False,
                "abierto": False,
                "ancho": 0,
                "alto": 0,
                "canales": 0,
                "fps_nominal": 0.0
            })
    return dispositivos


class CapturadorCamara:
    """
    Gestiona el ciclo de vida del flujo de video desde una cámara física o fuente simulada.
    Garantiza tolerancia a arranque breve, detección de pérdida de señal y liberación segura.
    """

    def __init__(
        self,
        indice: int = 1,
        backend: int = cv2.CAP_DSHOW,
        intentos_arranque: int = 5,
        max_fallos_consecutivos: int = 10,
        fuente_custom: Any = None
    ):
        self.indice = indice
        self.backend = backend
        self.intentos_arranque = intentos_arranque
        self.max_fallos_consecutivos = max_fallos_consecutivos
        self.fuente_custom = fuente_custom
        self.cap: Optional[Any] = None
        self.fallos_consecutivos: int = 0
        self.esta_activo: bool = False

    def abrir(self) -> bool:
        """Abre el dispositivo y calienta el sensor descartando fotogramas vacíos iniciales."""
        if self.fuente_custom is not None:
            self.cap = self.fuente_custom
        else:
            self.cap = cv2.VideoCapture(self.indice, self.backend)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir el dispositivo de video en índice {self.indice} con backend DirectShow. "
                "Verifique la conexión física o la aplicación de streaming (ej. DroidCam)."
            )

        # Tolerancia a arranque breve: descartar fotogramas de calentamiento
        fotograma_valido = False
        for intento in range(self.intentos_arranque):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                fotograma_valido = True
                break
            time.sleep(0.08)

        if not fotograma_valido:
            self.liberar()
            raise RuntimeError(
                f"El dispositivo en índice {self.indice} abrió conexión pero no entregó fotogramas válidos "
                f"tras {self.intentos_arranque} intentos de arranque inicial."
            )

        self.esta_activo = True
        self.fallos_consecutivos = 0
        return True

    def leer(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lee un fotograma BGR original. Lanza excepción si se interrumpe la señal sostenidamente."""
        if not self.esta_activo or self.cap is None:
            raise RuntimeError("El capturador no ha sido abierto o ya fue liberado.")

        ret, frame = self.cap.read()
        if ret and frame is not None and frame.size > 0:
            self.fallos_consecutivos = 0
            return (True, frame)

        self.fallos_consecutivos += 1
        if self.fallos_consecutivos >= self.max_fallos_consecutivos:
            self.liberar()
            raise RuntimeError(
                f"Pérdida de señal en la cámara (índice {self.indice}): {self.fallos_consecutivos} "
                "lecturas fallidas consecutivas. El dispositivo se desconectó o la transmisión finalizó."
            )

        return (False, None)

    def obtener_dimensiones(self) -> Tuple[int, int]:
        """Retorna (ancho, alto) de la resolución activa."""
        if not self.esta_activo or self.cap is None:
            return (0, 0)
        ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (ancho, alto)

    def liberar(self) -> None:
        """Libera de forma garantizada el descriptor de video."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.esta_activo = False

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.liberar()


def guardar_fotograma_atomico(
    frame_original: np.ndarray,
    ruta_destino: Path,
    registro: RegistroCaptura,
    manifiesto: ManifiestoDataset
) -> None:
    """
    Guarda el fotograma original en disco y actualiza el manifiesto de forma coherente y atómica.
    Protegido por cerrojo ante peticiones concurrentes.
    Si falla el guardado en disco o en manifiesto, revierte cambios para evitar archivos huérfanos.
    """
    with _LOCK_GUARDADO:
        ruta_destino = Path(ruta_destino).resolve()
        if ruta_destino.exists():
            raise FileExistsError(f"El archivo de destino ya existe y se prohíbe sobrescribir: {ruta_destino}")

        ruta_destino.parent.mkdir(parents=True, exist_ok=True)

        # 1. Guardar imagen original sin alteraciones
        exito_escritura = cv2.imwrite(str(ruta_destino), frame_original)
        if not exito_escritura:
            raise IOError(f"Fallo al escribir el fotograma en disco: {ruta_destino}")

        # 2. Calcular hash SHA-256 del archivo escrito
        hash_sha256 = calcular_hash_archivo(ruta_destino)
        registro.hash_contenido = hash_sha256
        registro.ruta = str(ruta_destino)

        # 3. Intentar agregar al manifiesto
        try:
            manifiesto.agregar_registro(registro)
            # Validar aislamiento y duplicados
            es_val_ses, err_ses = manifiesto.validar_aislamiento_sesiones()
            if not es_val_ses:
                raise ValueError(f"Violación de aislamiento de sesiones: {err_ses}")
            es_val_dup, err_dup = manifiesto.validar_duplicados_entre_particiones()
            if not es_val_dup:
                raise ValueError(f"Violación de duplicados entre particiones: {err_dup}")
            es_val_desc, err_desc = manifiesto.validar_aislamiento_desconocidos()
            if not es_val_desc:
                raise ValueError(f"Violación de aislamiento de desconocidos: {err_desc}")
            manifiesto.guardar()
        except Exception as e:
            # Reversión atómica: borrar imagen en disco si el manifiesto falla
            if ruta_destino.exists():
                ruta_destino.unlink(missing_ok=True)
            # Quitar registro del manifiesto en memoria si alcanzó a agregarse
            manifiesto.registros = [r for r in manifiesto.registros if r.identificador != registro.identificador]
            raise e


def ejecutar_vista_previa(
    indice: int = 1,
    detector: Optional[DetectorFacial] = None,
    capturador_custom: Optional[CapturadorCamara] = None,
    fps_max: int = 30
) -> None:
    """
    Abre la vista previa en vivo con selección explícita de cámara (DirectShow).
    Aplica el detector Haar sobre el fotograma original y dibuja estado y recuadro
    sobre una copia para pantalla. Permite salir limpiamente con Esc.
    """
    if detector is None:
        detector = crear_detector()

    capturador = capturador_custom or CapturadorCamara(indice=indice)
    titulo_ventana = f"Vista Previa - Reconocimiento Facial (Indice {indice} | ESC: Salir)"

    print(f"\nIniciando vista previa en vivo en cámara índice {indice} (DirectShow)...")
    print("Controles: Presione [ESC] para salir.")
    print("Mueva la cabeza para verificar recuadros al girar (frontal, izquierda, derecha, arriba, abajo).\n")

    try:
        capturador.abrir()
        ancho, alto = capturador.obtener_dimensiones()
        print(f"Flujo de video activo. Resolución nominal: {ancho}x{alto}")

        cv2.namedWindow(titulo_ventana, cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = capturador.leer()
            if not ret or frame is None:
                continue

            # Ejecutar detector en el fotograma ORIGINAL
            resultado_det = detector.detectar(frame, es_pre_recortada=False)

            # Dibujar sobre COPIA para pantalla (el original no se modifica)
            display = frame.copy()
            h_f, w_f = display.shape[:2]

            # Encabezado HUD superior
            cv2.rectangle(display, (0, 0), (w_f, 40), (25, 25, 25), -1)
            cv2.putText(
                display,
                f"CAM {indice} (DirectShow) | {w_f}x{h_f} | ESC: Salir",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            # Colores y mensajes según estado de detección
            if resultado_det.estado == EstadoDeteccion.UN_ROSTRO:
                color_box = (0, 255, 0)  # Verde
                texto_estado = "ESTADO: UN_ROSTRO (Detectado)"
                if resultado_det.recuadro is not None:
                    rx, ry, rw, rh = resultado_det.recuadro
                    cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), color_box, 2)
                    cv2.putText(
                        display,
                        f"Rostro [{rw}x{rh}]",
                        (rx, max(20, ry - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_box,
                        1,
                        cv2.LINE_AA
                    )
            elif resultado_det.estado == EstadoDeteccion.SIN_ROSTRO:
                color_box = (0, 180, 255)  # Ámbar/Naranja
                texto_estado = "ESTADO: SIN_ROSTRO (Gire hacia la camara)"
            else:  # VARIOS_ROSTROS
                color_box = (0, 0, 255)  # Rojo
                texto_estado = f"ESTADO: VARIOS_ROSTROS ({resultado_det.total_rostros_detectados} detectados)"

            # Barra de estado inferior
            cv2.rectangle(display, (0, h_f - 35), (w_f, h_f), (25, 25, 25), -1)
            cv2.putText(
                display,
                texto_estado,
                (12, h_f - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_box,
                2,
                cv2.LINE_AA
            )

            cv2.imshow(titulo_ventana, display)

            tecla = cv2.waitKey(int(1000 / fps_max)) & 0xFF
            if tecla == 27:  # Esc
                print("Salida solicitada por el usuario (ESC).")
                break

    finally:
        capturador.liberar()
        cv2.destroyAllWindows()
        print("Dispositivo de video y ventanas liberados correctamente.")


def ejecutar_modo_captura(
    indice: int,
    persona: str,
    sesion: str,
    pose: str,
    particion: str,
    consentimiento: bool,
    ruta_manifiesto: Path,
    carpeta_destino: Path,
    tipo_sujeto: str = "autorizado",
    detector: Optional[DetectorFacial] = None,
    capturador_custom: Optional[CapturadorCamara] = None,
    num_fotos_objetivo: int = 1,
    es_ensayo: bool = False
) -> List[Dict[str, Any]]:
    """
    Modo de captura manual interactivo.
    - Exige consentimiento explícito.
    - Orienta la pose desde la perspectiva del sujeto (sujeto gira a su izquierda o derecha).
    - Conserva el fotograma original sin dibujos ni reflejos horizontales.
    - Registra dimensiones reales y evita sobrescrituras.
    - Permite guardar intentos fallidos de detección con [F] para evaluar la tasa de fallos por pose.
    """
    if not consentimiento:
        raise ValueError("Se requiere consentimiento informado explícito (--consentimiento) para capturar.")

    if pose not in POSES_VALIDAS:
        raise ValueError(f"Pose '{pose}' no válida. Opciones: {sorted(POSES_VALIDAS)}")

    if particion not in PARTICIONES_VALIDAS:
        raise ValueError(f"Partición '{particion}' no válida. Opciones: {sorted(PARTICIONES_VALIDAS)}")

    if not es_ensayo and es_manifiesto_propio(ruta_manifiesto):
        es_val_proto, err_proto = validar_coherencia_sesion_particion_propio(sesion, particion)
        if not es_val_proto:
            raise ValueError(err_proto)

    if detector is None:
        detector = crear_detector()

    capturador = capturador_custom or CapturadorCamara(indice=indice)
    manifiesto = ManifiestoDataset(ruta_manifiesto)

    titulo = f"Captura: {persona} | {sesion} | {pose} ({particion}) - ESPACIO: Capturar | ESC: Salir"
    capturas_realizadas = []

    print("\n=======================================================")
    print(" MODO DE CAPTURA FACIAL EN VIVO")
    print("=======================================================")
    print(f" Sujeto:       {persona} ({tipo_sujeto})")
    print(f" Sesión:       {sesion}")
    print(f" Pose:         {pose.upper()} (ATENCIÓN: perspectiva del SUJETO)")
    print(f" Partición:    {particion.upper()}")
    print(f" Carpeta base: {carpeta_destino}")
    print(f" Manifiesto:   {ruta_manifiesto}")
    print(f" Ensayo:       {'SÍ (carpeta de ensayos)' if es_ensayo else 'NO (sesión de producción)'}")
    print("-------------------------------------------------------")
    print(" Controles en la ventana:")
    print("   [ESPACIO] / [ENTER]: Capturar fotograma actual.")
    print("   [F]:                 Forzar captura conservando fallo de detección (para protocolo).")
    print("   [ESC]:               Terminar captura.")
    print("=======================================================\n")

    try:
        capturador.abrir()
        cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)

        fotos_tomadas = 0

        while fotos_tomadas < num_fotos_objetivo:
            ret, frame = capturador.leer()
            if not ret or frame is None:
                continue

            # Evaluar detección en fotograma original
            resultado_det = detector.detectar(frame, es_pre_recortada=False)

            # Dibujar copia display
            display = frame.copy()
            h_f, w_f = display.shape[:2]

            # Header superior
            cv2.rectangle(display, (0, 0), (w_f, 50), (25, 25, 25), -1)
            cv2.putText(
                display,
                f"Sujeto: {persona} | Sesion: {sesion} | Pose: {pose.upper()} (persp. sujeto)",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                display,
                f"Particion: {particion.upper()} | Fotos: {fotos_tomadas}/{num_fotos_objetivo} | [ESPACIO]: Capturar",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            # Recuadro según detección
            if resultado_det.estado == EstadoDeteccion.UN_ROSTRO:
                color_st = (0, 255, 0)
                msg_st = "ESTADO: UN_ROSTRO - Listo para capturar"
                if resultado_det.recuadro is not None:
                    rx, ry, rw, rh = resultado_det.recuadro
                    cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), color_st, 2)
            elif resultado_det.estado == EstadoDeteccion.SIN_ROSTRO:
                color_st = (0, 180, 255)
                msg_st = "ESTADO: SIN_ROSTRO - [F] para forzar fallo de pose"
            else:
                color_st = (0, 0, 255)
                msg_st = f"ESTADO: VARIOS_ROSTROS ({resultado_det.total_rostros_detectados})"

            cv2.rectangle(display, (0, h_f - 30), (w_f, h_f), (25, 25, 25), -1)
            cv2.putText(display, msg_st, (10, h_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_st, 2, cv2.LINE_AA)

            cv2.imshow(titulo, display)

            tecla = cv2.waitKey(30) & 0xFF
            if tecla == 27:  # Esc
                print("Captura interrumpida por el usuario.")
                break

            guardar = False
            forzar_fallo = False
            if tecla in (32, 13):  # Espacio o Enter
                if resultado_det.estado == EstadoDeteccion.UN_ROSTRO:
                    guardar = True
                    forzar_fallo = False
                else:
                    print(f"\n[RECHAZADO] No se puede guardar captura normal con estado '{resultado_det.estado.value}'. "
                          "Se requiere exactamente UN_ROSTRO. Si desea registrar deliberadamente este fallo, presione [F].")
            elif tecla in (ord('f'), ord('F')):
                if resultado_det.estado in (EstadoDeteccion.SIN_ROSTRO, EstadoDeteccion.VARIOS_ROSTROS):
                    guardar = True
                    forzar_fallo = True
                else:
                    print("\n[RECHAZADO] No se puede registrar fallo: el detector identificó un rostro único (UN_ROSTRO). "
                          "Para captura normal presione [ESPACIO].")

            if guardar:
                persona_limpia = sanitizar_componente_ruta(persona)
                sesion_limpia = sanitizar_componente_ruta(sesion)
                pose_limpia = sanitizar_componente_ruta(pose)

                # Calcular correlativo de foto que no colisione
                foto_num = 1
                while True:
                    id_propuesto = f"{persona_limpia}_{sesion_limpia}_{pose_limpia}_f{foto_num:02d}"
                    subcarpeta = carpeta_destino / persona_limpia / sesion_limpia / pose_limpia
                    ruta_img_propuesta = subcarpeta / f"{id_propuesto}.jpg"
                    # Verificar si existe en manifiesto o en disco
                    existe_en_manifiesto = any(r.identificador == id_propuesto for r in manifiesto.registros)
                    if not ruta_img_propuesta.exists() and not existe_en_manifiesto:
                        break
                    foto_num += 1

                anotacion_det = (
                    f"dims={w_f}x{h_f}; detector={resultado_det.estado.value}; "
                    f"forzado={'si' if forzar_fallo else 'no'}; procedencia=camara_droidcam"
                )

                reg = RegistroCaptura(
                    identificador=id_propuesto,
                    ruta=str(ruta_img_propuesta),
                    procedencia="camara_droidcam",
                    persona=persona,
                    tipo_sujeto=tipo_sujeto,
                    sesion=sesion,
                    pose=pose,
                    particion=particion,
                    consentimiento=True,
                    anotaciones=anotacion_det
                )

                # Guardado atómico del fotograma ORIGINAL (sin dibujos)
                guardar_fotograma_atomico(
                    frame_original=frame,
                    ruta_destino=ruta_img_propuesta,
                    registro=reg,
                    manifiesto=manifiesto
                )

                fotos_tomadas += 1
                capturas_realizadas.append({
                    "identificador": id_propuesto,
                    "ruta": str(ruta_img_propuesta),
                    "estado_detector": resultado_det.estado.value,
                    "dimensiones": (w_f, h_f),
                    "hash_sha256": reg.hash_contenido
                })
                print(f"[OK] Captura {fotos_tomadas}/{num_fotos_objetivo} guardada: {id_propuesto} (Detector={resultado_det.estado.value})")

                # Breve pausa visual de confirmación (flash verde)
                flash = display.copy()
                cv2.rectangle(flash, (0, 0), (w_f, h_f), (0, 255, 0), 10)
                cv2.imshow(titulo, flash)
                cv2.waitKey(250)

    finally:
        capturador.liberar()
        cv2.destroyAllWindows()
        print("Finalizado modo de captura. Recursos liberados.")

    return capturas_realizadas
