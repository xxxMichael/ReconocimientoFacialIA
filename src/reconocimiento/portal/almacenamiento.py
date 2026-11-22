"""
Gestor de almacenamiento concurrente y seguro para capturas del portal público.
Almacena imágenes crudas fuera de directorios web/estáticos, valida decodificación,
controla idempotencia y actualiza el registro local sin exponer rutas del sistema.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

POSES_PROTOCOLO = ["frontal", "izquierda", "derecha", "arriba", "abajo"]
DISTANCIAS_PROTOCOLO = ["cercana", "media", "lejana"]
TOTAL_COMBINACIONES_PROTOCOLO = len(POSES_PROTOCOLO) * len(DISTANCIAS_PROTOCOLO)  # 15
LIMITE_TAMANO_BYTES = 15 * 1024 * 1024  # 15 Megabytes


class GestorAlmacenamientoExterno:
    """Gestiona la persistencia segura e idempotente de las capturas de voluntarios externos."""

    def __init__(self, dir_base: Optional[Path] = None, raiz_proyecto: Optional[Path] = None):
        self.raiz = Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[3]
        if dir_base:
            self.dir_base = Path(dir_base)
        else:
            self.dir_base = self.raiz / "data" / "recepcion_externa"

        self.dir_fotos = self.dir_base / "fotos"
        self.ruta_registro_json = self.dir_base / "registro_recepcion.json"
        self.ruta_manifiesto_csv = self.dir_base / "manifiesto_recepcion.csv"

        self._lock = threading.RLock()
        self._mtime = 0.0
        self._asegurar_directorios()
        self._registros: Dict[str, Dict[str, Any]] = {}
        self.cargar_registros()

    def _asegurar_directorios(self) -> None:
        self.dir_base.mkdir(parents=True, exist_ok=True)
        self.dir_fotos.mkdir(parents=True, exist_ok=True)

    def _sincronizar_si_cambio(self) -> None:
        if not self.ruta_registro_json.exists():
            self._registros = {}
            self._mtime = 0.0
            return
        try:
            mtime = self.ruta_registro_json.stat().st_mtime
            if mtime != getattr(self, "_mtime", 0.0):
                with open(self.ruta_registro_json, "r", encoding="utf-8") as f:
                    self._registros = json.load(f)
                self._mtime = mtime
        except Exception:
            pass

    def cargar_registros(self) -> None:
        with self._lock:
            if not self.ruta_registro_json.exists():
                self._registros = {}
                self._mtime = 0.0
                return
            try:
                with open(self.ruta_registro_json, "r", encoding="utf-8") as f:
                    self._registros = json.load(f)
                self._mtime = self.ruta_registro_json.stat().st_mtime
            except Exception:
                self._registros = {}

    def guardar_registros(self) -> None:
        tmp_json = self.ruta_registro_json.with_suffix(".tmp")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(self._registros, f, indent=2, ensure_ascii=False)
        tmp_json.replace(self.ruta_registro_json)
        try:
            self._mtime = self.ruta_registro_json.stat().st_mtime
        except Exception:
            pass

        # Actualizar CSV de recepción
        tmp_csv = self.ruta_manifiesto_csv.with_suffix(".tmp")
        campos = [
            "identificador_captura", "identificador_sujeto", "pose", "distancia",
            "tipo_sujeto", "grupo", "particion", "consentimiento", "fecha_utc",
            "sha256", "bytes_tamano", "ancho", "alto", "archivo_relativo"
        ]
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            for k, reg in self._registros.items():
                writer.writerow({
                    "identificador_captura": k,
                    "identificador_sujeto": reg.get("identificador_sujeto"),
                    "pose": reg.get("pose"),
                    "distancia": reg.get("distancia"),
                    "tipo_sujeto": reg.get("tipo_sujeto", "desconocido"),
                    "grupo": reg.get("grupo", "desconocidos_prueba"),
                    "particion": reg.get("particion", "prueba"),
                    "consentimiento": reg.get("consentimiento", True),
                    "fecha_utc": reg.get("fecha_utc"),
                    "sha256": reg.get("sha256"),
                    "bytes_tamano": reg.get("bytes_tamano"),
                    "ancho": reg.get("ancho"),
                    "alto": reg.get("alto"),
                    "archivo_relativo": reg.get("archivo_relativo"),
                })
        tmp_csv.replace(self.ruta_manifiesto_csv)

    def obtener_progreso(self, identificador: str) -> Dict[str, Any]:
        """Calcula las cantidades enviadas, pendientes y matriz de combinaciones del participante."""
        id_limpio = str(identificador).strip().lower()

        with self._lock:
            self._sincronizar_si_cambio()
            combinaciones_estado = {}
            for p in POSES_PROTOCOLO:
                for d in DISTANCIAS_PROTOCOLO:
                    combinaciones_estado[f"{p}_{d}"] = False

            enviadas = 0
            for k, reg in self._registros.items():
                if reg.get("identificador_sujeto") == id_limpio:
                    clave_comb = f"{reg.get('pose')}_{reg.get('distancia')}"
                    if clave_comb in combinaciones_estado:
                        combinaciones_estado[clave_comb] = True
                        enviadas += 1

            pendientes = max(0, TOTAL_COMBINACIONES_PROTOCOLO - enviadas)
            completado = (enviadas >= TOTAL_COMBINACIONES_PROTOCOLO)

            return {
                "identificador": id_limpio,
                "total_requeridas": TOTAL_COMBINACIONES_PROTOCOLO,
                "total_enviadas": enviadas,
                "total_pendientes": pendientes,
                "completado": completado,
                "combinaciones": combinaciones_estado,
                "poses": POSES_PROTOCOLO,
                "distancias": DISTANCIAS_PROTOCOLO,
            }

    def guardar_captura(
        self,
        identificador: str,
        pose: str,
        distancia: str,
        datos_imagen: bytes,
        consentimiento_confirmado: bool = True
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Guarda una fotografía de forma atómica e idempotente.
        Retorna (exito, mensaje_usuario, progreso_actualizado).
        """
        id_limpio = str(identificador).strip().lower()
        pose_limpia = str(pose).strip().lower()
        dist_limpia = str(distancia).strip().lower()

        if not consentimiento_confirmado:
            return False, "Debe otorgar consentimiento explícito antes de enviar fotografías.", {}

        if pose_limpia not in POSES_PROTOCOLO:
            return False, f"Pose '{pose}' inválida. Válidas: {', '.join(POSES_PROTOCOLO)}", {}

        if dist_limpia not in DISTANCIAS_PROTOCOLO:
            return False, f"Distancia '{distancia}' inválida. Válidas: {', '.join(DISTANCIAS_PROTOCOLO)}", {}

        if not datos_imagen or len(datos_imagen) == 0:
            return False, "La imagen recibida está vacía.", {}

        if len(datos_imagen) > LIMITE_TAMANO_BYTES:
            tam_mb = round(len(datos_imagen) / (1024 * 1024), 2)
            return False, f"El archivo ({tam_mb} MB) excede el tamaño máximo permitido de 15 MB.", {}

        # Validar decodificación como imagen
        arr = np.frombuffer(datos_imagen, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False, "El archivo enviado no pudo ser decodificado como una imagen válida.", {}

        alto, ancho = img_bgr.shape[:2]
        if alto < 100 or ancho < 100:
            return False, f"La resolución de la imagen ({ancho}x{alto}) es insuficiente.", {}

        digest = hashlib.sha256(datos_imagen).hexdigest()
        clave_captura = f"{id_limpio}_{pose_limpia}_{dist_limpia}"

        with self._lock:
            self._sincronizar_si_cambio()
            # Control de idempotencia / duplicados
            if clave_captura in self._registros:
                reg_existente = self._registros[clave_captura]
                if reg_existente.get("sha256") == digest:
                    # Mismo contenido (reintento o doble clic): responder éxito idempotente
                    progreso = self.obtener_progreso(id_limpio)
                    return True, "Fotografía registrada previamente (envío confirmado).", progreso
                else:
                    return False, (
                        f"Ya existe una fotografía registrada para la combinación '{pose_limpia}' "
                        f"a distancia '{dist_limpia}'. Para corregirla contacte al administrador."
                    ), {}

            # Crear carpeta individual para el participante
            carpeta_sujeto = self.dir_fotos / id_limpio
            carpeta_sujeto.mkdir(parents=True, exist_ok=True)

            nombre_archivo = f"{clave_captura}.jpg"
            ruta_archivo = carpeta_sujeto / nombre_archivo

            # Guardar la imagen original en disco de forma atómica
            tmp_archivo = ruta_archivo.with_suffix(".tmp")
            with open(tmp_archivo, "wb") as f:
                f.write(datos_imagen)
            tmp_archivo.replace(ruta_archivo)

            # Ruta relativa al directorio de recepción (nunca se expone la ruta absoluta en la API)
            rel_archivo = f"fotos/{id_limpio}/{nombre_archivo}"

            # Registrar metadatos
            registro_nuevo = {
                "identificador_captura": clave_captura,
                "identificador_sujeto": id_limpio,
                "pose": pose_limpia,
                "distancia": dist_limpia,
                "tipo_sujeto": "desconocido",
                "grupo": "desconocidos_prueba",
                "particion": "prueba",
                "consentimiento": True,
                "fecha_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": digest,
                "bytes_tamano": len(datos_imagen),
                "ancho": int(ancho),
                "alto": int(alto),
                "archivo_relativo": rel_archivo,
            }

            self._registros[clave_captura] = registro_nuevo
            self.guardar_registros()

        progreso_actualizado = self.obtener_progreso(id_limpio)
        return True, "Fotografía guardada con éxito.", progreso_actualizado

    def listar_registros_sujeto(self, identificador: str) -> List[Dict[str, Any]]:
        """Retorna registros de un sujeto (para uso interno o importación)."""
        id_limpio = str(identificador).strip().lower()
        with self._lock:
            return [
                dict(v) for v in self._registros.values()
                if v.get("identificador_sujeto") == id_limpio
            ]

    def listar_todos_los_registros(self) -> List[Dict[str, Any]]:
        """Retorna todos los registros guardados (para el importador local)."""
        with self._lock:
            self._sincronizar_si_cambio()
            return [dict(v) for v in self._registros.values()]
