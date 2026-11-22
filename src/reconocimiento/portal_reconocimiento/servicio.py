"""
Servicio de inferencia en memoria y registro anónimo de intentos para el portal remoto.
- Carga ambos candidatos congelados una sola vez al iniciar.
- Acepta únicamente identificadores de candidato de una lista cerrada.
- Decodifica y procesa fotogramas estrictamente en memoria, sin guardar imágenes.
- Protege los detectores/modelos compartidos con locks y limita la cola concurrente.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Optional
import cv2
import numpy as np

from reconocimiento.clasificacion.pipeline_candidato import PipelineCandidatoCongelado

CANDIDATO_PRINCIPAL_ID = "cnn_estandar_portal_061"
CANDIDATOS_PORTAL: Dict[str, Dict[str, str]] = {
    "cnn_estandar_portal_061": {
        "nombre": "CNN Estándar Portal Completo — 0.61",
        "ruta_config": "configuracion_candidato_estandar_portal_061.json",
    },
    "cnn_estandar_portal_061_calidad": {
        "nombre": "CNN Estándar 0.61 + Filtro Calidad (Fase 1)",
        "ruta_config": "configuracion_candidato_estandar_portal_061_calidad.json",
    },
    "cnn_fase2_robustez": {
        "nombre": "CNN Fase 2 Robustez — 0.62 (Experimental)",
        "ruta_config": "configuracion_candidato_fase2_robustez.json",
    },
    "cnn_antialiasing_052": {
        "nombre": "CNN Antialiasing — 0.52",
        "ruta_config": "configuracion_candidato_antialiasing_052.json",
    },
}

CATEGORIAS_REALES_VALIDAS = frozenset({"persona_1", "persona_2", "persona_3", "desconocido"})


def _sanitizar_id(valor: str, por_defecto: str = "anonimo") -> str:
    limpio = re.sub(r"[^a-zA-Z0-9_\-]", "", (valor or "").strip())
    return limpio[:48] if limpio else por_defecto


class ServidorOcupadoError(RuntimeError):
    """Se lanza cuando la cola de inferencia concurrente está saturada."""


class ServicioPortalReconocimiento:
    """Gestiona los candidatos congelados en memoria y el registro de intentos sin fotos."""

    def __init__(
        self,
        raiz_proyecto: Optional[Path] = None,
        ruta_registro_intentos: Optional[Path] = None,
        max_inferencias_simultaneas: int = 4,
    ) -> None:
        self.raiz = Path(raiz_proyecto) if raiz_proyecto else Path.cwd()
        self.ruta_registro_intentos = (
            Path(ruta_registro_intentos)
            if ruta_registro_intentos
            else (self.raiz / "resultados/portal_reconocimiento/intentos.jsonl")
        )
        self.semaforo_cola = threading.BoundedSemaphore(max(1, int(max_inferencias_simultaneas)))
        self.pipelines: Dict[str, PipelineCandidatoCongelado] = {}
        self.locks_candidato: Dict[str, threading.Lock] = {}
        self._lock_registro = threading.Lock()
        self._registros_por_id: Dict[str, Dict[str, Any]] = {}

        # Cargar ambos candidatos congelados una sola vez al iniciar
        for cid, meta in CANDIDATOS_PORTAL.items():
            ruta_cfg = self.raiz / meta["ruta_config"]
            self.pipelines[cid] = PipelineCandidatoCongelado(ruta_cfg, raiz_proyecto=self.raiz)
            self.locks_candidato[cid] = threading.Lock()

        self._cargar_registros_existentes()

    def _cargar_registros_existentes(self) -> None:
        if not self.ruta_registro_intentos.exists():
            return
        try:
            with self.ruta_registro_intentos.open("r", encoding="utf-8") as f:
                for linea in f:
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        reg = json.loads(linea)
                        id_i = str(reg.get("id_intento", "")).strip()
                        if id_i and id_i not in self._registros_por_id:
                            self._registros_por_id[id_i] = reg
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    def listar_candidatos(self) -> list[Dict[str, Any]]:
        salida = []
        for cid, meta in CANDIDATOS_PORTAL.items():
            pipe = self.pipelines[cid]
            salida.append({
                "id": cid,
                "nombre": meta["nombre"],
                "umbral": round(float(pipe.umbral_score), 2),
                "umbral_pct": round(float(pipe.umbral_score) * 100.0, 1),
                "metodo_remuestreo": pipe.metodo_remuestreo,
                "sha256_modelo": pipe.sha256_modelo,
                "es_principal": cid == CANDIDATO_PRINCIPAL_ID,
            })
        return salida

    def procesar_fotograma_bytes(
        self,
        frame_bytes: bytes,
        candidato_id: str,
        frame_seq: int = 0,
    ) -> Dict[str, Any]:
        """Decodifica JPEG en memoria y ejecuta inferencia con el candidato seleccionado."""
        if candidato_id not in self.pipelines:
            raise ValueError(f"Candidato no permitido: '{candidato_id}'")
        if not frame_bytes:
            raise ValueError("Fotograma vacío.")

        adquirido = self.semaforo_cola.acquire(timeout=0.05)
        if not adquirido:
            raise ServidorOcupadoError("El servidor está ocupado procesando otras solicitudes. Reintenta en un momento.")

        t0 = time.perf_counter()
        try:
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None or frame_bgr.size == 0:
                raise ValueError("No se pudo decodificar el fotograma en memoria.")

            pipe = self.pipelines[candidato_id]
            lock = self.locks_candidato[candidato_id]
            with lock:
                res_det, dec, probs = pipe.evaluar_fotograma_completo(frame_bgr)

            latencia_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            clase_predicha = None
            score = 0.0
            if probs is not None and len(probs) > 0:
                idx = int(np.argmax(probs))
                clase_predicha = pipe.mapeo_id_clase.get(idx, "desconocido")
                score = round(float(probs[idx]), 4)

            return {
                "status": "ok",
                "frame_seq": int(frame_seq),
                "candidato_id": candidato_id,
                "candidato_nombre": CANDIDATOS_PORTAL[candidato_id]["nombre"],
                "sha256_modelo": pipe.sha256_modelo,
                "umbral": round(float(pipe.umbral_score), 2),
                "umbral_pct": round(float(pipe.umbral_score) * 100.0, 1),
                "estado_detector": res_det.estado.value,
                "clase_predicha": clase_predicha,
                "score": score,
                "score_pct": round(score * 100.0, 1),
                "decision_puerta": dec.estado_puerta,
                "es_autorizado": bool(dec.es_autorizado),
                "persona_identificada": dec.persona_identificada if dec.es_autorizado else "",
                "motivo": dec.motivo_cierre if dec.estado_puerta == "CERRADA" else dec.mensaje,
                "latencia_ms": latencia_ms,
            }
        finally:
            self.semaforo_cola.release()

    def registrar_intento(
        self,
        frame_bytes: bytes,
        candidato_id: str,
        categoria_real: str,
        participante_id: str,
        id_intento: str,
    ) -> Dict[str, Any]:
        """
        Evalúa un único fotograma en memoria y guarda el resultado estructurado del intento
        (sin almacenar la imagen ni permitir que `categoria_real` influya en la inferencia).
        """
        cat_limpia = (categoria_real or "").strip().lower()
        if cat_limpia not in CATEGORIAS_REALES_VALIDAS:
            raise ValueError(
                f"Categoría real inválida: '{categoria_real}'. Debe ser una de: {sorted(CATEGORIAS_REALES_VALIDAS)}"
            )

        id_int_limpio = _sanitizar_id(id_intento, por_defecto="")
        if not id_int_limpio:
            raise ValueError("Se requiere id_intento válido para evitar duplicados.")

        part_limpio = _sanitizar_id(participante_id, por_defecto="anonimo")

        # Si ya existe este id_intento (reintento de red), devolver registro original sin reevaluar ni duplicar
        with self._lock_registro:
            if id_int_limpio in self._registros_por_id:
                reg_orig = dict(self._registros_por_id[id_int_limpio])
                return {
                    "status": "ok",
                    "duplicado": True,
                    "id_intento": id_int_limpio,
                    "registro": reg_orig,
                    "mensaje": "Intento ya registrado previamente (registro original recuperado sin duplicar).",
                }

        # Ejecutar inferencia sin pasar categoria_real
        resultado = self.procesar_fotograma_bytes(frame_bytes, candidato_id=candidato_id, frame_seq=0)

        registro = {
            "id_intento": id_int_limpio,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "participante_id": part_limpio,
            "candidato_id": resultado["candidato_id"],
            "candidato_nombre": resultado["candidato_nombre"],
            "sha256_modelo": resultado["sha256_modelo"],
            "umbral": resultado["umbral"],
            "categoria_real": cat_limpia,
            "etiqueta_autodeclarada": True,
            "clase_predicha": resultado["clase_predicha"],
            "score": resultado["score"],
            "estado_detector": resultado["estado_detector"],
            "decision_puerta": resultado["decision_puerta"],
            "motivo": resultado["motivo"],
            "latencia_ms": resultado["latencia_ms"],
        }

        with self._lock_registro:
            if id_int_limpio in self._registros_por_id:
                reg_orig = dict(self._registros_por_id[id_int_limpio])
                return {
                    "status": "ok",
                    "duplicado": True,
                    "id_intento": id_int_limpio,
                    "registro": reg_orig,
                    "mensaje": "Intento ya registrado previamente (registro original recuperado sin duplicar).",
                }
            self.ruta_registro_intentos.parent.mkdir(parents=True, exist_ok=True)
            with self.ruta_registro_intentos.open("a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            self._registros_por_id[id_int_limpio] = dict(registro)

        return {
            "status": "ok",
            "duplicado": False,
            "id_intento": id_int_limpio,
            "registro": registro,
            "resultado": resultado,
        }
