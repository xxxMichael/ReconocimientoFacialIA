"""
Gestor de invitaciones individuales para el portal público de captura independiente.
Garantiza que cada enlace corresponda a un identificador anónimo controlado por el servidor
(ej. externo_001), previniendo que el cliente web altere su identidad o grupo de estudio.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Dict, List, Optional, Set

from reconocimiento.adquisicion.propio import PERSONAS_AUTORIZADAS_PROPIO


def es_identificador_valido(identificador: str) -> bool:
    """Valida formato seguro del identificador (ej. externo_001, voluntario_a)."""
    if not identificador or not isinstance(identificador, str):
        return False
    identificador = identificador.strip().lower()
    return bool(re.match(r"^[a-z0-9_-]{3,32}$", identificador))


class GestorInvitaciones:
    """Gestiona la emisión, revocación y validación de tokens de invitación únicos."""

    def __init__(self, ruta_archivo: Optional[Path] = None, raiz_proyecto: Optional[Path] = None):
        self.raiz = Path(raiz_proyecto) if raiz_proyecto else Path(__file__).resolve().parents[3]
        if ruta_archivo:
            self.ruta_archivo = Path(ruta_archivo)
        else:
            self.ruta_archivo = self.raiz / "data" / "recepcion_externa" / "invitaciones.json"

        self._lock = threading.RLock()
        self._mtime = 0.0
        self._asegurar_directorio()
        self._invitaciones: Dict[str, Dict[str, Any]] = {}
        self.cargar()

    def _asegurar_directorio(self) -> None:
        self.ruta_archivo.parent.mkdir(parents=True, exist_ok=True)

    def _sincronizar_si_cambio(self) -> None:
        """Recarga el archivo si fue modificado externamente (ej. otra instancia o proceso)."""
        if not self.ruta_archivo.exists():
            self._invitaciones = {}
            self._mtime = 0.0
            return
        try:
            mtime = self.ruta_archivo.stat().st_mtime
            if mtime != getattr(self, "_mtime", 0.0):
                with open(self.ruta_archivo, "r", encoding="utf-8") as f:
                    self._invitaciones = json.load(f)
                self._mtime = mtime
        except Exception:
            pass

    def cargar(self) -> None:
        with self._lock:
            if not self.ruta_archivo.exists():
                self._invitaciones = {}
                self._mtime = 0.0
                return
            try:
                with open(self.ruta_archivo, "r", encoding="utf-8") as f:
                    self._invitaciones = json.load(f)
                self._mtime = self.ruta_archivo.stat().st_mtime
            except Exception:
                self._invitaciones = {}

    def guardar(self) -> None:
        self._asegurar_directorio()
        tmp_file = self.ruta_archivo.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self._invitaciones, f, indent=2, ensure_ascii=False)
        tmp_file.replace(self.ruta_archivo)
        try:
            self._mtime = self.ruta_archivo.stat().st_mtime
        except Exception:
            pass

    def obtener_identidades_restringidas(self) -> Set[str]:
        """
        Retorna identidades que bajo ninguna circunstancia pueden emitirse como desconocido externo:
        - Personas autorizadas del sistema (persona_1, persona_2, persona_3).
        - Identidades existentes en particiones de entrenamiento o validación/calibración.
        """
        restringidas: Set[str] = set(PERSONAS_AUTORIZADAS_PROPIO)

        # Revisar manifiesto propio (local o principal)
        manifiesto_propio = self.raiz / "data" / "propio" / "manifiesto_local.csv"
        if not manifiesto_propio.exists():
            manifiesto_propio = self.raiz / "data" / "propio" / "manifiesto.csv"

        if manifiesto_propio.exists():
            try:
                import csv
                with open(manifiesto_propio, "r", encoding="utf-8") as f:
                    for fila in csv.DictReader(f):
                        p = fila.get("persona", "").strip().lower()
                        part = fila.get("particion", "").strip().lower()
                        if p and part in {"entrenamiento", "validacion"}:
                            restringidas.add(p)
            except Exception:
                pass

        # Revisar manifiestos de entrenamiento y calibración de desconocidos LFW
        lfw_calib = self.raiz / "data" / "desconocidos_lfw" / "calibracion" / "manifiesto.csv"
        lfw_train = self.raiz / "data" / "desconocidos_lfw" / "entrenamiento_4clases" / "manifiesto.csv"
        for ruta_lfw in (lfw_calib, lfw_train):
            if ruta_lfw.exists():
                try:
                    import csv
                    with open(ruta_lfw, "r", encoding="utf-8") as f:
                        for fila in csv.DictReader(f):
                            id_lfw = fila.get("identidad_normalizada", "").strip().lower()
                            if id_lfw:
                                restringidas.add(id_lfw)
                except Exception:
                    pass

        return restringidas

    def crear_invitacion(
        self,
        identificador: str,
        notas: str = "",
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crea una nueva invitación individual para un participante anónimo.
        Retorna el registro completo de la invitación incluyendo el enlace generado.
        """
        identificador_limpio = str(identificador).strip().lower()

        if not es_identificador_valido(identificador_limpio):
            raise ValueError(
                f"Identificador '{identificador}' inválido. Debe contener entre 3 y 32 caracteres "
                "alfanuméricos, guiones o guiones bajos (ej. 'externo_001')."
            )

        restringidas = self.obtener_identidades_restringidas()
        if identificador_limpio in restringidas:
            raise ValueError(
                f"Conflicto de aislamiento: El identificador '{identificador_limpio}' está restringido "
                "(es un sujeto autorizado o pertenece a entrenamiento/calibración) y no puede utilizarse "
                "para desconocidos de prueba externa."
            )

        with self._lock:
            # Comprobar si ya existe una invitación activa para este identificador
            for token_existente, inv in self._invitaciones.items():
                if inv.get("identificador") == identificador_limpio and inv.get("activo"):
                    url = f"{base_url.rstrip('/')}/participar/{token_existente}" if base_url else f"/participar/{token_existente}"
                    inv_ret = dict(inv)
                    inv_ret["url"] = url
                    return inv_ret

            nuevo_token = secrets.token_urlsafe(16)
            ahora_iso = datetime.now(timezone.utc).isoformat()

            registro = {
                "token": nuevo_token,
                "identificador": identificador_limpio,
                "tipo_sujeto": "desconocido",
                "grupo": "desconocidos_prueba",
                "particion": "prueba",
                "activo": True,
                "creado_en": ahora_iso,
                "consentimiento_registrado": False,
                "consentimiento_timestamp": None,
                "notas": notas.strip() if notas else "",
            }

            self._invitaciones[nuevo_token] = registro
            self.guardar()

            inv_ret = dict(registro)
            inv_ret["url"] = f"{base_url.rstrip('/')}/participar/{nuevo_token}" if base_url else f"/participar/{nuevo_token}"
            return inv_ret

    def desactivar_invitacion(self, token_o_identificador: str) -> bool:
        """Desactiva/revoca una invitación mediante su token o su identificador de sujeto."""
        clave = str(token_o_identificador).strip()
        modificado = False
        with self._lock:
            # Buscar por token
            if clave in self._invitaciones:
                self._invitaciones[clave]["activo"] = False
                modificado = True
            else:
                # Buscar por identificador
                id_limpio = clave.lower()
                for tok, inv in self._invitaciones.items():
                    if inv.get("identificador") == id_limpio:
                        inv["activo"] = False
                        modificado = True

            if modificado:
                self.guardar()
        return modificado

    def activar_invitacion(self, token_o_identificador: str) -> bool:
        """Reactiva una invitación previamente desactivada."""
        clave = str(token_o_identificador).strip()
        modificado = False
        with self._lock:
            if clave in self._invitaciones:
                self._invitaciones[clave]["activo"] = True
                modificado = True
            else:
                id_limpio = clave.lower()
                for tok, inv in self._invitaciones.items():
                    if inv.get("identificador") == id_limpio:
                        inv["activo"] = True
                        modificado = True

            if modificado:
                self.guardar()
        return modificado

    def obtener_invitacion(self, token: str) -> Optional[Dict[str, Any]]:
        """Obtiene una invitación por token si existe y está activa."""
        if not token:
            return None
        tok_limpio = str(token).strip()
        with self._lock:
            self._sincronizar_si_cambio()
            inv = self._invitaciones.get(tok_limpio)
            if inv:
                return dict(inv)
        return None

    def registrar_consentimiento(self, token: str) -> bool:
        """Registra el consentimiento explícito del participante con marca de tiempo UTC."""
        tok_limpio = str(token).strip()
        with self._lock:
            self._sincronizar_si_cambio()
            inv = self._invitaciones.get(tok_limpio)
            if not inv or not inv.get("activo"):
                return False
            inv["consentimiento_registrado"] = True
            inv["consentimiento_timestamp"] = datetime.now(timezone.utc).isoformat()
            self.guardar()
            return True

    def listar_invitaciones(self, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna la lista de todas las invitaciones registradas."""
        with self._lock:
            self._sincronizar_si_cambio()
            salida = []
            for tok, inv in self._invitaciones.items():
                copia = dict(inv)
                url = f"{base_url.rstrip('/')}/participar/{tok}" if base_url else f"/participar/{tok}"
                copia["url"] = url
                salida.append(copia)
            return sorted(salida, key=lambda x: x.get("creado_en", ""), reverse=True)
