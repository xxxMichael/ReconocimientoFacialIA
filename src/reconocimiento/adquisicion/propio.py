"""
Módulo de adquisición para imágenes propias y gestión del manifiesto de captura.
Implementa el protocolo de 3 personas x 5 sesiones x 5 poses x 4 fotos = 300 fotos,
y valida el aislamiento estricto de sesiones entre particiones (proporción 3/1/1).
"""
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


POSES_VALIDAS = {"frontal", "izquierda", "derecha", "arriba", "abajo"}
PARTICIONES_VALIDAS = {"entrenamiento", "validacion", "prueba"}
TIPOS_SUJETO_VALIDOS = {"autorizado", "desconocido"}


def calcular_hash_archivo(ruta: Path) -> str:
    """Calcula el hash SHA-256 del contenido de un archivo en disco."""
    p = Path(ruta)
    if not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


DECISIONES_REVISION_VALIDAS = {"pendiente", "aprobada", "corregida", "excluida"}


@dataclass
class RegistroCaptura:
    """Metadatos completos de una imagen propia capturada."""
    identificador: str
    ruta: str
    procedencia: str
    persona: str
    tipo_sujeto: str  # 'autorizado' o 'desconocido'
    sesion: str
    pose: str  # 'frontal', 'izquierda', 'derecha', 'arriba', 'abajo'
    particion: str  # 'entrenamiento', 'validacion', 'prueba'
    consentimiento: bool = False
    anotaciones: str = ""
    hash_contenido: str = ""

    def obtener_metadatos_anotaciones(self) -> Dict[str, str]:
        """Parsea los pares clave=valor contenidos en anotaciones."""
        meta: Dict[str, str] = {}
        if not self.anotaciones:
            return meta
        partes = [p.strip() for p in self.anotaciones.split(";") if p.strip()]
        for p in partes:
            if "=" in p:
                k, v = p.split("=", 1)
                meta[k.strip().lower()] = v.strip()
        return meta

    @property
    def decision_humana(self) -> str:
        """Decisión humana de revisión: 'pendiente', 'aprobada', 'corregida' o 'excluida'."""
        meta = self.obtener_metadatos_anotaciones()
        dec = meta.get("decision", "").lower()
        if dec in DECISIONES_REVISION_VALIDAS:
            return dec
        return "pendiente"

    @property
    def estado_historico_detector(self) -> str:
        """Estado registrado por el detector facial en el momento de la captura."""
        meta = self.obtener_metadatos_anotaciones()
        return meta.get("estado", "")

    @property
    def caja_aprobada(self) -> Optional[Tuple[int, int, int, int]]:
        """Recuadro validado o corregido por el revisor humano (x, y, w, h)."""
        meta = self.obtener_metadatos_anotaciones()
        caja_str = meta.get("caja", "")
        if caja_str:
            try:
                partes = [int(float(x.strip())) for x in caja_str.split(",")]
                if len(partes) == 4:
                    return (partes[0], partes[1], partes[2], partes[3])
            except Exception:
                pass
        return None

    @property
    def caja_historica(self) -> Optional[Tuple[int, int, int, int]]:
        """Recuadro registrado originalmente al momento de la captura en vivo."""
        meta = self.obtener_metadatos_anotaciones()
        caja_str = meta.get("caja_historica", "")
        if caja_str:
            try:
                partes = [int(float(x.strip())) for x in caja_str.split(",")]
                if len(partes) == 4:
                    return (partes[0], partes[1], partes[2], partes[3])
            except Exception:
                pass
        return None

    @property
    def origen_caja(self) -> str:
        """Origen del recuadro actual: 'vivo', 'recalculada' o 'manual'."""
        meta = self.obtener_metadatos_anotaciones()
        return meta.get("origen_caja", "")

    def actualizar_revision(
        self,
        decision: str,
        caja: Optional[Tuple[int, int, int, int]] = None,
        origen_caja: str = "manual"
    ) -> None:
        """
        Actualiza la decisión de revisión humana y el recuadro aprobado/corregido
        conservando intactos el estado histórico del detector y los demás metadatos.
        """
        decision = decision.strip().lower()
        if decision not in DECISIONES_REVISION_VALIDAS:
            raise ValueError(f"Decisión inválida '{decision}'. Válidas: {sorted(DECISIONES_REVISION_VALIDAS)}")

        partes_existentes = [p.strip() for p in self.anotaciones.split(";") if p.strip()]
        claves_a_reemplazar = {"decision", "caja", "origen_caja"}
        nuevas_partes = []

        for p in partes_existentes:
            if "=" in p:
                k = p.split("=", 1)[0].strip().lower()
                if k in claves_a_reemplazar:
                    continue
            nuevas_partes.append(p)

        nuevas_partes.append(f"decision={decision}")
        if caja is not None and len(caja) == 4:
            x, y, w, h = (int(caja[0]), int(caja[1]), int(caja[2]), int(caja[3]))
            nuevas_partes.append(f"caja={x},{y},{w},{h}")
            nuevas_partes.append(f"origen_caja={origen_caja}")

        self.anotaciones = ";".join(nuevas_partes)

    def validar(self) -> None:
        """Verifica que los campos contengan valores dentro del dominio acordado."""
        if not self.identificador:
            raise ValueError("El identificador no puede estar vacío.")
        if self.pose not in POSES_VALIDAS:
            raise ValueError(f"Pose '{self.pose}' inválida. Opciones: {sorted(POSES_VALIDAS)}")
        if self.particion not in PARTICIONES_VALIDAS:
            raise ValueError(f"Partición '{self.particion}' inválida. Opciones: {sorted(PARTICIONES_VALIDAS)}")
        if self.tipo_sujeto not in TIPOS_SUJETO_VALIDOS:
            raise ValueError(f"Tipo de sujeto '{self.tipo_sujeto}' inválido. Opciones: {sorted(TIPOS_SUJETO_VALIDOS)}")
        if not self.consentimiento:
            raise ValueError("No se puede registrar una captura sin consentimiento informado explícito.")

        # Calcular hash de contenido automáticamente si el archivo existe en disco y no se suministró hash
        if not self.hash_contenido and self.ruta:
            p = Path(self.ruta)
            if p.is_file():
                self.hash_contenido = calcular_hash_archivo(p)


class ManifiestoDataset:
    """Gestor del archivo de manifiesto para el dataset de imágenes propias."""

    def __init__(self, ruta_archivo: Optional[Path] = None):
        self.ruta_archivo = Path(ruta_archivo) if ruta_archivo else Path("data/propio/manifiesto.csv")
        self.registros: List[RegistroCaptura] = []
        if self.ruta_archivo.exists():
            self.cargar()

    def identificadores_existentes(self) -> Set[str]:
        """Retorna el conjunto de identificadores ya registrados."""
        return {r.identificador for r in self.registros}

    def agregar_registro(self, registro: RegistroCaptura) -> None:
        registro.validar()
        # Verificar identificador único
        for r in self.registros:
            if r.identificador == registro.identificador:
                raise ValueError(f"Identificador duplicado: {registro.identificador}")

            # Asegurar que ambos tengan hash si sus archivos existen en disco
            if not r.hash_contenido and r.ruta and Path(r.ruta).is_file():
                r.hash_contenido = calcular_hash_archivo(Path(r.ruta))
            if not registro.hash_contenido and registro.ruta and Path(registro.ruta).is_file():
                registro.hash_contenido = calcular_hash_archivo(Path(registro.ruta))

            # Verificar duplicidad de contenido entre particiones
            if registro.hash_contenido and r.hash_contenido:
                if registro.hash_contenido == r.hash_contenido and registro.particion != r.particion:
                    raise ValueError(
                        f"Fuga de imagen detectada: El archivo '{registro.ruta}' ({registro.identificador}) "
                        f"tiene contenido idéntico a '{r.ruta}' ({r.identificador}), "
                        f"pero están en particiones distintas ('{registro.particion}' vs '{r.particion}'). "
                        f"Rechazado para evitar inflar la exactitud del modelo."
                    )
        self.registros.append(registro)

    def guardar(self) -> None:
        """
        Escribe el manifiesto de forma atómica:
        1. Escribe en un archivo temporal en el mismo directorio.
        2. Hace flush y cierra el descriptor.
        3. Reemplaza atómicamente el archivo final para evitar corrupciones o archivos incompletos.
        """
        self.ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        campos = [
            "identificador", "ruta", "procedencia", "persona", "tipo_sujeto",
            "sesion", "pose", "particion", "consentimiento", "hash_contenido", "anotaciones"
        ]
        ruta_tmp = self.ruta_archivo.with_name(f"{self.ruta_archivo.name}.tmp")
        try:
            with open(ruta_tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=campos)
                writer.writeheader()
                for r in self.registros:
                    writer.writerow(asdict(r))
                f.flush()
            ruta_tmp.replace(self.ruta_archivo)
        except Exception as e:
            if ruta_tmp.exists():
                ruta_tmp.unlink(missing_ok=True)
            raise e

    def cargar(self) -> None:
        self.registros.clear()
        if not self.ruta_archivo.exists():
            return
        with open(self.ruta_archivo, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                registro = RegistroCaptura(
                    identificador=row["identificador"],
                    ruta=row["ruta"],
                    procedencia=row["procedencia"],
                    persona=row["persona"],
                    tipo_sujeto=row["tipo_sujeto"],
                    sesion=row["sesion"],
                    pose=row["pose"],
                    particion=row["particion"],
                    consentimiento=row["consentimiento"].lower() in {"true", "1", "si", "yes"},
                    anotaciones=row.get("anotaciones", ""),
                    hash_contenido=row.get("hash_contenido", "")
                )
                registro.validar()
                self.registros.append(registro)

    def buscar_registro(self, identificador: str) -> Optional[RegistroCaptura]:
        """Busca un registro por su identificador único."""
        for r in self.registros:
            if r.identificador == identificador:
                return r
        return None

    def actualizar_revision_captura(
        self,
        identificador: str,
        decision: str,
        caja: Optional[Tuple[int, int, int, int]] = None,
        origen_caja: str = "manual"
    ) -> bool:
        """
        Actualiza atómicamente la decisión humana de revisión y su recuadro.
        Conserva el original y el estado histórico del detector.
        """
        reg = self.buscar_registro(identificador)
        if reg is None:
            return False
        reg.actualizar_revision(decision=decision, caja=caja, origen_caja=origen_caja)
        self.guardar()
        return True

    def validar_aislamiento_sesiones(self) -> Tuple[bool, List[str]]:
        """
        Garantiza que ninguna sesión de una misma persona cruce de partición.
        Retorna (es_valido, lista_errores).
        """
        errores = []
        # Mapeo: (persona, sesion) -> set(particiones)
        sesiones_particion: Dict[Tuple[str, str], Set[str]] = {}

        for r in self.registros:
            clave = (r.persona, r.sesion)
            if clave not in sesiones_particion:
                sesiones_particion[clave] = set()
            sesiones_particion[clave].add(r.particion)

        for (persona, sesion), parts in sesiones_particion.items():
            if len(parts) > 1:
                errores.append(
                    f"Fuga de sesión detectada: Persona '{persona}', Sesión '{sesion}' "
                    f"aparece en múltiples particiones: {sorted(parts)}"
                )

        return (len(errores) == 0, errores)

    def validar_duplicados_entre_particiones(self) -> Tuple[bool, List[str]]:
        """
        Garantiza que ninguna imagen física idéntica (por hash SHA-256) se encuentre
        presente en más de una partición (ej. en entrenamiento y en prueba),
        incluso si los archivos tienen nombres o rutas distintas.
        """
        errores = []
        hashes_por_particion: Dict[str, Dict[str, List[str]]] = {}

        for r in self.registros:
            h = r.hash_contenido
            if not h and r.ruta and Path(r.ruta).is_file():
                h = calcular_hash_archivo(Path(r.ruta))
                r.hash_contenido = h

            if h:
                if h not in hashes_por_particion:
                    hashes_por_particion[h] = {}
                if r.particion not in hashes_por_particion[h]:
                    hashes_por_particion[h][r.particion] = []
                hashes_por_particion[h][r.particion].append(f"{r.identificador} ({r.ruta})")

        for h, parts in hashes_por_particion.items():
            if len(parts) > 1:
                detalles = "; ".join([f"{part}: {', '.join(ids)}" for part, ids in parts.items()])
                errores.append(
                    f"Fuga de imagen detectada: Hash SHA-256 '{h[:12]}...' repetido en particiones distintas: [{detalles}]"
                )

        return (len(errores) == 0, errores)

    def resumen_conteo(self) -> Dict[str, any]:
        """Genera conteo de muestras por persona, sesión, pose y partición."""
        por_particion = {"entrenamiento": 0, "validacion": 0, "prueba": 0}
        por_persona: Dict[str, int] = {}
        por_pose: Dict[str, int] = {}
        desconocidos = 0

        for r in self.registros:
            por_particion[r.particion] = por_particion.get(r.particion, 0) + 1
            por_persona[r.persona] = por_persona.get(r.persona, 0) + 1
            por_pose[r.pose] = por_pose.get(r.pose, 0) + 1
            if r.tipo_sujeto == "desconocido":
                desconocidos += 1

        return {
            "total_muestras": len(self.registros),
            "por_particion": por_particion,
            "por_persona": por_persona,
            "por_pose": por_pose,
            "desconocidos": desconocidos
        }

    def verificar_protocolo_completo(self) -> Dict[str, Any]:
        """
        Distingue claramente:
        1. Aislamiento de sesiones (condición obligatoria de integridad).
        2. Control de duplicados físicos entre particiones (condición obligatoria contra fuga de datos).
        3. Cumplimiento formal del reparto 3/1/1 por persona (3 personas x 5 sesiones x 5 poses x 4 fotos = 300 fotos).
        4. Distinción entre procedencia confirmada (cámara en vivo DroidCam/Webcam) e integración sintética/piloto.
        """
        aisl_ok, err_aisl = self.validar_aislamiento_sesiones()
        dup_ok, err_dup = self.validar_duplicados_entre_particiones()

        # Análisis por persona autorizada
        personas_autorizadas = sorted(list(set(r.persona for r in self.registros if r.tipo_sujeto == "autorizado")))
        desglose_personas = {}

        for p in personas_autorizadas:
            muestras_p = [r for r in self.registros if r.persona == p]
            # Mapeo sesion -> particion
            ses_train = set(r.sesion for r in muestras_p if r.particion == "entrenamiento")
            ses_val = set(r.sesion for r in muestras_p if r.particion == "validacion")
            ses_test = set(r.sesion for r in muestras_p if r.particion == "prueba")
            poses_presentes = set(r.pose for r in muestras_p)

            cumple_3_1_1 = (len(ses_train) == 3 and len(ses_val) == 1 and len(ses_test) == 1)
            cumple_5_poses = (poses_presentes == POSES_VALIDAS)
            total_fotos = len(muestras_p)
            cumple_100_fotos = (total_fotos == 100)

            desglose_personas[p] = {
                "total_muestras": total_fotos,
                "sesiones_entrenamiento": sorted(list(ses_train)),
                "sesiones_validacion": sorted(list(ses_val)),
                "sesiones_prueba": sorted(list(ses_test)),
                "cumple_reparto_3_1_1": cumple_3_1_1,
                "poses_registradas": sorted(list(poses_presentes)),
                "cumple_5_poses": cumple_5_poses,
                "protocolo_persona_completo": (cumple_3_1_1 and cumple_5_poses and cumple_100_fotos)
            }

        # Conteo de procedencia
        procedencias: Dict[str, int] = {}
        for r in self.registros:
            procedencias[r.procedencia] = procedencias.get(r.procedencia, 0) + 1

        camara_confirmada = any(
            k in {"camara_droidcam", "camara_web", "camara"} for k in procedencias.keys()
        )

        n_personas = len(personas_autorizadas)
        todas_personas_completas = (
            n_personas >= 3 and all(d["protocolo_persona_completo"] for d in desglose_personas.values())
        )
        protocolo_formal_300 = (
            aisl_ok and dup_ok and todas_personas_completas and len(self.registros) >= 300
        )

        es_piloto_parcial_valido = (
            aisl_ok and dup_ok and not protocolo_formal_300 and len(self.registros) > 0
        )

        return {
            "aislamiento_sesiones_valido": aisl_ok,
            "errores_aislamiento": err_aisl,
            "sin_duplicados_entre_particiones": dup_ok,
            "errores_duplicados": err_dup,
            "protocolo_formal_300_completo": protocolo_formal_300,
            "es_piloto_parcial_valido": es_piloto_parcial_valido,
            "total_registros": len(self.registros),
            "personas_autorizadas_evaluadas": n_personas,
            "desglose_por_persona": desglose_personas,
            "procedencias_detectadas": procedencias,
            "procedencia_camara_confirmada": camara_confirmada,
            "estado_dataset": (
                "EVALUACION_FORMAL_300" if protocolo_formal_300
                else ("PILOTO_PARCIAL_VALIDO" if es_piloto_parcial_valido else "INVALIDO_CON_ERRORES")
            )
        }


def generar_manifiesto_plantilla_300() -> ManifiestoDataset:
    """
    Genera la plantilla del plan acordado:
    3 personas autorizadas x 5 sesiones x 5 poses x 4 fotos = 300 imágenes.
    Proporción 3/1/1:
    - Sesiones 1, 2, 3 -> entrenamiento (60%, 180 fotos)
    - Sesión 4         -> validación (20%, 60 fotos)
    - Sesión 5         -> prueba (20%, 60 fotos)
    """
    manifiesto = ManifiestoDataset(Path("data/propio/manifiesto_plantilla_300.csv"))
    manifiesto.registros.clear()

    personas = ["persona_1", "persona_2", "persona_3"]
    sesiones_map = {
        "sesion_1": "entrenamiento",
        "sesion_2": "entrenamiento",
        "sesion_3": "entrenamiento",
        "sesion_4": "validacion",
        "sesion_5": "prueba",
    }
    poses = ["frontal", "izquierda", "derecha", "arriba", "abajo"]

    for persona in personas:
        for sesion, particion in sesiones_map.items():
            for pose in poses:
                for foto in range(1, 5):
                    id_muestra = f"{persona}_{sesion}_{pose}_f{foto:02d}"
                    ruta = f"data/propio/imagenes/{persona}/{sesion}/{pose}/{id_muestra}.jpg"
                    registro = RegistroCaptura(
                        identificador=id_muestra,
                        ruta=ruta,
                        procedencia="camara_web",
                        persona=persona,
                        tipo_sujeto="autorizado",
                        sesion=sesion,
                        pose=pose,
                        particion=particion,
                        consentimiento=True,
                        anotaciones="Plantilla plan 300 fotos"
                    )
                    manifiesto.agregar_registro(registro)

    # Añadir 20 casos de voluntarios desconocidos para pruebas de apertura
    for desc_idx in range(1, 6):
        desc_id = f"desconocido_{desc_idx:02d}"
        for pose in poses:
            id_muestra = f"{desc_id}_s01_{pose}_f01"
            ruta = f"data/propio/imagenes/desconocidos/{desc_id}/{id_muestra}.jpg"
            registro = RegistroCaptura(
                identificador=id_muestra,
                ruta=ruta,
                procedencia="camara_web",
                persona=desc_id,
                tipo_sujeto="desconocido",
                sesion="sesion_desc",
                pose=pose,
                particion="prueba",
                consentimiento=True,
                anotaciones="Voluntario no autorizado para prueba de rechazo"
            )
            manifiesto.agregar_registro(registro)

    manifiesto.guardar()
    return manifiesto
