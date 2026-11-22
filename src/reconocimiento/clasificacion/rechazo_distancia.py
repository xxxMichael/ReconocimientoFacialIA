"""
Módulo para el rechazo explícito por distancia a representaciones de entrenamiento.

Proporciona:
1. BancoReferencias: Almacén de vectores de entrenamiento por clase (ajustado EXCLUSIVAMENTE con sesiones 01–03).
2. Cálculo de métricas de distancia (mínima al cluster, centroide, euclidiana y coseno).
3. Función unificada evaluar_decision_acceso_con_distancia que implementa la regla conjunta:
   - Detección válida (UN_ROSTRO).
   - Sujeto autorizado.
   - Puntaje >= umbral_confianza.
   - Distancia a la clase predicha <= umbral_distancia.
   - Validación exhaustiva de entradas (NaN, Inf, normas cero, configuración parcial, etc.).
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from reconocimiento.clasificacion.regla_acceso import (
    DecisionAcceso,
    ETIQUETAS_DESCONOCIDO_RESERVADAS
)
from reconocimiento.preprocesamiento.detector import EstadoDeteccion


METRICAS_VALIDAS = {"min_cos", "cent_cos", "min_l2", "cent_l2"}


def _es_flotante_valido(valor: Any) -> bool:
    """Verifica si un valor es un flotante finito (no NaN, no Inf, no None)."""
    if valor is None:
        return False
    try:
        f = float(valor)
        return math.isfinite(f)
    except (TypeError, ValueError, OverflowError):
        return False


class BancoReferencias:
    """
    Almacén de vectores de representación de entrenamiento por identidad autorizada.
    Ajustado exclusivamente sobre las sesiones de entrenamiento (01–03).
    """

    def __init__(
        self,
        referencias_por_clase: Dict[str, np.ndarray],
        metadatos: Optional[Dict[str, Any]] = None
    ):
        self.referencias_por_clase: Dict[str, np.ndarray] = {}
        self.centroides_por_clase: Dict[str, np.ndarray] = {}
        self.metadatos: Dict[str, Any] = metadatos or {}

        for clase, matriz in referencias_por_clase.items():
            arr = np.asarray(matriz, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"La matriz de referencias para '{clase}' debe ser 2D (N, D), obtenida: {arr.shape}")
            if arr.shape[0] == 0:
                raise ValueError(f"La clase '{clase}' no contiene vectores de referencia.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"La matriz de referencias para '{clase}' contiene valores no finitos (NaN o Inf).")
            self.referencias_por_clase[clase] = arr
            self.centroides_por_clase[clase] = np.mean(arr, axis=0)

    @property
    def clases(self) -> List[str]:
        return sorted(self.referencias_por_clase.keys())

    @property
    def dimension(self) -> int:
        for m in self.referencias_por_clase.values():
            return int(m.shape[1])
        return 0

    @property
    def total_vectores(self) -> int:
        return sum(len(m) for m in self.referencias_por_clase.values())

    @property
    def memoria_ram_bytes(self) -> int:
        return sum(m.nbytes for m in self.referencias_por_clase.values()) + sum(c.nbytes for c in self.centroides_por_clase.values())

    def calcular_distancia(
        self,
        vector: Optional[np.ndarray],
        clase: str,
        metrica: str = "min_cos"
    ) -> float:
        """
        Calcula la distancia entre un vector de prueba y las referencias de la clase indicada.
        Lanza excepciones explícitas si el vector es inválido o la clase carece de referencias.
        """
        if metrica not in METRICAS_VALIDAS:
            raise ValueError(f"Métrica desconocida '{metrica}'. Válidas: {sorted(METRICAS_VALIDAS)}")

        if vector is None:
            raise ValueError("VECTOR_NULO")

        v = np.asarray(vector, dtype=np.float32).ravel()
        if v.size == 0:
            raise ValueError("VECTOR_NULO")

        if not np.all(np.isfinite(v)):
            raise ValueError("VALOR_NO_FINITO")

        if clase not in self.referencias_por_clase or len(self.referencias_por_clase[clase]) == 0:
            raise KeyError(f"CLASE_SIN_REFERENCIAS ({clase})")

        ref = self.referencias_por_clase[clase]
        if v.shape[0] != ref.shape[1]:
            raise ValueError(
                f"DIMENSION_INCOMPATIBLE: vector tiene {v.shape[0]}, "
                f"referencias de '{clase}' tienen {ref.shape[1]}."
            )

        cent = self.centroides_por_clase[clase]

        if metrica == "min_l2":
            dists = np.linalg.norm(ref - v, axis=1)
            dist_min = float(np.min(dists))
            if not math.isfinite(dist_min):
                raise ValueError("DISTANCIA_NO_FINITA")
            return max(0.0, dist_min)

        elif metrica == "cent_l2":
            dist_cent = float(np.linalg.norm(cent - v))
            if not math.isfinite(dist_cent):
                raise ValueError("DISTANCIA_NO_FINITA")
            return max(0.0, dist_cent)

        elif metrica == "min_cos":
            norm_v = float(np.linalg.norm(v))
            if norm_v == 0.0 or not math.isfinite(norm_v):
                raise ValueError("VECTOR_NORMA_INVALIDA")
            norm_ref = np.linalg.norm(ref, axis=1)
            if np.any(norm_ref == 0.0) or not np.all(np.isfinite(norm_ref)):
                raise ValueError("REFERENCIA_NORMA_INVALIDA")
            sims = np.dot(ref, v) / (norm_ref * norm_v)
            sims = np.clip(sims, -1.0, 1.0)
            dist_cos = float(1.0 - np.max(sims))
            if not math.isfinite(dist_cos):
                raise ValueError("DISTANCIA_NO_FINITA")
            return max(0.0, dist_cos)

        elif metrica == "cent_cos":
            norm_v = float(np.linalg.norm(v))
            if norm_v == 0.0 or not math.isfinite(norm_v):
                raise ValueError("VECTOR_NORMA_INVALIDA")
            norm_cent = float(np.linalg.norm(cent))
            if norm_cent == 0.0 or not math.isfinite(norm_cent):
                raise ValueError("CENTROIDE_NORMA_INVALIDA")
            sim = float(np.dot(cent, v) / (norm_cent * norm_v))
            sim = float(np.clip(sim, -1.0, 1.0))
            dist_cos = float(1.0 - sim)
            if not math.isfinite(dist_cos):
                raise ValueError("DISTANCIA_NO_FINITA")
            return max(0.0, dist_cos)

        raise ValueError(f"Métrica no manejada: {metrica}")

    def serializar(self) -> Dict[str, Any]:
        """Convierte el banco a formato JSON compatible para auditoría y trazabilidad."""
        datos = {
            "clases": self.clases,
            "dimension": self.dimension,
            "total_vectores": self.total_vectores,
            "memoria_ram_bytes": self.memoria_ram_bytes,
            "conteo_por_clase": {c: len(m) for c, m in self.referencias_por_clase.items()},
            "metadatos": self.metadatos,
            "referencias": {c: m.tolist() for c, m in self.referencias_por_clase.items()},
            "centroides": {c: self.centroides_por_clase[c].tolist() for c in self.clases},
        }
        bytes_canonicos = json.dumps(datos["referencias"], sort_keys=True).encode("utf-8")
        datos["sha256_referencias"] = hashlib.sha256(bytes_canonicos).hexdigest()
        return datos

    def guardar_npz(self, ruta_archivo: Path) -> str:
        """Guarda los arrays de referencia comprimidos y retorna el hash SHA-256."""
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        arrays_dict = {f"ref_{c}": m for c, m in self.referencias_por_clase.items()}
        np.savez_compressed(ruta_archivo, **arrays_dict)
        contenido = ruta_archivo.read_bytes()
        sha = hashlib.sha256(contenido).hexdigest()

        # Guardar metadatos JSON complementarios
        ruta_meta = ruta_archivo.with_suffix(".json")
        meta = {
            "clases": self.clases,
            "dimension": self.dimension,
            "total_vectores": self.total_vectores,
            "memoria_ram_bytes": self.memoria_ram_bytes,
            "conteo_por_clase": {c: len(m) for c, m in self.referencias_por_clase.items()},
            "sha256_npz": sha,
            "metadatos": self.metadatos,
        }
        ruta_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return sha

    @classmethod
    def cargar_npz(cls, ruta_archivo: Path) -> "BancoReferencias":
        """Carga un banco de referencias guardado en formato NPZ y valida su integridad por hash y metadatos."""
        ruta_archivo = Path(ruta_archivo)
        if not ruta_archivo.is_file():
            raise FileNotFoundError(f"No existe el archivo de banco: {ruta_archivo}")

        ruta_meta = ruta_archivo.with_suffix(".json")
        if not ruta_meta.is_file():
            raise ValueError(
                f"METADATOS_FALTANTES: No existe el archivo de metadatos {ruta_meta} "
                f"para verificar la integridad del banco {ruta_archivo}."
            )

        contenido = ruta_archivo.read_bytes()
        sha_calculado = hashlib.sha256(contenido).hexdigest()

        try:
            meta_json = json.loads(ruta_meta.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"METADATOS_CORRUPTOS: Error al leer metadatos de {ruta_meta}: {e}")

        sha_esperado = meta_json.get("sha256_npz")
        if not sha_esperado:
            raise ValueError(
                f"HASH_NO_ESPECIFICADO: Los metadatos de {ruta_archivo} en {ruta_meta} "
                f"no contienen 'sha256_npz' para verificar su integridad."
            )

        if sha_calculado != sha_esperado:
            raise ValueError(
                f"HASH_BANCO_INCONSISTENTE: el archivo {ruta_archivo} fue modificado o está corrupto. "
                f"Esperado: {sha_esperado}, calculado: {sha_calculado}."
            )

        metadatos = meta_json.get("metadatos", {})
        datos_npz = np.load(ruta_archivo)
        referencias = {}
        for k in datos_npz.files:
            if k.startswith("ref_"):
                clase = k[4:]
                referencias[clase] = datos_npz[k]
        return cls(referencias_por_clase=referencias, metadatos=metadatos)


def evaluar_decision_acceso_con_distancia(
    nombre_predicho: str,
    confianza: float,
    umbral_confianza: float,
    personas_autorizadas: List[str],
    vector_representacion: Optional[np.ndarray] = None,
    banco_referencias: Optional[BancoReferencias] = None,
    umbral_distancia: Optional[float] = None,
    metrica_distancia: str = "min_cos",
    estado_deteccion: EstadoDeteccion = EstadoDeteccion.UN_ROSTRO,
    motivo_fallo: Optional[str] = None
) -> DecisionAcceso:
    """
    Evalúa la decisión de acceso físico simulado incorporando el rechazo por distancia.

    Condiciones para ABRIR la puerta:
    1. Sin fallo previo de procesamiento o captura (motivo_fallo es None).
    2. Exactamente un rostro detectado (UN_ROSTRO).
    3. Sujeto predicho en personas_autorizadas (no desconocido).
    4. Confianza y umbral_confianza válidos (finitos, en [0, 1]), con confianza >= umbral_confianza.
    5. Configuración de distancia completa: banco y umbral deben coexistir.
       (Si ambos son None, aplica Regla A sin degradación silenciosa; si solo uno está presente, rechaza por configuración incompleta).
    6. Umbral de distancia finito y >= 0.0.
    7. Distancia calculada válida (finita, >= 0) y distancia <= umbral_distancia.

    Si falla cualquier condición, la puerta se mantiene CERRADA y se explicita el motivo.
    El rechazo conserva la persona predicha original para auditoría sin alterarla silenciosamente.
    """
    nom_limpio = (nombre_predicho or "").strip()
    nom_lower = nom_limpio.lower()

    # 0. Comprobación de fallo previo en adquisición o procesamiento
    if motivo_fallo:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=float(confianza) if _es_flotante_valido(confianza) else 0.0,
            umbral=float(umbral_confianza) if _es_flotante_valido(umbral_confianza) else 0.0,
            motivo_cierre=motivo_fallo,
            mensaje=f"ACCESO DENEGADO: {motivo_fallo}",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    # 1. Comprobación de detección física
    if estado_deteccion != EstadoDeteccion.UN_ROSTRO:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=float(confianza) if _es_flotante_valido(confianza) else 0.0,
            umbral=float(umbral_confianza) if _es_flotante_valido(umbral_confianza) else 0.0,
            motivo_cierre=f"DETECCION_{estado_deteccion.value}",
            mensaje=f"ACCESO DENEGADO: Estado detector '{estado_deteccion.value}'",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    # 2. Comprobación de etiquetas de desconocido explícito
    if nom_lower in ETIQUETAS_DESCONOCIDO_RESERVADAS:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=float(confianza) if _es_flotante_valido(confianza) else 0.0,
            umbral=float(umbral_confianza) if _es_flotante_valido(umbral_confianza) else 0.0,
            motivo_cierre=f"CLASE_DESCONOCIDO ({nom_limpio})",
            mensaje=f"ACCESO DENEGADO: Sujeto desconocido",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    # 3. Comprobación de identidad dentro de la lista de autorizados
    if nom_limpio not in personas_autorizadas:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada="DESCONOCIDO",
            confianza=float(confianza) if _es_flotante_valido(confianza) else 0.0,
            umbral=float(umbral_confianza) if _es_flotante_valido(umbral_confianza) else 0.0,
            motivo_cierre=f"CLASE_NO_AUTORIZADA ({nom_limpio})",
            mensaje=f"ACCESO DENEGADO: Sujeto no enrolado ({nom_limpio})",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    # 4. Validación de finitud y rango para confianza y umbral_confianza
    if not _es_flotante_valido(confianza) or not (0.0 <= float(confianza) <= 1.0):
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=float(confianza) if _es_flotante_valido(confianza) else float("nan"),
            umbral=float(umbral_confianza) if _es_flotante_valido(umbral_confianza) else float("nan"),
            motivo_cierre="PUNTAJE_INVALIDO",
            mensaje=f"ACCESO DENEGADO: Confianza no válida ({confianza})",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    if not _es_flotante_valido(umbral_confianza) or not (0.0 <= float(umbral_confianza) <= 1.0):
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=float(confianza),
            umbral=float(umbral_confianza) if _es_flotante_valido(umbral_confianza) else float("nan"),
            motivo_cierre="UMBRAL_CONFIANZA_INVALIDO",
            mensaje=f"ACCESO DENEGADO: Umbral de confianza no válido ({umbral_confianza})",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    f_confianza = float(confianza)
    f_umbral_conf = float(umbral_confianza)

    # 5. Comprobación del umbral de confianza del clasificador
    if f_confianza < f_umbral_conf:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=f_confianza,
            umbral=f_umbral_conf,
            motivo_cierre=f"RECHAZO_UMBRAL ({f_confianza:.4f} < {f_umbral_conf:.4f})",
            mensaje=f"ACCESO DENEGADO: Confianza insuficiente ({f_confianza * 100:.1f}% < {f_umbral_conf * 100:.0f}%)",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    # 6. Validación de integridad de la configuración de distancia
    tiene_banco = banco_referencias is not None
    tiene_umbral_d = umbral_distancia is not None

    if tiene_banco != tiene_umbral_d:
        # Configuración parcial (uno presente, el otro ausente): no se degrada silenciosamente a Regla A
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=f_confianza,
            umbral=f_umbral_conf,
            motivo_cierre="CONFIGURACION_DISTANCIA_INCOMPLETA",
            mensaje="ACCESO DENEGADO: Configuración de distancia incompleta (se exige banco y umbral conjuntamente)",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    # Caso Regla A intencional: ambos son None
    if not tiene_banco and not tiene_umbral_d:
        return DecisionAcceso(
            estado_puerta="ABIERTA",
            es_autorizado=True,
            persona_identificada=nom_limpio,
            confianza=f_confianza,
            umbral=f_umbral_conf,
            motivo_cierre="",
            mensaje=f"ACCESO CONCEDIDO: {nom_limpio} ({f_confianza * 100:.1f}%)",
            distancia=None,
            umbral_distancia=None
        )

    # Caso Regla B: ambos están presentes
    if not _es_flotante_valido(umbral_distancia) or float(umbral_distancia) < 0.0:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=f_confianza,
            umbral=f_umbral_conf,
            motivo_cierre="UMBRAL_DISTANCIA_INVALIDO",
            mensaje=f"ACCESO DENEGADO: Umbral de distancia no válido ({umbral_distancia})",
            distancia=None,
            umbral_distancia=float(umbral_distancia) if _es_flotante_valido(umbral_distancia) else None
        )

    f_umbral_dist = float(umbral_distancia)

    # 7. Cálculo y evaluación de distancia a las referencias de la clase predicha
    try:
        dist = banco_referencias.calcular_distancia(
            vector=vector_representacion,
            clase=nom_limpio,
            metrica=metrica_distancia
        )
    except (ValueError, KeyError) as e:
        motivo_err = str(e)
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=f_confianza,
            umbral=f_umbral_conf,
            motivo_cierre=f"RECHAZO_REPRESENTACION ({motivo_err})",
            mensaje=f"ACCESO DENEGADO: Representación o referencias inválidas ({motivo_err})",
            distancia=None,
            umbral_distancia=f_umbral_dist
        )

    if not math.isfinite(dist) or dist < 0.0:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=f_confianza,
            umbral=f_umbral_conf,
            motivo_cierre="DISTANCIA_NO_FINITA",
            mensaje="ACCESO DENEGADO: Distancia calculada no finita",
            distancia=None,
            umbral_distancia=f_umbral_dist
        )

    if dist > f_umbral_dist:
        return DecisionAcceso(
            estado_puerta="CERRADA",
            es_autorizado=False,
            persona_identificada=nom_limpio,
            confianza=f_confianza,
            umbral=f_umbral_conf,
            motivo_cierre=f"RECHAZO_DISTANCIA ({dist:.4f} > {f_umbral_dist:.4f})",
            mensaje=f"ACCESO DENEGADO: Representación lejana de {nom_limpio} (d={dist:.4f} > {f_umbral_dist:.4f})",
            distancia=dist,
            umbral_distancia=f_umbral_dist
        )

    # Cumplió puntaje y distancia
    return DecisionAcceso(
        estado_puerta="ABIERTA",
        es_autorizado=True,
        persona_identificada=nom_limpio,
        confianza=f_confianza,
        umbral=f_umbral_conf,
        motivo_cierre="",
        mensaje=f"ACCESO CONCEDIDO: {nom_limpio} ({f_confianza * 100:.1f}%, d={dist:.4f})",
        distancia=dist,
        umbral_distancia=f_umbral_dist
    )
