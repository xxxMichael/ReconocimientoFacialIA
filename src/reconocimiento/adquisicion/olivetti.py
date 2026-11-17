"""
Módulo de adquisición para el dataset público Olivetti Faces (AT&T/ORL).
Proporciona carga, generación de identificadores estables y partición 3/1/1 estratificada por sujeto.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.datasets import fetch_olivetti_faces


@dataclass
class DatasetParticionado:
    """Contenedor de datos particionados con identificadores estables."""
    X_train: np.ndarray
    y_train: np.ndarray
    ids_train: List[str]

    X_val: np.ndarray
    y_val: np.ndarray
    ids_val: List[str]

    X_test: np.ndarray
    y_test: np.ndarray
    ids_test: List[str]

    n_clases: int
    dimensiones_originales: Tuple[int, int]
    pre_recortada: bool = True


def cargar_olivetti(
    data_dir: Optional[Path] = None,
    semilla: int = 42,
    descargar: bool = True
) -> DatasetParticionado:
    """
    Carga el dataset Olivetti Faces y realiza partición reproducible 3/1/1
    (6 imágenes entrenamiento, 2 validación y 2 prueba por cada uno de los 40 sujetos).

    Total: 400 imágenes de 64x64 píxeles en escala de grises [0, 1].
    - Entrenamiento: 240 imágenes (60%)
    - Validación:     80 imágenes (20%)
    - Prueba:         80 imágenes (20%)
    """
    data_home = str(data_dir) if data_dir is not None else None
    dataset = fetch_olivetti_faces(
        data_home=data_home,
        shuffle=False,
        download_if_missing=descargar,
        random_state=semilla
    )

    imagenes = dataset.images  # (400, 64, 64) float32 en [0, 1]
    etiquetas = dataset.target  # (400,) int con valores 0..39
    total_muestras = len(etiquetas)
    n_clases = len(np.unique(etiquetas))

    # Generación de identificadores estables y partición estratificada 6 / 2 / 2 por sujeto
    rng = np.random.RandomState(semilla)

    indices_train = []
    indices_val = []
    indices_test = []

    ids_train = []
    ids_val = []
    ids_test = []

    for sujeto in range(n_clases):
        idxs_sujeto = np.where(etiquetas == sujeto)[0]
        # Permutar determinísticamente los 10 índices del sujeto
        perm = rng.permutation(idxs_sujeto)
        # Asignación fija 3/1/1 (6 / 2 / 2)
        train_sel = perm[:6]
        val_sel = perm[6:8]
        test_sel = perm[8:10]

        indices_train.extend(train_sel)
        indices_val.extend(val_sel)
        indices_test.extend(test_sel)

        for i, idx in enumerate(train_sel):
            ids_train.append(f"olivetti_s{sujeto:02d}_img{idx:03d}_train{i}")
        for i, idx in enumerate(val_sel):
            ids_val.append(f"olivetti_s{sujeto:02d}_img{idx:03d}_val{i}")
        for i, idx in enumerate(test_sel):
            ids_test.append(f"olivetti_s{sujeto:02d}_img{idx:03d}_test{i}")

    indices_train = np.array(indices_train)
    indices_val = np.array(indices_val)
    indices_test = np.array(indices_test)

    return DatasetParticionado(
        X_train=imagenes[indices_train],
        y_train=etiquetas[indices_train],
        ids_train=ids_train,
        X_val=imagenes[indices_val],
        y_val=etiquetas[indices_val],
        ids_val=ids_val,
        X_test=imagenes[indices_test],
        y_test=etiquetas[indices_test],
        ids_test=ids_test,
        n_clases=n_clases,
        dimensiones_originales=(64, 64),
        pre_recortada=True
    )
