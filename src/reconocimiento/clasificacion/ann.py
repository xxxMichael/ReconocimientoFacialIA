"""
Implementación de Red Neuronal Artificial Densa (ANN / MLP) para reconocimiento facial.
Basada en los ejercicios prácticos de clase (AI/NNA):
- Representación de entrada: 256 características de histogramas regionales (idénticas a KNN, Árbol y SVM).
- Arquitectura compacta adaptada a N_train = 240 (6 imágenes por sujeto):
  Entrada (256) -> Lineal(128) -> ReLU -> BatchNorm1d -> Dropout(0.25)
                -> Lineal(64) -> ReLU -> Dropout(0.25)
                -> Lineal(40) [Logits para 40 clases]
- Función de pérdida: CrossEntropyLoss (Softmax + NLLLoss).
- Optimizador: Adam con regularización L2 (weight_decay=1e-4).
- Selección de época basada en la máxima exactitud sobre el conjunto de validación (Early Checkpoint).
- Guardado de curvas de aprendizaje (pérdida y exactitud) en curvas_aprendizaje.png.
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from reconocimiento.clasificacion.base import (
    ClasificadorFacialBase,
    ReporteEvaluacion,
    ResumenExperimento
)


class RedDensaPyTorch:
    """Módulo PyTorch para la red neuronal densa."""
    def __init__(self, n_in: int = 256, n_h1: int = 128, n_h2: int = 64, n_out: int = 40, p_drop: float = 0.25):
        import torch
        import torch.nn as nn

        class _ModuloANN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(n_in, n_h1)
                self.bn1 = nn.BatchNorm1d(n_h1)
                self.relu = nn.ReLU()
                self.drop1 = nn.Dropout(p_drop)
                self.fc2 = nn.Linear(n_h1, n_h2)
                self.drop2 = nn.Dropout(p_drop)
                self.fc3 = nn.Linear(n_h2, n_out)

            def forward(self, x):
                x = self.drop1(self.relu(self.bn1(self.fc1(x))))
                x = self.drop2(self.relu(self.fc2(x)))
                return self.fc3(x)

        self.net = _ModuloANN()

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


class ClasificadorANN(ClasificadorFacialBase):
    """
    Clasificador ANN denso con PyTorch, checkpointing sobre validación y
    registro de curvas de pérdida y exactitud.
    """

    def __init__(
        self,
        n_entrada: int = 256,
        n_oculta1: int = 128,
        n_oculta2: int = 64,
        n_clases: int = 40,
        dropout: float = 0.25,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        tamano_lote: int = 16,
        epocas: int = 120,
        semilla: int = 42
    ):
        self.n_entrada = n_entrada
        self.n_oculta1 = n_oculta1
        self.n_oculta2 = n_oculta2
        self.n_clases = n_clases
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.tamano_lote = tamano_lote
        self.epocas = epocas
        self.semilla = semilla

        self._fijar_semilla(self.semilla)
        self.red_wrapper = RedDensaPyTorch(
            n_in=self.n_entrada,
            n_h1=self.n_oculta1,
            n_h2=self.n_oculta2,
            n_out=self.n_clases,
            p_drop=self.dropout
        )
        self.modelo = self.red_wrapper.net
        self.clases_ = np.arange(self.n_clases)
        self.mejor_epoca_: int = 0
        self.mejor_val_acc_: float = 0.0
        self.historia_entrenamiento: Dict[str, List[float]] = {
            "loss_train": [],
            "loss_val": [],
            "acc_train": [],
            "acc_val": []
        }

    @staticmethod
    def _fijar_semilla(seed: int) -> None:
        import torch
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entrenamiento básico sin validación intermedia.
        Recomendado: usar entrenar_con_validacion para checkpointing óptimo.
        """
        self.entrenar_con_validacion(X_train, y_train, X_train, y_train)

    def entrenar_con_validacion(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        pesos_clases: Optional[Any] = None,
        metrica_checkpoint: str = "acc",
        pesos_muestras: Optional[np.ndarray] = None,
    ) -> None:
        """
        Entrena la red durante el número de épocas especificado, evaluando
        al final de cada época en validación y guardando los mejores pesos
        según la métrica de checkpoint ('acc' o 'macro_recall').
        """
        import copy
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        self._fijar_semilla(self.semilla)

        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        if pesos_muestras is not None:
            w_muestras_t = torch.tensor(np.asarray(pesos_muestras, dtype=np.float32), dtype=torch.float32)
            dataset_tr = TensorDataset(X_tr_t, y_tr_t, w_muestras_t)
            criterio_tr = nn.CrossEntropyLoss(reduction="none")
            criterio_val = nn.CrossEntropyLoss()
        else:
            dataset_tr = TensorDataset(X_tr_t, y_tr_t)
            if pesos_clases is not None:
                w_tensor = torch.as_tensor(pesos_clases, dtype=torch.float32)
                criterio_tr = nn.CrossEntropyLoss(weight=w_tensor)
                criterio_val = criterio_tr
            else:
                criterio_tr = nn.CrossEntropyLoss()
                criterio_val = criterio_tr

        loader_tr = DataLoader(dataset_tr, batch_size=self.tamano_lote, shuffle=True)

        optimizador = optim.Adam(
            self.modelo.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        mejor_score = -1.0
        mejor_pesos = None
        self.historia_entrenamiento = {
            "loss_train": [],
            "loss_val": [],
            "acc_train": [],
            "acc_val": []
        }

        for ep in range(1, self.epocas + 1):
            self.modelo.train()
            loss_acum = 0.0
            correctos_tr = 0
            total_tr = 0

            for lote in loader_tr:
                if pesos_muestras is not None:
                    bx, by, bw = lote
                    optimizador.zero_grad()
                    salida = self.modelo(bx)
                    loss = (criterio_tr(salida, by) * bw).mean()
                else:
                    bx, by = lote
                    optimizador.zero_grad()
                    salida = self.modelo(bx)
                    loss = criterio_tr(salida, by)
                loss.backward()
                optimizador.step()

                loss_acum += loss.item() * len(by)
                preds = torch.argmax(salida, dim=1)
                correctos_tr += (preds == by).sum().item()
                total_tr += len(by)

            loss_tr_ep = loss_acum / total_tr
            acc_tr_ep = (correctos_tr / total_tr) * 100.0

            # Evaluación en validación
            self.modelo.eval()
            with torch.no_grad():
                out_val = self.modelo(X_val_t)
                loss_val_ep = criterio_val(out_val, y_val_t).item()
                preds_val = torch.argmax(out_val, dim=1)
                acc_val_ep = ((preds_val == y_val_t).sum().item() / len(y_val_t)) * 100.0

            self.historia_entrenamiento["loss_train"].append(round(loss_tr_ep, 4))
            self.historia_entrenamiento["loss_val"].append(round(loss_val_ep, 4))
            self.historia_entrenamiento["acc_train"].append(round(acc_tr_ep, 2))
            self.historia_entrenamiento["acc_val"].append(round(acc_val_ep, 2))

            if metrica_checkpoint == "macro_recall":
                recalls = []
                for c in range(self.n_clases):
                    mask_c = (y_val_t == c)
                    total_c = mask_c.sum().item()
                    if total_c > 0:
                        rec_c = ((preds_val[mask_c] == c).sum().item() / total_c) * 100.0
                        recalls.append(rec_c)
                score_val_ep = float(np.mean(recalls)) if recalls else acc_val_ep
            else:
                score_val_ep = acc_val_ep

            if score_val_ep > mejor_score:
                mejor_score = score_val_ep
                self.mejor_epoca_ = ep
                mejor_pesos = copy.deepcopy(self.modelo.state_dict())

        # Restaurar mejor modelo
        if mejor_pesos is not None:
            self.modelo.load_state_dict(mejor_pesos)
        self.mejor_val_acc_ = mejor_score
        self.modelo.eval()

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predice las identidades para la matriz de características."""
        import torch
        self.modelo.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32))
            logits = self.modelo(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades mediante Softmax sobre los logits."""
        import torch
        self.modelo.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32))
            logits = self.modelo(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def guardar(self, ruta_archivo: Path) -> None:
        """Guarda los pesos y metadatos del modelo en formato PyTorch."""
        import torch
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.modelo.state_dict(),
            "config": {
                "n_entrada": self.n_entrada,
                "n_oculta1": self.n_oculta1,
                "n_oculta2": self.n_oculta2,
                "n_clases": self.n_clases,
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "tamano_lote": self.tamano_lote,
                "epocas": self.epocas,
                "semilla": self.semilla
            },
            "mejor_epoca": self.mejor_epoca_,
            "mejor_val_acc": self.mejor_val_acc_,
            "historia": self.historia_entrenamiento
        }, ruta_archivo)

    @classmethod
    def cargar(cls, ruta_archivo: Path) -> "ClasificadorANN":
        """Carga un modelo guardado previamente."""
        import torch
        checkpoint = torch.load(ruta_archivo, weights_only=False)
        cfg = checkpoint["config"]
        instancia = cls(**cfg)
        instancia.modelo.load_state_dict(checkpoint["state_dict"])
        instancia.mejor_epoca_ = checkpoint.get("mejor_epoca", 0)
        instancia.mejor_val_acc_ = checkpoint.get("mejor_val_acc", 0.0)
        instancia.historia_entrenamiento = checkpoint.get("historia", {})
        instancia.modelo.eval()
        return instancia

    def ejecutar_experimento_completo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        ids_train: List[str],
        X_val: np.ndarray,
        y_val: np.ndarray,
        ids_val: List[str],
        X_test: np.ndarray,
        y_test: np.ndarray,
        ids_test: List[str],
        nombre_dataset: str = "olivetti"
    ) -> ResumenExperimento:
        """
        Entrena con monitoreo en validación, evalúa en las tres particiones y calcula métricas.
        """
        self.entrenar_con_validacion(X_train, y_train, X_val, y_val)

        rep_train = self.evaluar_particion(X_train, y_train, ids_train, "entrenamiento")
        rep_val = self.evaluar_particion(X_val, y_val, ids_val, "validacion")
        rep_test = self.evaluar_particion(X_test, y_test, ids_test, "prueba")

        brecha = round(rep_train.exactitud_porcentaje - rep_test.exactitud_porcentaje, 2)

        config = {
            "arquitectura": f"256 -> Linear({self.n_oculta1}) -> BN -> ReLU -> Drop({self.dropout}) -> Linear({self.n_oculta2}) -> ReLU -> Drop({self.dropout}) -> Linear({self.n_clases})",
            "parametros_entrenables": self.red_wrapper.contar_parametros(),
            "optimizador": "Adam",
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "tamano_lote": self.tamano_lote,
            "total_epocas": self.epocas,
            "mejor_epoca_seleccionada": self.mejor_epoca_,
            "mejor_exactitud_val": self.mejor_val_acc_,
            "semilla": self.semilla
        }

        return ResumenExperimento(
            algoritmo="ann",
            dataset=nombre_dataset,
            configuracion=config,
            reporte_train=rep_train,
            reporte_val=rep_val,
            reporte_test=rep_test,
            brecha_generalizacion_puntos=brecha
        )
