"""
Implementación de Red Neuronal Convolucional (CNN) para reconocimiento facial.
Basada en el principio de aprendizaje jerárquico de representaciones espaciales:
- Entrada espacial directa: Rostros en escala de grises de 64 x 64 píxeles (rango [0.0, 1.0]).
- Arquitectura compacta adaptada a N_train = 240 (6 imágenes por identidad):
  * Bloque 1: Conv2d(1, 16, 3, pad=1) -> BatchNorm2d(16) -> ReLU -> MaxPool2d(2, 2) [Salida: 16 x 32 x 32]
  * Bloque 2: Conv2d(16, 32, 3, pad=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2, 2) [Salida: 32 x 16 x 16]
  * Bloque 3: Conv2d(32, 64, 3, pad=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2, 2) [Salida: 64 x 8 x 8 = 4096]
  * Clasificador Denso: Aplanar -> Linear(4096, 128) -> ReLU -> Dropout(0.35) -> Linear(128, 40)
- Jerarquía de características aprendidas:
  1. Filtros convolucionales iniciales: detección de bordes, líneas y texturas locales de piel.
  2. Filtros intermedios: partes faciales (ojos, cejas, tabique nasal, boca).
  3. Filtros profundos: combinaciones globales y relaciones métricas entre rasgos.
  4. Capas densas: discriminación de identidad sobre el espacio latente.
- Contraste con características del grupo:
  Mientras que KNN, Árbol, SVM y ANN emplean los histogramas regionales fijos (4x4, 16 bins),
  la CNN aprende sus propios filtros espaciales directamente de la matriz de píxeles normalizada.
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


class RedConvolucionalPyTorch:
    """Módulo PyTorch para la arquitectura CNN compacta."""
    def __init__(self, n_clases: int = 40, p_drop: float = 0.35):
        import torch
        import torch.nn as nn

        class _ModuloCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # Bloque 1: Entrada 1x64x64 -> 16x32x32
                self.b1_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.b1_bn = nn.BatchNorm2d(16)
                self.b1_relu = nn.ReLU()
                self.b1_pool = nn.MaxPool2d(2, 2)

                # Bloque 2: 16x32x32 -> 32x16x16
                self.b2_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.b2_bn = nn.BatchNorm2d(32)
                self.b2_relu = nn.ReLU()
                self.b2_pool = nn.MaxPool2d(2, 2)

                # Bloque 3: 32x16x16 -> 64x8x8
                self.b3_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.b3_bn = nn.BatchNorm2d(64)
                self.b3_relu = nn.ReLU()
                self.b3_pool = nn.MaxPool2d(2, 2)

                # Clasificador: 64*8*8 = 4096 -> 128 -> 40
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(64 * 8 * 8, 128)
                self.fc1_relu = nn.ReLU()
                self.dropout = nn.Dropout(p_drop)
                self.fc2 = nn.Linear(128, n_clases)

            def forward(self, x):
                # x shape: [B, 1, 64, 64]
                x = self.b1_pool(self.b1_relu(self.b1_bn(self.b1_conv(x))))
                x = self.b2_pool(self.b2_relu(self.b2_bn(self.b2_conv(x))))
                x = self.b3_pool(self.b3_relu(self.b3_bn(self.b3_conv(x))))
                x = self.flatten(x)
                x = self.dropout(self.fc1_relu(self.fc1(x)))
                return self.fc2(x)

        self.net = _ModuloCNN()

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


class ClasificadorCNN(ClasificadorFacialBase):
    """
    Clasificador CNN que opera sobre tensores de imagen 2D (B, 1, 64, 64).
    """

    def __init__(
        self,
        n_clases: int = 40,
        dropout: float = 0.35,
        lr: float = 0.0005,
        weight_decay: float = 1e-4,
        tamano_lote: int = 16,
        epocas: int = 80,
        semilla: int = 42
    ):
        self.n_clases = n_clases
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.tamano_lote = tamano_lote
        self.epocas = epocas
        self.semilla = semilla

        self._fijar_semilla(self.semilla)
        self.red_wrapper = RedConvolucionalPyTorch(
            n_clases=self.n_clases,
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

    @staticmethod
    def _asegurar_tensor_4d(X: np.ndarray) -> np.ndarray:
        """
        Garantiza que X tenga dimensión (N, 1, H, W).
        Si entra (N, H, W) -> expande a (N, 1, H, W).
        Si entra (N, H*W) -> redimensiona a (N, 1, 64, 64).
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 3:
            return np.expand_dims(X_arr, axis=1)
        elif X_arr.ndim == 2:
            n_muestras = X_arr.shape[0]
            lado = int(np.round(np.sqrt(X_arr.shape[1])))
            return X_arr.reshape((n_muestras, 1, lado, lado))
        elif X_arr.ndim == 4:
            return X_arr
        raise ValueError(f"Dimensión de entrada inválida para CNN: {X_arr.shape}")

    def entrenar(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Entrenamiento estándar sin validación."""
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
        Entrena la CNN con optimizador Adam y guarda el checkpoint con mejor métrica en validación
        ('acc' o 'macro_recall').
        """
        import copy
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        self._fijar_semilla(self.semilla)

        X_tr_4d = self._asegurar_tensor_4d(X_train)
        X_val_4d = self._asegurar_tensor_4d(X_val)

        X_tr_t = torch.tensor(X_tr_4d, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_4d, dtype=torch.float32)
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

            # Validación
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

        if mejor_pesos is not None:
            self.modelo.load_state_dict(mejor_pesos)
        self.mejor_val_acc_ = mejor_score
        self.modelo.eval()

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predice identidades para las imágenes dadas."""
        import torch
        self.modelo.eval()
        X_4d = self._asegurar_tensor_4d(X)
        with torch.no_grad():
            X_t = torch.tensor(X_4d, dtype=torch.float32)
            logits = self.modelo(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades por Softmax sobre las salidas de la CNN."""
        import torch
        self.modelo.eval()
        X_4d = self._asegurar_tensor_4d(X)
        with torch.no_grad():
            X_t = torch.tensor(X_4d, dtype=torch.float32)
            logits = self.modelo(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def extraer_caracteristicas(self, X: np.ndarray) -> np.ndarray:
        """
        Extrae la representación latente de 128 dimensiones anterior a la capa clasificadora final (fc2).
        Garantiza que el modelo opere en modo eval() y sin dropout activo.
        """
        import torch
        self.modelo.eval()
        X_4d = self._asegurar_tensor_4d(X)
        with torch.no_grad():
            X_t = torch.tensor(X_4d, dtype=torch.float32)
            m = self.modelo
            x = m.b1_pool(m.b1_relu(m.b1_bn(m.b1_conv(X_t))))
            x = m.b2_pool(m.b2_relu(m.b2_bn(m.b2_conv(x))))
            x = m.b3_pool(m.b3_relu(m.b3_bn(m.b3_conv(x))))
            flat = m.flatten(x)
            feat = m.fc1_relu(m.fc1(flat))
        return feat.cpu().numpy().astype(np.float32)

    def guardar(self, ruta_archivo: Path) -> None:
        """Guarda los pesos entrenados y metadatos en formato PyTorch."""
        import torch
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.modelo.state_dict(),
            "config": {
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
    def cargar(cls, ruta_archivo: Path) -> "ClasificadorCNN":
        """Carga un modelo CNN serializado."""
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
        Ejecuta el ciclo de entrenamiento CNN con checkpoint en validación
        y evaluación final en prueba.
        """
        self.entrenar_con_validacion(X_train, y_train, X_val, y_val)

        rep_train = self.evaluar_particion(X_train, y_train, ids_train, "entrenamiento")
        rep_val = self.evaluar_particion(X_val, y_val, ids_val, "validacion")
        rep_test = self.evaluar_particion(X_test, y_test, ids_test, "prueba")

        brecha = round(rep_train.exactitud_porcentaje - rep_test.exactitud_porcentaje, 2)

        config = {
            "arquitectura": "Conv(1->16, 3x3) -> BN -> ReLU -> Pool(2x2) [32x32] -> Conv(16->32, 3x3) -> BN -> ReLU -> Pool(2x2) [16x16] -> Conv(32->64, 3x3) -> BN -> ReLU -> Pool(2x2) [8x8] -> Flatten(4096) -> Linear(4096, 128) -> ReLU -> Drop(0.35) -> Linear(128, 40)",
            "dimensiones_bloques": {
                "bloque1_salida": [16, 32, 32],
                "bloque2_salida": [32, 16, 16],
                "bloque3_salida": [64, 8, 8],
                "clasificador_entrada": 4096,
                "clasificador_oculta": 128,
                "clasificador_salida": 40
            },
            "parametros_entrenables": self.red_wrapper.contar_parametros(),
            "optimizador": "Adam",
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "tamano_lote": self.tamano_lote,
            "total_epocas": self.epocas,
            "mejor_epoca_seleccionada": self.mejor_epoca_,
            "mejor_exactitud_val": self.mejor_val_acc_,
            "representacion": "Espacial 2D directa (64x64 gris) - Representation Learning",
            "semilla": self.semilla
        }

        return ResumenExperimento(
            algoritmo="cnn",
            dataset=nombre_dataset,
            configuracion=config,
            reporte_train=rep_train,
            reporte_val=rep_val,
            reporte_test=rep_test,
            brecha_generalizacion_puntos=brecha
        )
