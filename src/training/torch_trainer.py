import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.config import Config
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

logger = logging.getLogger(__name__)


class TorchTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)

    def train_epoch(self, loader, is_graph: bool = False) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            if is_graph:
                data = batch.to(Config.DEVICE)
                outputs = self.model(data)
                labels = data.y.to(Config.DEVICE)
            else:
                inputs, labels = batch
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            if torch.isnan(loss):
                logger.error("Loss is NaN.")
                return float("nan")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader) if len(loader) > 0 else 0.0

    def eval_epoch(self, loader, is_graph: bool = False) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                if is_graph:
                    data = batch.to(Config.DEVICE)
                    outputs = self.model(data)
                    labels = data.y.to(Config.DEVICE)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(Config.DEVICE)
                    labels = labels.to(Config.DEVICE)
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(loader) if len(loader) > 0 else 0.0


def make_tabular_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=shuffle)


def make_gnn_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> GeoDataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    n = X_t.shape[0]
    edge_index = torch.tensor(
        [[i, j] for i in range(n) for j in range(n) if i != j],
        dtype=torch.long,
    ).t().contiguous()
    graph = Data(x=X_t, edge_index=edge_index, y=y_t)
    return GeoDataLoader([graph], batch_size=1, shuffle=shuffle)
