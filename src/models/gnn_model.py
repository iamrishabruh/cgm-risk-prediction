import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config.config import Config


class PatientGNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim=Config.GNN_HIDDEN_DIM,
        dropout=Config.GNN_DROPOUT,
        num_layers=Config.GNN_NUM_LAYERS,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.classifier = torch.nn.Linear(hidden_dim, Config.NUM_CLASSES)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(F.relu(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
