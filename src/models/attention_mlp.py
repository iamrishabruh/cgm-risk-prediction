import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config


class AttentionMLP(nn.Module):
    def __init__(
        self,
        input_dim=Config.NUM_FEATURES,
        hidden_dim=256,
        num_classes=Config.NUM_CLASSES,
        dropout=0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = x * attn_weights
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc_out(x)
