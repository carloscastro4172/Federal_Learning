import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    """MLP adaptable al número real de features. Devuelve logits (sin sigmoid)."""
    def __init__(self, in_features: int, seed: int = 42, p_dropout: float = 0.3):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        return self.fc3(x)  # logits
