import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    Mixture of Experts Gating Network.
    Input: Tabular Features (Morphology + History + Omics)
    Output: Softmax Weights [w_xgb, w_lgb, w_cnn]
    """
    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts), # Dynamic Experts
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.net(x)
