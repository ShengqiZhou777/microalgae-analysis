import torch
import torch.nn as nn

class MorphLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super(MorphLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        # x shape: (Batch, Sequence=3, Features)
        
        # LSTM output: (Batch, Seq, Hidden)
        lstm_out, _ = self.lstm(x)
        
        # We only care about the last time step (T)
        last_hidden = lstm_out[:, -1, :]
        
        # Prediction
        out = self.fc(last_hidden)
        return out.squeeze()
