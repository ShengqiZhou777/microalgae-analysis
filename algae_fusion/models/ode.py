
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import torchdiffeq for adjoint sensitivity method (memory efficient gradients)
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    """
    Learns the derivative function dy/dt = f(y, t).
    Replaces the fixed Logistic/Gompertz equation with a Neural Network.
    
    Paper: 'Neural Ordinary Differential Equations' (Chen et al., 2018)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(ODEFunc, self).__init__()
        # [Non-Autonomous Upgrade] Input dim + 1 for time 't'
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim) # Output dim is still 'input_dim' (dy/dt)
        )
        # Initialize close to 0 to start with stable dynamics
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, t, y):
        # [Non-Autonomous Upgrade] Concatenate scalar t to vector y
        # t is a scalar tensor (shape [] or [1]). y is [Batch, Dim].
        # We need to broadcast t to [Batch, 1]
        batch_size = y.shape[0]
        t_vec = torch.ones(batch_size, 1).to(y.device) * t
        
        # Concat: [B, D] + [B, 1] -> [B, D+1]
        y_and_t = torch.cat([y, t_vec], dim=1)
        
        return self.net(y_and_t)

class NeuralODEPure(nn.Module):
    """
    Pure Neural ODE solver.
    Solves y(t) given y(t0) and dy/dt = ODEFunc(y).
    """
    def __init__(self, input_dim, hidden_dim=64, method='dopri5'):
        super().__init__()
        self.ode_func = ODEFunc(input_dim, hidden_dim)
        self.method = method

    def forward(self, y0, t):
        """
        Args:
            y0: Initial state [B, D]
            t: Time points to solve for [T] (must be strictly increasing)
        Returns:
            y_pred: [T, B, D] -> Permuted to [B, T, D] likely needed by caller
        """
        # odeint returns shape [T, B, D]
        out = odeint(self.ode_func, y0, t, method=self.method)
        return out.permute(1, 0, 2) # Return [B, T, D]

class ODERNN(nn.Module):
    """
    ODE-RNN Model for Irregularly Sampled Time Series.
    
    Paper: 'Latent ODEs for Irregularly-Sampled Time Series' (Rubanova et al., 2019)
    
    Mechanism:
    - Between observations: Solve posterior ODE (update latent state continuously).
    - At observations: Update latent state using RNN/GRU cell with new data.
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.gru = nn.GRUCell(input_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x, t, mask=None):
        """
        Args:
            x: Observed data [B, T, input_dim]
            t: Time points [T] (or [B, T] if strictly supported, but odeint usually takes 1D T)
            mask: [B, T] indicating present observations (1) or missing (0). 
                  If None, assume all present.
        """
        batch_size, num_steps, _ = x.shape
        device = x.device
        
        # Initial latent state h0
        h = torch.zeros(batch_size, self.latent_dim).to(device)
        
        outputs = []
        
        # Iterate through time steps
        for i in range(num_steps - 1):
            # 1. Update state at current observation t[i]
            # (Standard RNN update)
            if mask is not None:
                # Only update if observed; otherwise keep previous state (or just let GRU handle 0 input?)
                # Paper usually suggests: h_new = GRU(h_ode, x_i) 
                # Here we do a simple GRU update.
                xi = x[:, i, :]
                hi_new = self.gru(xi, h)
                # Apply mask: if missing, keep h (which is h_ode from previous step)
                m = mask[:, i].unsqueeze(-1)
                h = m * hi_new + (1 - m) * h
            else:
                 h = self.gru(x[:, i, :], h)
            
            outputs.append(h)

            # 2. Solve ODE from t[i] to t[i+1]
            t_span = t[i:i+2]
            # odeint returns [2, B, D], we take the state at t[i+1] (index 1)
            h_next = odeint(self.ode_func, h, t_span)[1]
            h = h_next
            
        # Final step update
        if mask is not None:
             m = mask[:, -1].unsqueeze(-1)
             h = m * self.gru(x[:, -1, :], h) + (1 - m) * h
        else:
             h = self.gru(x[:, -1, :], h)
        outputs.append(h)
        
        return torch.stack(outputs, dim=1) # [B, T, D]

class NeuralODEParameterizer(nn.Module):
    """
    [Legacy/Compatibility Wrapper] 
    Originally predicted r, K for fixed ODE.
    Now acts as an Encoder to predict the INITIAL STATE (y0) or Context for Neural ODE.
    """
    def __init__(self, input_dim, latent_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim) 
        )
        
    def forward(self, x):
        # Maps static features -> Initial Latent State y0
        return self.fc(x)

# Example wrapper for 'GrowthODE' to maintain compatibility with your pipeline
class GrowthODE(nn.Module):
    """
    New 'Neural' GrowthODE.
    Combines Parameterizer (Encoder) + Neural ODE Solver.
    """
    def __init__(self, model_type="ode_rnn", input_dim=2, latent_dim=64, ode_hidden_dim=128, hidden_dim=None):
        super().__init__()
        if hidden_dim is not None:
            latent_dim = hidden_dim
        # model_type argument kept for compatibility
        # For pipeline compatibility, we use ODERNN
        # input_dim: features from dataset
        # hidden_dim: latent state size of ODE
        self.ode_net = ODERNN(input_dim, latent_dim, hidden_dim=ode_hidden_dim)
        
    def forward(self, x, t, mask=None):
        """
        Forward pass using ODERNN.
        Args:
            x: [B, T, D] Features
            t: [T] or [B, T] Time points
            mask: [B, T] Mask
        """
        return self.ode_net(x, t, mask)
