
import torch
import torch.nn as nn
from torchdiffeq import odeint

class AutonomousODEFunc(nn.Module):
    """
    The Physics Engine: dy/dt = f(y)
    Explicitly Autonomous (Time-Invariant).
    The input 't' is ignored, making the dynamics depend ONLY on the state 'y'.
    """
    def __init__(self, latent_dim=64):
        super(AutonomousODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.Tanh(), 
            nn.Dropout(0.2), # Regularization
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )

    def forward(self, t, y):
        # t is required by odeint signature but ignored by the model logic
        return self.net(y)

class ODERNNEncoder(nn.Module):
    """
    Encodes a sparse/irregular sequence (x, t) into a latent distribution q(z0).
    Mechanism: 
    1. Jump between observations using ODE (solve gap).
    2. Update state at observation using GRU.
    3. Final state -> (mu, logvar).
    """
    def __init__(self, input_dim, latent_dim, ode_func):
        super(ODERNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ode_func = ode_func # Shared physics engine (optional, but good for consistency)
        
        self.gru = nn.GRUCell(input_dim, latent_dim)
        
        # Variational Heads
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x_seq, t_seq):
        """
        x_seq: [Batch, Time, Feat] (Padding should be handled if batching variable lengths)
        t_seq: [Batch, Time] or [Time] if shared. Assume [Batch, Time] for generality.
        """
        batch_size = x_seq.size(0)
        seq_len = x_seq.size(1)
        
        # Initialize hidden state h0
        h = torch.zeros(batch_size, self.latent_dim).to(x_seq.device)
        
        # Iterate through the sequence
        for i in range(seq_len):
            # If not the first step, evolve h from previous time to current time
            if i > 0:
                t_prev = t_seq[:, i-1]
                t_curr = t_seq[:, i]
                # Note: This is simplified. Strict batching with variable t is hard.
                # Assuming t_seq is same for all batch items or we run loop carefully.
                # For this implementation, we assume shared t_seq across batch for simplicity
                # or we rely on odeint to handle scalar times.
                
                # Check if gap > 0 (handle padding or same-time inputs)
                # Here we use a simplified assumption: t is shared [Time] vector
                # If t is per-sample, we need sophisticated masking.
                
                # Using mean time jump for batch (heuristic) or strictly shared time
                # Let's assume shared t for now (standard for lab experiments)
                t_start = t_seq[0, i-1]
                t_end = t_seq[0, i]
                
                if t_end > t_start:
                   # Evolve h: [t_prev, t_curr]
                   # odeint returns [2, Batch, Latent], we take the 2nd one (end)
                   h = odeint(self.ode_func, h, torch.tensor([t_start, t_end]).to(x_seq.device), rtol=1e-3, atol=1e-3)[1]
            
            # GRU Update with observation
            # Check for mask (if x is 0 padded). Assuming mask input or simple check.
            # Here we just update.
            h = self.gru(x_seq[:, i], h)
            
        # Reparameterization
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class LatentODE(nn.Module):
    """
    The VAE Wrapper.
    """
    def __init__(self, input_dim, latent_dim=64):
        super(LatentODE, self).__init__()
        self.ode_func = AutonomousODEFunc(latent_dim)
        self.encoder = ODERNNEncoder(input_dim, latent_dim, self.ode_func)
        
        # Decoder: Latent(t) -> x(t)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq, t_seq, t_eval):
        """
        Training Forward Pass:
        1. Encode sequence -> q(z0)
        2. Sample z0
        3. Solve ODE for t_eval -> z(t)
        4. Decode -> x_pred(t)
        """
        # Encode
        mu, logvar = self.encoder(x_seq, t_seq)
        
        # Sample
        z0 = self.reparameterize(mu, logvar)
        
        # Integration (Generate Trajectory)
        # z_traj: [Time, Batch, Latent]
        z_traj = odeint(self.ode_func, z0, t_eval, rtol=1e-3, atol=1e-3)
        
        # Decode
        T, B, L = z_traj.shape
        x_pred = self.decoder(z_traj.view(T*B, L)).view(T, B, -1)
        
        # Return everything for Loss
        return x_pred, mu, logvar

    def infer_single_point(self, x_obs, t_obs, t_eval):
        """
        Single Point Inference (e.g. t=20 -> Full Trajectory)
        This requires 'Reverse Encoding' or just treating it as a length-1 sequence
        and relying on the ODE-RNN to bridge 0->20 gap effectively (reverse solve).
        
        Better strategy for single point t=20:
        1. Encode x_obs as if it happened at t=0? No.
        2. We need to find z0 such that z(20) approx z_enc(x20).
        
        For this implementation, we'll stick to the ODERNN capacity:
        If we feed (x_obs, t_obs) to Encoder, it evolves h0(0) -> h(t_obs) -> update -> h_final.
        We treat h_final as z(t_obs).
        Then we solve BACKWARDS to get z0.
        """
        # 1. Encode Observation to get posterior z(t_obs)
        # Note: ODERNN normally gives z0. But if we run it up to t_obs, the hidden state
        # effectively captures the state AT t_obs (plus history).
        # For single point, h = GRU(x_obs, 0). This is just embedding x_obs.
        # This h represents z(t_obs).
        
        # Quick Hack for Single Point:
        # Map x_obs -> z_obs using a simple Projector (or use the GRU part of encoder)
        # Here we reuse Encoder logic but carefully.
        
        batch_size = x_obs.size(0)
        h = torch.zeros(batch_size, self.encoder.latent_dim).to(x_obs.device)
        
        # Evolve 0 -> t_obs? No, we don't know z0.
        # We assume h0 = 0 (prior).
        
        # Standard ODERNN behavior:
        # h0 --(ODE 0->t)--> h_pre --(GRU x)--> h_post
        # This h_post is supposed to be the summary of the sequence.
        # In VAE Latent ODE framework, this summary is mapped to z0 parameter.
        
        # Let's rely on the trained Encoder.
        # If trained with Sparse Data (random t_start), it learns to map (x_t, t) -> z0.
        # So we just feed it formatted as sequence.
        
        # Format input as sequence length 1
        x_seq_in = x_obs.unsqueeze(1) # [B, 1, D]
        t_seq_in = t_obs.unsqueeze(1) # [B, 1]
        
        # Encode -> z0
        mu, logvar = self.encoder(x_seq_in, t_seq_in)
        z0 = mu # Use mean for deterministic inference
        
        # Decode Full Trajectory
        z_traj = odeint(self.ode_func, z0, t_eval, rtol=1e-3, atol=1e-3)
        T, B, L = z_traj.shape
        x_pred = self.decoder(z_traj.view(T*B, L)).view(T, B, -1)
        
        return x_pred
