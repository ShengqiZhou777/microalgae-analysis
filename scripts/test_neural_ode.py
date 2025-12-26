
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from algae_fusion.models.ode import NeuralODEPure, ODERNN

def test_pure_neural_ode():
    print("\n--- Testing Pure Neural ODE ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup
    input_dim = 2
    model = NeuralODEPure(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 2. Fake Data (Spiral)
    # y0 = [2, 0]
    y0 = torch.tensor([[2.0, 0.0]]).to(device) # Batch size 1
    t = torch.linspace(0, 25, 100).to(device)
    
    # Forward Pass
    print("  Running Forward Pass...")
    try:
        pred_y = model(y0, t) # Shape [B, T, D]
        print(f"  Output Shape: {pred_y.shape} (Expected: [1, 100, 2])")
    except Exception as e:
        print(f"  [Stack Trace] Forward failed: {e}")
        raise e

    # Backward Pass
    print("  Running Backward Pass...")
    target = torch.zeros_like(pred_y).to(device) # Dummy target
    loss = torch.mean((pred_y - target)**2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check if weights changed
    print(f"  Loss: {loss.item():.4f}")
    print("  Gradients checked OK.")

def test_ode_rnn():
    print("\n--- Testing ODE-RNN (Irregular Sampling) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup
    input_dim = 3   # e.g., features
    latent_dim = 4  # hidden state
    model = ODERNN(input_dim, latent_dim).to(device)
    
    # 2. Fake Data with Irregular Time
    batch_size = 2
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    # Time points must be strictly increasing for odeint
    t = torch.sort(torch.rand(seq_len) * 10)[0].to(device)
    
    # Mask (some observations missing)
    mask = torch.randint(0, 2, (batch_size, seq_len)).float().to(device)
    
    print(f"  Input: {x.shape}, Time points: {len(t)}")
    
    # Forward
    out = model(x, t, mask=mask)
    print(f"  Output Shape: {out.shape} (Expected: [{batch_size}, {seq_len}, {latent_dim}])")
    
    # Backward
    loss = out.sum()
    loss.backward()
    print("  Backward Pass OK.")

if __name__ == "__main__":
    try:
        test_pure_neural_ode()
        test_ode_rnn()
        print("\n[SUCCESS] All Neural ODE tests passed!")
    except ImportError as e:
        print(f"\n[FAILURE] Missing Dependency: {e}")
        print("Please ensure 'torchdiffeq' is installed: pip install torchdiffeq")
    except Exception as e:
        print(f"\n[FAILURE] Test Crashed: {e}")
        import traceback
        traceback.print_exc()
