import torch
import numpy as np
from tqdm import tqdm
from algae_fusion.config import DEVICE

def train_epoch(model, loader, criterion, optimizer, scaler_amp=None):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        if scaler_amp is not None:
            with torch.amp.autocast('cuda'):
                pred = model(x).squeeze(1)
                loss = criterion(pred, y)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            pred = model(x).squeeze(1)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def eval_epoch(model, loader, criterion, max_batches=None):
    model.eval()
    preds = []
    total = 0.0
    n_samples = 0
    n_batches = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Val", leave=False)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).squeeze(1)
            loss = criterion(pred, y)
            bs = x.size(0)
            total += loss.item() * bs
            n_samples += bs
            preds.append(pred.cpu().numpy())
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break
    if n_samples == 0:
        return 0.0, np.array([])
    return total / n_samples, np.concatenate(preds)
