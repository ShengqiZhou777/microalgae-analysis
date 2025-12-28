import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.getcwd())

from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.models.ode import GrowthODE


def load_ode_config(target, condition):
    config_path = f"weights/ode_{target}_{condition}_config.json"
    defaults = {
        "latent_dim": 64,
        "ode_hidden_dim": 128,
        "decoder_hidden": 64,
        "decoder_dropout": 0.2,
        "population_mean": False,
    }
    if not os.path.exists(config_path):
        return defaults
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return {**defaults, **config}


def aggregate_population_mean(df_in, split_label="TEST"):
    if df_in.empty:
        return df_in
    df_out = df_in.copy()
    df_out["time"] = pd.to_numeric(df_out["time"], errors="coerce")
    group_cols = ["condition", "time"]
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    grouped = df_out.groupby(group_cols, as_index=False)[numeric_cols].mean()
    grouped["split_set"] = split_label
    if "Source_Path" in df_out.columns:
        src = df_out.groupby(group_cols)["Source_Path"].first().reset_index()
        grouped = grouped.merge(src, on=group_cols, how="left")
    if "file" in df_out.columns:
        grouped["file"] = grouped.apply(
            lambda row: f"{row['condition']}_{row['time']}_mean",
            axis=1,
        )
    return grouped


def build_model(input_dim, ode_config):
    latent_dim = ode_config["latent_dim"]
    ode_hidden_dim = ode_config["ode_hidden_dim"]
    ode_core = GrowthODE(input_dim=input_dim, latent_dim=latent_dim, ode_hidden_dim=ode_hidden_dim).to(DEVICE)

    class ODEProjector(nn.Module):
        def __init__(self, ode, latent_dim, decoder_hidden, decoder_dropout):
            super().__init__()
            self.ode = ode
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden),
                nn.Tanh(),
                nn.Dropout(p=decoder_dropout),
                nn.Linear(decoder_hidden, 1),
            )

        def forward(self, x, t, mask):
            h = self.ode.ode_net(x, t, mask)
            return self.proj(h).squeeze(-1)

    model = ODEProjector(
        ode_core,
        latent_dim,
        ode_config["decoder_hidden"],
        ode_config["decoder_dropout"],
    ).to(DEVICE)
    return model


def predict_sequence(model, features, times, time_scale):
    x = torch.tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    t = torch.tensor(times, dtype=torch.float32, device=DEVICE)
    t_grid = t / time_scale
    mask = torch.ones(1, len(times), device=DEVICE)
    with torch.no_grad():
        preds = model(x, t_grid, mask).squeeze(0).cpu().numpy()
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Dry_Weight")
    parser.add_argument("--condition", type=str, default="All")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--population-mean", action="store_true", help="Force population-mean aggregation for inference.")
    args = parser.parse_args()

    df_test = pd.read_csv("data/dataset_test.csv")
    if args.condition != "All":
        df_test = df_test[df_test["condition"] == args.condition].reset_index(drop=True)

    ode_config = load_ode_config(args.target, args.condition)
    use_population_mean = bool(ode_config.get("population_mean")) or args.population_mean

    scaler_path = f"weights/ode_{args.target}_{args.condition}_scaler.joblib"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)

    model_path = f"weights/ode_{args.target}_{args.condition}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    df_pred = df_test.copy()
    if use_population_mean:
        df_pred = aggregate_population_mean(df_pred, "TEST")

    feature_cols = [c for c in df_pred.columns if c not in NON_FEATURE_COLS + [args.target]]
    time_scale = pd.to_numeric(df_pred["time"], errors="coerce").max()
    time_scale = float(time_scale) if pd.notna(time_scale) and time_scale > 0 else 1.0

    model = build_model(len(feature_cols), ode_config)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    group_col = "condition" if use_population_mean else ("group_idx" if "group_idx" in df_pred.columns else "file")
    df_pred = df_pred.sort_values([group_col, "time"]).reset_index(drop=True)

    predictions = np.zeros(len(df_pred), dtype=np.float32)
    for _, group in df_pred.groupby(group_col):
        times = group["time"].values.astype(np.float32)
        features = group[feature_cols].values.astype(np.float32)
        pred_scaled = predict_sequence(model, features, times, time_scale)
        predictions[group.index.values] = pred_scaled

    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    df_pred[f"{args.target}_pred"] = predictions_inv

    if use_population_mean:
        df_out = df_test.merge(
            df_pred[["condition", "time", f"{args.target}_pred"]],
            on=["condition", "time"],
            how="left",
        )
    else:
        df_out = df_test.copy()
        df_out[f"{args.target}_pred"] = df_pred[f"{args.target}_pred"]

    os.makedirs("results/ode_predictions", exist_ok=True)
    output_path = args.output or f"results/ode_predictions/{args.target}_{args.condition}_test_predictions.csv"
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    # --- Visualization ---
    plot_dir = "results/ode_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot true vs pred for each group
    group_col = "condition" if use_population_mean else ("group_idx" if "group_idx" in df_pred.columns else "file")
    unique_groups = df_pred[group_col].unique()
    
    # Limit to reasonable number for visualization if not population mean
    if not use_population_mean and len(unique_groups) > 20:
        print(f"Plotting subset of 20/{len(unique_groups)} groups for trajectory viz...")
        viz_groups = unique_groups[:20]
    else:
        viz_groups = unique_groups
        
    for gid in viz_groups:
        sub = df_pred[df_pred[group_col] == gid]
        sub = sub.sort_values("time")
        
        # True
        if args.target in sub.columns:
            plt.plot(sub["time"], sub[args.target], 'o-', alpha=0.3, color='black', label='Ground Truth' if gid == viz_groups[0] else "")
            
        # Pred
        plt.plot(sub["time"], sub[f"{args.target}_pred"], '--', alpha=0.7, color='red', label='ODE Prediction' if gid == viz_groups[0] else "")

    plt.xlabel("Time (h)")
    plt.ylabel(args.target)
    plt.title(f"ODE Test Predictions (Trajectory): {args.target} ({args.condition})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    traj_path = f"{plot_dir}/{args.target}_{args.condition}_trajectory.png"
    plt.savefig(traj_path, dpi=150)
    plt.close()
    print(f"Saved trajectory visualization to {traj_path}")

    # --- Scatter Plot (All Data) ---
    if args.target in df_out.columns:
        y_true = df_out[args.target].values
        y_pred = df_out[f"{args.target}_pred"].values
        
        # Remove NaNs
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) > 0:
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
            
            # Diagonal line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            plt.xlabel(f"True {args.target}")
            plt.ylabel(f"Predicted {args.target}")
            plt.title(f"ODE Scatter: {args.target} ({args.condition})\nR2={r2:.3f}, RMSE={rmse:.3f}")
            plt.grid(True, alpha=0.3)
            
            scatter_path = f"{plot_dir}/{args.target}_{args.condition}_scatter.png"
            plt.savefig(scatter_path, dpi=150)
            plt.close()
            print(f"Saved scatter visualization to {scatter_path}")


if __name__ == "__main__":
    main()
