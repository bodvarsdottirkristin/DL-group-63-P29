import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# import yaml
# import wandb
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import glob
import pandas as pd

from src.models.trajectory_predictor import TrajectoryPredictor, trajectory_loss
from src.utils.seed import set_seed
from src.datasets.window_maker import load_parquet_files

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

#WANDB_API_KEY = os.getenv("WANDB_API_KEY")
#wandb.login(key=WANDB_API_KEY)

def _train_cluster_predictor(past: np.ndarray,future: np.ndarray, cfg: dict,cid: int):
    """
    Train TrajectoryPredictor on one cluster’s windows.
    past:   (N, 30, 5)
    future: (N, 30, 5)
    cfg: config dict with keys
    cid: cluster id
    """

    # --- Train/Val split ---
    idx_train, idx_val = train_test_split(
        np.arange(len(past)),
        test_size=0.2,
        random_state=42,
    )


    past_train_np   = past[idx_train]    # (N_train, 30, 5)
    future_train_np = future[idx_train]  # (N_train, 30, 5)
    past_val_np     = past[idx_val]
    future_val_np   = future[idx_val]

    # Normalize based on train stats
    train_all = np.concatenate([past_train_np, future_train_np], axis=1)  # (N_train, 60, 5)

    mean = train_all.mean(axis=(0, 1), keepdims=True) 
    std  = train_all.std(axis=(0, 1), keepdims=True)

    past_train_norm   = (past_train_np   - mean) / std
    future_train_norm = (future_train_np - mean) / std
    past_val_norm     = (past_val_np     - mean) / std
    future_val_norm   = (future_val_np   - mean) / std

    past_train   = torch.tensor(past_train_norm,   dtype=torch.float32)
    future_train = torch.tensor(future_train_norm, dtype=torch.float32)
    past_val     = torch.tensor(past_val_norm,     dtype=torch.float32)
    future_val   = torch.tensor(future_val_norm,   dtype=torch.float32)

    #past_train   = torch.tensor(past_train_np,   dtype=torch.float32)
    #future_train = torch.tensor(future_train_np, dtype=torch.float32)
    #past_val     = torch.tensor(past_val_np,     dtype=torch.float32)
    #future_val   = torch.tensor(future_val_np,   dtype=torch.float32)
    
    train_loader = DataLoader(
        TensorDataset(past_train, future_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        TensorDataset(past_val, future_val),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # --- Model ---
    model = TrajectoryPredictor(
        input_dim=5,
        output_dim=5,
        hidden_dim=20,
        num_layers_encoder=2,
        num_layers_decoder=2
    ).to(cfg["device"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    print(f"Training TrajectoryPredictor on cluster {cid}, n={len(past)}")

    best_val_mse = float("inf")
    best_epoch = -1

    train_losses = []
    val_losses = []

    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for xb_past, xb_future in tqdm(
            train_loader,
            desc=f"Cluster {cid} Epoch {epoch}/{cfg['epochs']}"
        ):
            xb_past   = xb_past.to(cfg["device"])  # (B, 30, 5)
            xb_future = xb_future.to(cfg["device"])  # (B, 30, 5)
            
            # All sequences are 30 timesteps
            # lengths = torch.full((xb_past.size(0),), xb_past.size(1), dtype=torch.long, device=cfg["device"])
            # print("lengths", lengths)

            optimizer.zero_grad()

            preds = model(
                x=xb_past,
                target_length=xb_future.size(1),
                targets=xb_future,
                teacher_forcing_ratio=cfg["teacher_forcing"]
            )

            loss = trajectory_loss(preds, xb_future)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * xb_past.size(0)
            total_train_samples += xb_past.size(0)

        train_mse = total_train_loss / total_train_samples
        train_losses.append(train_mse)

        # --- VALIDATION ---
        model.eval()
        with torch.no_grad():
            xv_past   = past_val.to(cfg["device"])
            xv_future = future_val.to(cfg["device"])

            # lengths_val = torch.full((xv_past.size(0),), xv_past.size(1), dtype=torch.long, device=cfg["device"])

            preds = model(
                x=xv_past,
                target_length=xv_future.size(1),
                targets=None,
                teacher_forcing_ratio=0.0
            )
            val_mse = F.mse_loss(preds, xv_future, reduction="mean").item()
            val_losses.append(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch

        print(f"Epoch {epoch}/{cfg['epochs']} "
              f"- Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

        # wandb.log({
        #     "epoch": epoch,
        #     "cluster_id": cid,
        #     "train_mse": train_mse,
        #     "val_mse": val_mse,
        # })
    stats = {
        "cluster_id": cid,
        "n_samples": len(past),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "train_mse_curve": train_losses,
        "val_mse_curve": val_losses,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "final_train_mse": train_losses[-1],
        "final_val_mse": val_losses[-1],
        "final_train_rmse": float(np.sqrt(train_losses[-1])),
        "final_val_rmse": float(np.sqrt(val_losses[-1])),
        "total_epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
    }
        
    return model, stats

def run_predictor_cluster(past, future, sweep=False, cfg=None):
    """
    Train predictor for one cluster.
    
    Args:
        past, future: arrays of past/future windows
        labels: cluster labels for each window
        sweep: (deprecated, not used)
        cfg: config dict with keys: batch_size, epochs, teacher_forcing, seed, cluster_id, device, out_dir
    """

    # # Load config (commented out - pass cfg as argument instead)
    # with open("./predictor_cluster_config.yaml") as file:
    #     CONFIG = yaml.safe_load(file)

    # if sweep:
    #     wandb.init(project="trajectory-predictor", entity="ais-prediction")
    #     cfg = dict(wandb.config)
    # else:
    #     cfg_flat = {k: v["value"] if isinstance(v, dict) else v 
    #                 for k, v in CONFIG["parameters"].items()}
    #     wandb.init(project="trajectory-predictor", entity="ais-prediction", config=cfg_flat)
    #     cfg = cfg_flat
    
    if cfg is None:
        raise ValueError("Must pass cfg dict with training parameters")

    cid = cfg["cluster_id"]

    # Set seed and device
    set_seed(cfg["seed"])
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Filter only this cluster’s windows
    print(f"Training cluster {cid}: n={len(past)}")

    # wandb naming
    import datetime
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # wandb.run.name = f"pred_cluster_{cid}_{time_str}"

    # wandb.log({
    #     "cluster_id": cid,
    #     "cluster_size": len(Xp),
    # })

    # Train
    model, stats = _train_cluster_predictor(past, future, cfg, cid)

    # Save model
    os.makedirs(cfg["out_dir"], exist_ok=True)
    model_path = os.path.join(cfg["out_dir"], f"pred_cluster_{cid}.pt")
    torch.save(model.state_dict(), model_path)
    # wandb.save(model_path)

    # wandb.log({"best_val_mse": stats["best_val_mse"]})
    # wandb.run.summary["best_val_mse"] = stats["best_val_mse"]

    print(f"Model saved to {model_path}")
    print("Training complete.")



if __name__ == "__main__":
    
    past, future, cluster_id = load_parquet_files(input_path='data/aisdk/processed/windows_30_30', cluster_id=1)

    print(past.shape, future.shape)
    run_predictor_cluster(past, future, sweep=False, cfg={
        "batch_size": 64,
        "epochs": 10,
        "teacher_forcing": 0.5,
        "seed": 42,
        "device": "cpu",
        "cluster_id": 1,
        "weight_decay": 1e-5,
        "lr": 0.001,
        "out_dir": "models/predictor_models"
    })
        