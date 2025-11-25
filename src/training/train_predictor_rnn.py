import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import yaml
import wandb
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.models.trajectory_predictor import TrajectoryPredictor, trajectory_loss
from src.utils.seed import set_seed

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)



def _train_cluster_predictor(past: np.ndarray,future: np.ndarray,cfg: dict,cid: int):
    """
    Train TrajectoryPredictor on one cluster’s windows.
    past:   (N, 30, 5)
    future: (N, 30, 5)
    """
    # --- Train/Val split ---
    idx_train, idx_val = train_test_split(
        np.arange(len(past)),
        test_size=0.2,
        random_state=42
    )
    past_train   = torch.tensor(past[idx_train],   dtype=torch.float32)
    future_train = torch.tensor(future[idx_train], dtype=torch.float32)
    past_val     = torch.tensor(past[idx_val],     dtype=torch.float32)
    future_val   = torch.tensor(future[idx_val],   dtype=torch.float32)

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
        num_layers_encoder=1,
        num_layers_decoder=1
    ).to(cfg["device"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=1e-5,
    )

    print(f"🚀 Training TrajectoryPredictor on cluster {cid}, n={len(past)}")
    best_val_mse = float("inf")

    for epoch in range(1, 10 + 1):

        # --- TRAIN ---
        model.train()
        total = 0.0

        for xb_past, xb_future in tqdm(
            train_loader,
            desc=f"Cluster {cid} Epoch {epoch}/{cfg['epochs']}"
        ):
            xb_past   = xb_past.to(cfg["device"])
            xb_future = xb_future.to(cfg["device"])

            optimizer.zero_grad()

            preds = model(
                xb_past,
                target_length=xb_future.size(1),
                targets=xb_future,
                teacher_forcing_ratio=cfg["teacher_forcing"]
            )

            loss = trajectory_loss(preds, xb_future)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total += loss.item() * xb_past.size(0)

        train_mse = total / len(train_loader.dataset)

        # --- VALIDATION ---
        model.eval()
        with torch.no_grad():
            xv_past   = past_val.to(cfg["device"])
            xv_future = future_val.to(cfg["device"])
            preds = model(
                xv_past,
                target_length=xv_future.size(1),
                targets=None,
                teacher_forcing_ratio=0.0
            )
            val_mse = F.mse_loss(preds, xv_future, reduction="mean").item()

        best_val_mse = min(best_val_mse, val_mse)

        print(f"Epoch {epoch}/{cfg['epochs']} "
              f"- Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

        wandb.log({
            "epoch": epoch,
            "cluster_id": cid,
            "train_mse": train_mse,
            "val_mse": val_mse,
        })

    return model, best_val_mse

def run_predictor_cluster(past, future, labels, sweep=False):
    """
    Train predictor for ONE cluster, using wandb (same pattern as AE training).
    """

    # Load config
    with open("./predictor_cluster_config.yaml") as file:
        CONFIG = yaml.safe_load(file)

    if sweep:
        wandb.init(project="trajectory-predictor", entity="ais-prediction")
        cfg = dict(wandb.config)
    else:
        cfg_flat = {k: v["value"] if isinstance(v, dict) else v 
                    for k, v in CONFIG["parameters"].items()}
        wandb.init(project="trajectory-predictor", entity="ais-prediction", config=cfg_flat)
        cfg = cfg_flat

    cid = cfg["cluster_id"]

    # Set seed and device
    set_seed(cfg["seed"])
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Filter only this cluster’s windows
    Xp = past[labels == cid]
    Xf = future[labels == cid]
    print(f"Training cluster {cid}: n={len(Xp)}")

    # wandb naming
    import datetime
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name = f"pred_cluster_{cid}_{time_str}"

    wandb.log({
        "cluster_id": cid,
        "cluster_size": len(Xp),
    })

    # Train
    model, best_val = _train_cluster_predictor(Xp, Xf, cfg, cid)

    # Save model
    os.makedirs(cfg["out_dir"], exist_ok=True)
    model_path = os.path.join(cfg["out_dir"], f"pred_cluster_{cid}.pt")
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    wandb.log({"best_val_mse": best_val})
    wandb.run.summary["best_val_mse"] = best_val

    print(f"Model saved to {model_path}")
    print("Training complete.")


