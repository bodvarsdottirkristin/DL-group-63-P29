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
from tqdm import tqdm

from src.models.trajectory_predictor import TrajectoryPredictor, trajectory_loss, DEVICE
from src.utils.seed import set_seed
from src.utils.config import flatten_config

# path to .env file (one level up)
dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

# TODO: 
#       Normalize the data??

def _train_cluster_predictor(X_cluster: np.ndarray, y_cluster: np.ndarray, cid: int, cfg: dict):

    """
    Train an RNN that predicts a future AIS sequence based on a past
    AIS sequence (e.g., 30×5). Handles train/val split, dataloaders,
    training loop, validation loop, and wandb logging.

    Inputs:
        X_cluster  — np.array (N_cluster, seq_len, features)
        y_cluster  — np.array (N_cluster,) cluster labels
        cfg        — dict of training hyperparameters

    Returns:
        model — trained RNN 
        best_val_mse — lowest validation loss
    """

    # Traejectories
    X_train, X_val, y_train, y_val  = train_test_split(X_cluster, y_cluster, test_size=cfg["val_split"], random_state=cfg["seed"])
    
    # Normalize

    # Windowmaker
    X_train_w, X_val, y_train, y_val = window_maker(X_train, X_val, y_train, y_val)

    # NORMALIZE CODE FROM KRISTIN
    # train_all = np.concatenate([past_train_np, future_train_np], axis=1)  # (N_train, 60, 5)

    # mean = train_all.mean(axis=(0, 1), keepdims=True) 
    # std  = train_all.std(axis=(0, 1), keepdims=True)

    # past_train_norm   = (past_train_np   - mean) / std
    # future_train_norm = (future_train_np - mean) / std
    # past_val_norm     = (past_val_np     - mean) / std
    # future_val_norm   = (future_val_np   - mean) / std


    X_train_t   = torch.tensor(X_train, dtype=torch.float32)
    X_val_t     = torch.tensor(X_val  , dtype=torch.float32)

    y_train_t   = torch.tensor(y_train, dtype=torch.float32)
    y_val_t     = torch.tensor(y_val  , dtype=torch.float32)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,    # because all sequences are of the same size
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # --- Model, optimizer, loss ---
    n_features = X_train.shape[-1]

    model = TrajectoryPredictor(
        input_dim=n_features,
        output_dim=n_features,
        hidden_dim=cfg["hidden_size"],
        num_layers_encoder=cfg["num_layers_encoder"],
        num_layers_decoder=cfg["num_layers_decoder"]
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    print(f"Training started on device: {DEVICE}...")
    print(f"Cluster ID: {cid}, Cluster Size: {X_train_t.shape[0]}")
    
    best_val_mse = float("inf")

    # ------ Training loop ------ 
    train_samples = 0
    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        total = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Cluster {cid} Epoch {epoch}/{cfg['epochs']}"):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)


            opt.zero_grad()
            yb_pred = model(xb, target_length=yb.size(1), targets=yb, teacher_forcing_ratio=cfg["teacher_forcing"])
            
            loss = trajectory_loss(yb_pred, yb)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["max_norm"])
            opt.step()
            total += loss.item() * xb.size(0)
            train_samples += xb.size(0)


        train_mse = total / len(X_train_t)


        # --- VALIDATION ---
        model.eval()
        val_total = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                yb_pred = model(
                    xb,
                    target_length=yb.size(1),
                    targets=None,
                    teacher_forcing_ratio=0.0
                )

                loss = trajectory_loss(yb_pred, yb)
                bs = xb.size(0)

                val_total += loss * bs
                val_samples += bs

        val_mse = val_total / val_samples

        best_val_mse = min(best_val_mse,val_mse)

        print(
            f"Epoch {epoch}/{cfg['epochs']} - "
            f"Train MSE : {train_mse:.6f} - "
            f"Val MSE   : {val_mse:.6f}"
        )

        # ------- WandB logging -------
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "cluster_id": cid,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "cluster_id": cid,
                "n_samples": X_cluster.shape[0],
                "n_train": X_train.shape[0],
                "n_val": X_val.shape[0],
                "best_val_mse": best_val_mse,
                "total_epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
            })

    return model, best_val_mse

def run_predictor_cluster(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, cid: int, sweep=False):
    """
    Top-level training wrapper. Loads config, starts wandb,
    calls the training function, saves the model, and logs results.

    Inputs:
        X — all past windows                (N, seq_len, features)
        y — all future windows              (N, seq_len, features) 
        cluster_labels - all cluster lables (N,)
        cid - specified cluster
        sweep — enable wandb hyperparameter sweep mode

    Output:
        None (saves model + logs wandb stats)
    """
    # Load config with hyperparameters
    with open("src/configs/trajectory_predictor.yaml") as file:
        CONFIG = yaml.safe_load(file)

    if sweep:
        wandb.init(project="trajectory-predictor", entity="ais-maritime-data")
        cfg = dict(wandb.config)
    else:
        conf = flatten_config(CONFIG)
        wandb.init(project="trajectory-predictor", entity="ais-maritime-data", config=conf)
        cfg = conf
        
    # Set seeds
    set_seed()
    
    # Filter on cluster
    X_cluster = X[cluster_labels == cid]
    y_cluster = y[cluster_labels == cid]

    print(f"Training cluster {cid}: n={X_cluster.shape[0]}")

    # wandb naming
    if wandb.run:
        import datetime
        # timestamp readable
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.run.name = f"pred_cluster_{cid}_{time_str}"

    # info logs
    wandb.log({
        "cluster_id": cid,
        "cluster_size": X_cluster.shape[0],
    })

    # Call function to train the model
    model, best_val = _train_cluster_predictor(X_cluster, y_cluster, cid, cfg)

    # Save the trained model
    os.makedirs(cfg["out_dir"], exist_ok=True)
    model_path = os.path.join(cfg["out_dir"], f"pred_cluster_{cid}.pt")
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    # Sweep summary (single scalar per run)
    wandb.log({
        "best_val_mse": best_val
    })
    wandb.run.summary["best_val_mse"] = best_val

    print(f"Model saved to {model_path}")
    print("Training complete.")


