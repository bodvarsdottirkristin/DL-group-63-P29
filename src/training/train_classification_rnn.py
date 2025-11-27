import numpy as np

import torch
import torch.nn as nn

import wandb
import os
import yaml

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.models.classification_rnn import ClassificationRNN, DEVICE
from src.utils.seed import set_seed
from src.utils.config import flatten_config


# TODO: 
#   Implement weight decay?
#   Implement train / test!!
#   Tengja WandB 

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

def train_classification_rnn(X: np.ndarray, y: np.ndarray, cfg):
    """
    Train an RNN classifier that predicts a cluster label from a past
    AIS sequence (e.g., 30×5). Handles train/val split, dataloaders,
    training loop, validation loop, and wandb logging.

    Inputs:
        X  — np.array (N, seq_len, features)
        y  — np.array (N,) cluster labels
        cfg — dict of training hyperparameters

    Returns:
        model — trained RNN classifier
        best_val_loss — lowest validation loss
    """
       
    # ------ Train / val split ------
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=cfg["val_split"],random_state=cfg["seed"],stratify=y) 
    y_train, y_val = np.asarray(y_train).astype(int), np.asarray(y_val).astype(int)


    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # ------ Model, optimizer, loss ------
    n_features = X_train.shape[-1]
    num_classes = cfg.get("num_classes", int(y.max()) + 1)

    model = ClassificationRNN(
        input_size=n_features,
        hidden_size=cfg["hidden_size"],
        num_classes=num_classes, 
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = nn.CrossEntropyLoss()

    print(f"Training started on device: {DEVICE}...")

    best_val_loss = float("inf")

    # ------ Training loop ------
    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
            xb = xb.to(DEVICE)          # (B, seq_len, num_features)
            yb = yb.to(DEVICE)          # (B,)

            opt.zero_grad()
            logits = model(xb)          # (B, num_classes)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            train_loss_total += loss.item()

            # accuracy
            preds = logits.argmax(dim=1)        # (B,)
            train_correct += (preds == yb).sum().item()
            train_samples += yb.size(0)

        train_loss = train_loss_total / len(train_loader)
        train_acc = train_correct / train_samples if train_samples > 0 else 0.0

        # ------ Validation ------
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = crit(logits, yb)
                val_loss_total += loss.item()

                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_samples += yb.size(0)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_samples if val_samples > 0 else 0.0
            
        best_val_loss = min(best_val_loss, val_loss)

        print(
            f"Epoch {epoch}/{cfg['epochs']} - "
            f"avg train loss: {train_loss:.4f} - "
            f"val loss: {val_loss:.4f}"
        )

        # ------ WandB logging ------
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "n_samples": X.shape[0],
                "n_train": X_train.shape[0],
                "n_val": X_val.shape[0],
                "best_val_loss": best_val_loss,
                "total_epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
            })

    return model, best_val_loss

def run_classification_train_rnn(X: np.ndarray, y: np.ndarray, sweep:bool =False):
    """
    Top-level training wrapper. Loads config, starts wandb,
    calls the training function, saves the model, and logs results.

    Inputs:
        X — past windows (N, seq_len, features)
        y — cluster labels
        sweep — enable wandb hyperparameter sweep mode

    Output:
        None (saves model + logs wandb stats)
    """


    with open("src/configs/classification_rnn.yaml") as file:
        CONFIG = yaml.safe_load(file)

    if sweep:
        wandb.init(project="classification_rnn", entity="ais-maritime-data")
        cfg = dict(wandb.config)

    else:
        conf = flatten_config(CONFIG)
        wandb.init(project="classification_rnn", entity="ais-maritime-data", config=conf)
        cfg = conf

    # set seeds and create output dir
    set_seed()

    # Nickname the run
    if wandb.run:
        import datetime
        # timestamp readable 
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.run.name = f"prediction_rnn_" + time_str
    
    # Train the autoencoder for this cluster
    model, best_val_loss = train_classification_rnn(X, y, cfg)

    # Save the trained model
    model_path = os.path.join(cfg["out_dir"], "classification_rnn_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    wandb.save(model_path)

    # sweep summary (single scalar per run)
    wandb.log({
        "best_val_loss": best_val_loss,
    })

    wandb.run.summary["best_val_loss"] = best_val_loss
    
    print("Training complete.")