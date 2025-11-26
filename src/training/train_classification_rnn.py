import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from src.models.classification_rnn import ClassificationRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: 
#   Implement weight decay?
#   Implement train / test!!
#   Tengja WandB 

# path to .env file (one level up)
# dotenv_path = Path(__file__).parents[1] / '.env'
# load_dotenv(dotenv_path=dotenv_path)

# WANDB_API_KEY = os.getenv("WANDB_API_KEY")
# wandb.login(key=WANDB_API_KEY)

dotenv_path = Path(__file__).parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path)

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login(key=WANDB_API_KEY)

# ...existing code...

def flatten_config(config):
    """Extracts flat config from sweep-style YAML for normal runs."""
    params = config.get("parameters", {})
    flat = {}
    for k, v in params.items():
        if "value" in v:
            flat[k] = v["value"]
        elif "values" in v:
            flat[k] = v["values"][0]  # pick first for normal run
        elif "min" in v and "max" in v:
            flat[k] = v.get("min")    # pick min for normal run
        else:
            flat[k] = v
    return flat


with open("./pred_rnn_config.yaml") as file:
    CONFIG = yaml.safe_load(file)

def train_classification_rnn(X: np.ndarray, y: np.ndarray, cfg):
    """
    Train a classification RNN so that it can classify a cluster
    based on a past sequence.

    Parameters
    ----------
    X : np.ndarray of shape (N, length_seq, num_features) - past_window
    y : np.ndarray of shape (N,) with cluster ids for each window
    cfg : dict-like with hyperparameters
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

    # ------ Model, optimizer, loss ------
    n_features = X_train.shape[-1]
    num_classes = cfg.get("num_classes", int(y.max()) + 1)

    model = ClassificationRNN(
        input_size=n_features,
        hidden_size=cfg["hidden_size"],
        num_classes=num_classes,   # adjust if your class uses a different name
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()

    print(f"Training started on device: {DEVICE}...")

    best_val_loss = float("inf")

    # ------ Training loop ------
    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        total = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
            xb = xb.to(DEVICE)          # (B, seq_len, num_features)
            yb = yb.to(DEVICE)          # (B,)

            opt.zero_grad()
            logits = model(xb)          # (B, num_classes)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            total += loss.item()

        train_loss = total / len(train_loader)

        # ------ Validation ------
        xv = X_val_t.to(DEVICE)
        yv = y_val_t.to(DEVICE)

        with torch.no_grad():
            logits_val = model(xv)
            val_loss = crit(logits_val, yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"Epoch {epoch}/{cfg['epochs']} - "
            f"avg train loss: {train_loss:.4f} - "
            f"val loss: {val_loss:.4f}"
        )

        # ------ WandB logging ------
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "avg_train_loss": train_loss,
                "avg_val_loss": val_loss,
            })

    return model, best_val_loss


def evaluate(model, loader, criterion, device: str = "cuda"):

    model.to(DEVICE)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

    return total_loss / len(loader)

def run_classification_train_rnn(X: np.ndarray, y: np.ndarray, sweep:bool =False):
    if sweep:
        wandb.init(project="maritime-data", entity="ais-maritime-data")
        cfg = dict(wandb.config)

    else:
        conf = flatten_config(CONFIG)
        wandb.init(project="maritime-data", entity="ais-maritime-data", config=conf)
        cfg = conf

    # set seeds and create output dir
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # Nickname the run
    if wandb.run:
        import datetime
        # timestamp readable 
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.run.name = f"prediction_rnn_" + time_str
    
    # Train the autoencoder for this cluster
    model, best_val = train_classification_rnn(X, y, cfg)

    # Save the trained model
    model_path = os.path.join(cfg["out_dir"], f"prediction_rnn_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    wandb.save(model_path)

    # sweep summary (single scalar per run)
    wandb.log({
        "best_val_loss": best_val_loss,
    })
    wandb.run.summary["best_val_loss"] = best_val_loss
    
    print("Training complete.")