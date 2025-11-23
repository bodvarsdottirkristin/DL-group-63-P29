import argparse
import numpy as np
import torch

from src.datasets.AISDataSet import get_dataloaders
from src.models.vrae import VRAE, vae_loss
from src.models.hdbscan import cluster_latent_space
from src.preprocessing.trajectory_builder import build_trajectories_from_parquet
from src.visualization.visualize_hdbscan import plot_hdbscan_latent
from src.utils.seed import set_seed

def train_one_epoch(model, dataloader, optimizer, device, beta: float = 1.0):
    """
    One training epoch over the dataloader.
    Returns average total loss, recon loss, and KL loss.
    """
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    total_samples = 0

    for x in dataloader:
        x = x.to(device)  # (B, T, D)
        batch_size, seq_len, _ = x.shape

        lengths = torch.full(
            (batch_size,),
            seq_len,
            dtype=torch.long,
            device=device,
        )

        reconstruction, mu, logvar = model(x, lengths)
        loss, recon, kld = vae_loss(reconstruction, x, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_kld += kld.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_recon = total_recon / total_samples
    avg_kld = total_kld / total_samples

    return avg_loss, avg_recon, avg_kld


def evaluate(model, dataloader, device, beta: float = 1.0):
    """
    Evaluation (no grad) over val/test dataloader.
    Returns average total loss, recon loss, and KL loss.
    """
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    total_samples = 0

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            batch_size, seq_len, _ = x.shape

            lengths = torch.full(
                (batch_size,),
                seq_len,
                dtype=torch.long,
                device=device,
            )

            reconstruction, mu, logvar = model(x, lengths)
            loss, recon, kld = vae_loss(reconstruction, x, mu, logvar, beta=beta)

            total_loss += loss.item() * batch_size
            total_recon += recon.item() * batch_size
            total_kld += kld.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_recon = total_recon / total_samples
    avg_kld = total_kld / total_samples

    return avg_loss, avg_recon, avg_kld


def collect_latent_mus(model, dataloader, device):
    """
    Run encoder on all sequences in dataloader and collect mu.
    Returns mus: (N, latent_dim)
    """
    model.eval()
    mus = []

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            batch_size, seq_len, _ = x.shape
            lengths = torch.full(
                (batch_size,),
                seq_len,
                dtype=torch.long,
                device=device,
            )
            mu, logvar = model.encode(x, lengths)  # (B, latent_dim)
            mus.append(mu.cpu().numpy())

    return np.concatenate(mus, axis=0)  # (N, latent_dim)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VRAE on AIS trajectories")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to AIS parquet file (folder or file)",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help="Sequence length (number of time steps per trajectory)",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dim of GRU encoder/decoder",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Latent dim (mu/logvar)",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight on KL term in VAE loss",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_path",
        type=str,
        default="vrae_checkpoint.pt",
        help="Where to save best VRAE weights",
    )

    parser.add_argument(
        "--mmsi",
        type=str,
        default=None,
        help=(
            "Comma-separated list of MMSIs to include (for debugging / subset). "
            "If None, use all MMSIs."
        ),
    )

    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Stride when building fixed-length trajectories from AIS",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------- setup -------------
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mmsi is not None:
        mmsi_whitelist = [m.strip() for m in args.mmsi.split(",") if m.strip()]
        print(f"Restricting to MMSIs: {mmsi_whitelist}")
    else:
        mmsi_whitelist = None

    # ------------- data -------------
    print("Loading trajectories from parquet...")
    trajectories = build_trajectories_from_parquet(
        parquet_path=args.data_path,
        seq_len=args.seq_len,
        step=args.step,
        mmsi_whitelist=mmsi_whitelist,
    )
    print(f"Trajectories shape: {trajectories.shape}")  # (N, T, D)

    train_loader, val_loader, test_loader, mean, std = get_dataloaders(
        trajectories,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    input_dim = trajectories.shape[2]

    # ------------- model & optimizer -------------
    model = VRAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    # ------------- training loop -------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_recon, train_kld = train_one_epoch(
            model, train_loader, optimizer, device, beta=args.beta
        )
        val_loss, val_recon, val_kld = evaluate(
            model, val_loader, device, beta=args.beta
        )

        print(
            f"  Train: loss={train_loss:.4f}, recon={train_recon:.4f}, KL={train_kld:.4f}"
        )
        print(
            f"  Val:   loss={val_loss:.4f},   recon={val_recon:.4f},   KL={val_kld:.4f}"
        )

        # save best model wrt validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> New best model saved to {args.save_path}")

    # ------------- load best model (optional) -------------
    print(f"\nLoading best model from {args.save_path}")
    model.load_state_dict(torch.load(args.save_path, map_location=device))

    # ------------- encode test set & cluster -------------
    print("Encoding test set to latent space...")
    mus = collect_latent_mus(model, test_loader, device)
    print(f"Latent mus shape: {mus.shape}")  # (N_test, latent_dim)

    print("Clustering latent space with HDBSCAN...")
    labels, clusterer, scaler = cluster_latent_space(mus)

    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster labels and counts (including -1 for noise):")
    for lab, cnt in zip(unique, counts):
        print(f"  label={lab:3d}  count={cnt}")

    n_clusters = (unique != -1).sum()
    print(f"Number of clusters (excluding noise): {n_clusters}")

    plot_hdbscan_latent(mus, labels, out_path="plots/hdbscan_latent.png")


if __name__ == "__main__":
    main()
