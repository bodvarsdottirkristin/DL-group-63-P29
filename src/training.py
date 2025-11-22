import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from src.datasets.AISDataSet import get_dataloaders
from src.models.VRAE import VRAE, vae_loss
from src.models.HDBSCAN import cluster_latent_space
from src.preprocessing.trajectory_builder import build_trajectories_from_parquet
from src.visualization.visualize_hdbscan import plot_hdbscan_latent


def train_epoch(model, dataloader, optimizer, device, beta=1.0):
    model.train()
    total_loss = total_recon = total_kld = 0.0
    n_batches = 0

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

        total_loss += loss.item()
        total_recon += recon.item()
        total_kld += kld.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_kld / n_batches,
    )



def eval_epoch(model, dataloader, device, beta=1.0):
    model.eval()
    total_loss = total_recon = total_kld = 0.0
    n_batches = 0

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

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            n_batches += 1

    return (
        total_loss / n_batches,
        total_recon / n_batches,
        total_kld / n_batches,
    )

def collect_latent_mus(model, dataloader, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to AIS parquet file")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="vrae_checkpoint.pt")
    parser.add_argument(
    "--mmsi",
    type=str,
    default=None,
    help="Comma-separated list of MMSIs to include (for debugging / subset).",
)
    args = parser.parse_args()

    if args.mmsi is not None:
      # e.g. "--mmsi 123456789,987654321,111222333"
      mmsi_whitelist = [m.strip() for m in args.mmsi.split(",") if m.strip()]
    else:
        mmsi_whitelist = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Build (N, T, D) trajectories from parquet
    print("Loading trajectories from parquet...")
    trajectories = build_trajectories_from_parquet(
        parquet_path=args.data_path,
        seq_len=args.seq_len,
        step=1,
        mmsi_whitelist=mmsi_whitelist,
    )
    print(f"Trajectories shape: {trajectories.shape}")  # (N, T, D)

    # 2) Build train/val/test loaders
    train_loader, val_loader, test_loader, mean, std = get_dataloaders(
        trajectories,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    input_dim = trajectories.shape[2]

    # 3) Init VRAE
    model = VRAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Train loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_kld = train_epoch(
            model, train_loader, optimizer, device, beta=args.beta
        )
        val_loss, val_recon, val_kld = eval_epoch(
            model, val_loader, device, beta=args.beta
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} (Recon {train_recon:.4f}, KL {train_kld:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Recon {val_recon:.4f}, KL {val_kld:.4f})"
        )

    # 5) Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved VRAE checkpoint to {args.save_path}")

    # 6) Latent mus on test set
    print("Encoding test set to latent space...")
    mus = collect_latent_mus(model, test_loader, device)
    print(f"Latent mus shape: {mus.shape}")  # (N_test, latent_dim)

    # 7) HDBSCAN clustering
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
