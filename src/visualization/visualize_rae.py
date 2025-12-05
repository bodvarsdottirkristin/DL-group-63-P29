import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

import plotly.express as px
from pyproj import Transformer

import os
import joblib

import importlib
import json
import itertools

def visualize_rae_latent_space(model, data_loader, device, save_dir='plots/rae_visualizations', max_samples=5000):
    """
    Comprehensive latent space visualization for RAE clustering analysis.
    
    Args:
        model: Trained RAE model
        data_loader: DataLoader with trajectories
        device: torch device
        save_dir: Directory to save plots
        max_samples: Max number of samples to visualize (for performance)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Collect latent representations
    all_latents = []
    all_lengths = []
    recon_errors = []  # Calculate per-sample immediately instead of storing full reconstructions
    
    with torch.no_grad():
        for batch_data, lengths in data_loader:
            batch_data = batch_data.to(device)
            lengths = lengths.to(device)
            
            # Get latent representations and reconstructions
            reconstruction, z = model(batch_data, lengths, teacher_forcing_ratio=0.0)
            
            # Calculate reconstruction error per sample in this batch
            for i in range(batch_data.size(0)):
                actual_len = lengths[i].item()
                orig = batch_data[i, :actual_len, :]
                recon = reconstruction[i, :actual_len, :]
                mse = ((orig - recon) ** 2).mean().item()
                recon_errors.append(mse)
            
            all_latents.append(z.cpu())
            all_lengths.append(lengths.cpu())
            
            if len(all_latents) * batch_data.size(0) >= max_samples:
                break
    
    # Concatenate all data
    all_latents = torch.cat(all_latents, dim=0).numpy()
    all_lengths = torch.cat(all_lengths, dim=0).numpy()
    recon_errors = np.array(recon_errors)
    
    print(f"Collected {len(all_latents)} samples for visualization")
    
    latent_dim = all_latents.shape[1]
    
    # ============================================================
    # 1. PCA Visualization (2D and 3D)
    # ============================================================
    from sklearn.decomposition import PCA
    
    fig = plt.figure(figsize=(20, 5))
    
    # 2D PCA
    pca_2d = PCA(n_components=2)
    latents_pca_2d = pca_2d.fit_transform(all_latents)
    
    ax1 = fig.add_subplot(141)
    scatter = ax1.scatter(latents_pca_2d[:, 0], latents_pca_2d[:, 1], 
                         c=all_lengths, cmap='viridis', alpha=0.6, s=15)
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)')
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)')
    ax1.set_title('RAE Latent Space - PCA 2D\nColored by Trajectory Length')
    plt.colorbar(scatter, ax=ax1, label='Trajectory Length')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(142)
    scatter2 = ax2.scatter(latents_pca_2d[:, 0], latents_pca_2d[:, 1], 
                          c=recon_errors, cmap='plasma', alpha=0.6, s=15)
    ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)')
    ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)')
    ax2.set_title('RAE Latent Space - PCA 2D\nColored by Reconstruction Error')
    plt.colorbar(scatter2, ax=ax2, label='MSE')
    ax2.grid(True, alpha=0.3)
    
    # 3D PCA
    pca_3d = PCA(n_components=3)
    latents_pca_3d = pca_3d.fit_transform(all_latents)
    
    ax3 = fig.add_subplot(143, projection='3d')
    scatter3d = ax3.scatter(latents_pca_3d[:, 0], latents_pca_3d[:, 1], latents_pca_3d[:, 2],
                           c=all_lengths, cmap='viridis', alpha=0.6, s=15)
    ax3.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
    ax3.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
    ax3.set_title('RAE Latent Space - PCA 3D')
    
    # Explained variance plot
    ax4 = fig.add_subplot(144)
    pca_full = PCA()
    pca_full.fit(all_latents)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax4.plot(range(1, len(cumsum)+1), cumsum, 'bo-')
    ax4.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax4.set_xlabel('Number of Components')
    ax4.set_ylabel('Cumulative Explained Variance')
    ax4.set_title('PCA Scree Plot')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_pca_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 2. t-SNE Visualization
    # ============================================================
    from sklearn.manifold import TSNE
    
    print("Computing t-SNE... (this may take a minute)")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_tsne = tsne.fit_transform(all_latents)
    
    scatter1 = axes[0].scatter(latents_tsne[:, 0], latents_tsne[:, 1], 
                              c=all_lengths, cmap='viridis', alpha=0.6, s=15)
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    axes[0].set_title('t-SNE Visualization\nColored by Length')
    plt.colorbar(scatter1, ax=axes[0], label='Trajectory Length')
    axes[0].grid(True, alpha=0.3)
    
    scatter2 = axes[1].scatter(latents_tsne[:, 0], latents_tsne[:, 1], 
                              c=recon_errors, cmap='plasma', alpha=0.6, s=15)
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].set_title('t-SNE Visualization\nColored by Reconstruction Error')
    plt.colorbar(scatter2, ax=axes[1], label='MSE')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_tsne_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 3. Latent Dimension Analysis
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    dim_indices = range(latent_dim)
    
    # Distribution per dimension
    axes[0].boxplot([all_latents[:, i] for i in dim_indices], labels=dim_indices)
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Latent Value')
    axes[0].set_title('Distribution per Dimension')
    axes[0].grid(True, alpha=0.3)
    
    # Variance per dimension (activity indicator)
    var_per_dim = all_latents.var(axis=0)
    axes[1].bar(dim_indices, var_per_dim, alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Variance per Dimension\n(Higher = more discriminative for clustering)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_dimension_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 4. Clustering Quality Indicators
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Distance matrix heatmap (subsample for performance)
    n_subsample = min(500, len(all_latents))
    subsample_idx = np.random.choice(len(all_latents), n_subsample, replace=False)
    subsample_latents = all_latents[subsample_idx]
    
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(subsample_latents, metric='euclidean'))
    
    im1 = axes[0, 0].imshow(distances, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Pairwise Distance Matrix\n({n_subsample} samples)')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0, 0], label='Euclidean Distance')
    
    # Distance distribution
    upper_tri_distances = distances[np.triu_indices_from(distances, k=1)]
    axes[0, 1].hist(upper_tri_distances, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 1].set_xlabel('Pairwise Distance')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Pairwise Distances\n' + 
                        f'Mean={upper_tri_distances.mean():.3f}, ' + 
                        f'Std={upper_tri_distances.std():.3f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction error distribution
    axes[1, 0].hist(recon_errors, bins=50, alpha=0.7, edgecolor='black', color='coral')
    axes[1, 0].axvline(recon_errors.mean(), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean={recon_errors.mean():.4f}')
    axes[1, 0].set_xlabel('Reconstruction MSE')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reconstruction Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latent space density (norm distribution)
    latent_norms = np.linalg.norm(all_latents, axis=1)
    axes[1, 1].hist(latent_norms, bins=50, alpha=0.7, edgecolor='black', color='mediumseagreen')
    axes[1, 1].axvline(latent_norms.mean(), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean={latent_norms.mean():.2f}')
    axes[1, 1].set_xlabel('Latent Vector Norm')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Latent Space Density\n(Distribution of ||z||)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/04_clustering_quality.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 5. Trajectory Reconstruction Examples (sample from data loader)
    # ============================================================
    print("Generating reconstruction examples...")
    
    # Get a fresh batch for visualization
    example_batch, example_lengths = next(iter(data_loader))
    example_batch = example_batch.to(device)
    example_lengths = example_lengths.to(device)
    
    with torch.no_grad():
        example_recon, _ = model(example_batch, example_lengths, teacher_forcing_ratio=0.0)
    
    example_batch = example_batch.cpu().numpy()
    example_recon = example_recon.cpu().numpy()
    example_lengths = example_lengths.cpu().numpy()
    
    n_viz = min(6, example_batch.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for plot_idx in range(n_viz):
        ax = axes[plot_idx // 3, plot_idx % 3]
        
        orig_traj = example_batch[plot_idx]
        recon_traj = example_recon[plot_idx]
        length = int(example_lengths[plot_idx])
        
        # Plot spatial trajectory (UTM_x, UTM_y)
        ax.plot(orig_traj[:length, 0], orig_traj[:length, 1], 
               'b-', linewidth=2, label='Original', alpha=0.7)
        ax.plot(recon_traj[:length, 0], recon_traj[:length, 1], 
               'r--', linewidth=2, label='Reconstructed', alpha=0.7)
        ax.scatter(orig_traj[0, 0], orig_traj[0, 1], 
                  c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(orig_traj[length-1, 0], orig_traj[length-1, 1], 
                  c='red', s=100, marker='x', label='End', zorder=5)
        
        mse = ((orig_traj[:length] - recon_traj[:length])**2).mean()
        
        ax.set_xlabel('UTM_x (normalized)')
        ax.set_ylabel('UTM_y (normalized)')
        ax.set_title(f'Sample {plot_idx}\nMSE={mse:.4f}, Length={length}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.suptitle('Reconstruction Quality Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/05_reconstruction_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 6. Potential Cluster Preview (K-means)
    # ============================================================
    from sklearn.cluster import KMeans
    
    print("Running K-means clustering preview...")
    
    # Try multiple k values
    k_values = [3, 5, 8, 10]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    for idx, k in enumerate(k_values):
        ax = axes[idx // 2, idx % 2]
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(all_latents)
        
        # Plot on PCA space
        scatter = ax.scatter(latents_pca_2d[:, 0], latents_pca_2d[:, 1], 
                           c=cluster_labels, cmap='tab10', alpha=0.6, s=15)
        
        # Plot cluster centers
        centers_pca = pca_2d.transform(kmeans.cluster_centers_)
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                  c='red', marker='X', s=200, edgecolor='black', linewidth=2,
                  label='Centroids', zorder=10)
        
        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'K-Means Clustering (k={k})\nSilhouette Score: {calculate_silhouette(all_latents, cluster_labels):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.suptitle('Potential Clustering Results (K-means)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_kmeans_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # ============================================================
    # 7. Summary Statistics Report
    # ============================================================
    print("\n" + "="*70)
    print("RAE LATENT SPACE ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(all_latents)}")
    print(f"  Latent dimensions: {latent_dim}")
    print(f"  Avg trajectory length: {all_lengths.mean():.1f} ± {all_lengths.std():.1f}")
    
    print(f"\n🎯 Latent Space Statistics:")
    print(f"  Mean: {all_latents.mean():.4f}")
    print(f"  Std:  {all_latents.std():.4f}")
    print(f"  Norm (mean): {latent_norms.mean():.4f}")
    print(f"  Norm (std):  {latent_norms.std():.4f}")
    
    print(f"\n📈 Reconstruction Quality:")
    print(f"  Mean MSE: {recon_errors.mean():.6f}")
    print(f"  Std MSE:  {recon_errors.std():.6f}")
    print(f"  Min MSE:  {recon_errors.min():.6f}")
    print(f"  Max MSE:  {recon_errors.max():.6f}")
    
    print(f"\n🔍 Clustering Indicators:")
    print(f"  Mean pairwise distance: {upper_tri_distances.mean():.4f}")
    print(f"  Std pairwise distance:  {upper_tri_distances.std():.4f}")
    print(f"  Variance per dim (mean): {var_per_dim.mean():.4f}")
    print(f"  Variance per dim (std):  {var_per_dim.std():.4f}")
    print(f"  PCA 95% variance: {np.searchsorted(cumsum, 0.95) + 1} components")
    
    # Active dimensions (variance > 0.01)
    active_dims = (var_per_dim > 0.01).sum()
    print(f"  Active dimensions: {active_dims}/{latent_dim} ({100*active_dims/latent_dim:.1f}%)")
    
    # Health assessment
    print(f"\n🏥 Health Assessment:")
    
    health_score = 0
    max_score = 5
    
    if all_latents.std() > 0.3:
        print("  ✅ Overall std > 0.3: Good spread")
        health_score += 1
    else:
        print("  ❌ Overall std < 0.3: Poor spread")
    
    if recon_errors.mean() < 0.01:
        print("  ✅ Low reconstruction error: Good learning")
        health_score += 1
    else:
        print("  ⚠️  High reconstruction error: May need more training")
    
    if upper_tri_distances.mean() > 1.0:
        print("  ✅ Good separation between samples")
        health_score += 1
    else:
        print("  ⚠️  Low separation: May have collapsed representations")
    
    if active_dims >= latent_dim * 0.7:
        print(f"  ✅ {100*active_dims/latent_dim:.0f}% dimensions active: Good capacity usage")
        health_score += 1
    else:
        print(f"  ⚠️  Only {100*active_dims/latent_dim:.0f}% dimensions active")
    
    if var_per_dim.mean() > 0.1:
        print("  ✅ Good variance: Suitable for clustering")
        health_score += 1
    else:
        print("  ❌ Low variance: May struggle with clustering")
    
    print(f"\n📊 Overall Health Score: {health_score}/{max_score}")
    
    if health_score >= 4:
        print("  🎉 EXCELLENT - Ready for clustering!")
    elif health_score >= 3:
        print("  ✅ GOOD - Should work for clustering")
    elif health_score >= 2:
        print("  ⚠️  MODERATE - Consider tuning hyperparameters")
    else:
        print("  ❌ POOR - Needs significant improvement")
    
    print("="*70 + "\n")
    
    # Save summary to file
    with open(f'{save_dir}/summary_report.txt', 'w') as f:
        f.write("RAE LATENT SPACE ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Samples: {len(all_latents)}\n")
        f.write(f"Latent dimensions: {latent_dim}\n")
        f.write(f"Latent std: {all_latents.std():.4f}\n")
        f.write(f"Mean MSE: {recon_errors.mean():.6f}\n")
        f.write(f"Mean pairwise distance: {upper_tri_distances.mean():.4f}\n")
        f.write(f"Active dimensions: {active_dims}/{latent_dim}\n")
        f.write(f"Health score: {health_score}/{max_score}\n")
    
    print(f"✅ All visualizations saved to '{save_dir}/'")
    
    return {
        'latents': all_latents,
        'recon_errors': recon_errors,
        'var_per_dim': var_per_dim,
        'health_score': health_score,
        'pca_2d': pca_2d,
        'latents_pca_2d': latents_pca_2d,
        'latents_tsne': latents_tsne
    }

def calculate_silhouette(X, labels):
    """Helper function to calculate silhouette score"""
    from sklearn.metrics import silhouette_score
    try:
        return silhouette_score(X, labels)
    except:
        return 0.0