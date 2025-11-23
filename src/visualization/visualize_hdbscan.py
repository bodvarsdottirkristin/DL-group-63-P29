import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_hdbscan_latent(mus, labels, out_path="plots/hdbscan_latent.png"):
    """
    mus:    np.ndarray (N, latent_dim)
    labels: np.ndarray (N,) from HDBSCAN (cluster ids, -1 = noise)
    out_path: where to save the PNG
    """

    mus = np.asarray(mus)
    labels = np.asarray(labels)

    # 1) Reduce to 2D with PCA (simple & fast)
    pca = PCA(n_components=2)
    mus_2d = pca.fit_transform(mus)  # (N, 2)

    # 2) Prepare colors: -1 (noise) gets a special color
    unique_labels = np.unique(labels)
    n_clusters = np.sum(unique_labels != -1)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Unique labels (incl. noise): {unique_labels}")
    print(f"Number of clusters (excluding noise): {n_clusters}")

    # Colormap: use a qualitative map; noise as light gray
    cmap = plt.get_cmap("tab20")
    label_to_color = {}

    cluster_labels = [lab for lab in unique_labels if lab != -1]
    for i, lab in enumerate(cluster_labels):
        label_to_color[lab] = cmap(i % 20)

    label_to_color[-1] = (0.8, 0.8, 0.8, 0.5)  # noise = semi-transparent gray

    # 3) Plot
    plt.figure(figsize=(8, 6))

    for lab in unique_labels:
        mask = labels == lab
        color = label_to_color[lab]
        label_name = f"cluster {lab}" if lab != -1 else "noise"
        plt.scatter(
            mus_2d[mask, 0],
            mus_2d[mask, 1],
            s=10,
            alpha=0.7,
            c=[color],
            label=label_name,
        )

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("HDBSCAN clusters in VRAE latent space")

    # Avoid too many legend entries if many clusters
    if len(unique_labels) <= 20:
        plt.legend(loc="best", fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved HDBSCAN latent plot to {out_path}")
