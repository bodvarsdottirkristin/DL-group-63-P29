from sklearn.preprocessing import StandardScaler
import hdbscan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clustering of latent representation of trajectories
def cluster_latent_space(mus):
    """
    mus: np.ndarray (N, latent_dim)
    Returns:
        labels: np.ndarray (N,) with cluster labels (or -1 for noise in HDBSCAN)
        clusterer: fitted clustering object
        scaler: fitted StandardScaler
    """

    # 1) Standardize latent vectors
    scaler = StandardScaler()
    mus_scaled = scaler.fit_transform(mus)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=30,   # TODO: Tune
        min_samples=None,      # TODO: Tune
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(mus_scaled)

    return labels, clusterer, scaler

