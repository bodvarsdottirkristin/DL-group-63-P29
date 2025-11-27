"""Build fixed-length trajectory sequences from preprocessed parquet data."""

import pandas as pd
import numpy as np
import torch


def build_trajectories_from_parquet(
    parquet_path: str,
    seq_len: int = 64,
    step: int = 1,
    mmsi_whitelist=None,
) -> torch.Tensor:
    """Load preprocessed AIS parquet, segment into trajectories, create sliding-window
    sequences of shape (N, seq_len, 5).

    Args:
        parquet_path: path to parquet file or directory (readable by pd.read_parquet)
                      Expects partitioned by MMSI and/or Trajectory
        seq_len: desired sequence length (time steps per window)
        step: stride when sliding window (1 = all overlapping windows, >1 = sparser)
        mmsi_whitelist: optional list of MMSI strings/ints to restrict to (for debugging)

    Returns:
        trajectories: torch.FloatTensor of shape (N, seq_len, 5)
                     where the 5 features are ['UTM_x', 'UTM_y', 'SOG', 'v_east', 'v_north']

    Raises:
        ValueError: if no sequences are produced (e.g., all trajectories too short)
    """
    df = pd.read_parquet(parquet_path)
    
    print(f"Loaded parquet with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    if "Timestamp" not in df.columns:
        raise ValueError("Expected 'Timestamp' column in parquet file.")

    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Ensure MMSI and Trajectory columns exist (they may be partition columns)
    if "MMSI" not in df.columns:
        raise ValueError("Expected 'MMSI' column or partition (check parquet structure).")
    
    if "Trajectory" not in df.columns:
        # Try "Segment" as fallback for older data
        if "Segment" in df.columns:
            print("Using 'Segment' column as trajectory ID.")
            df["Trajectory"] = df["Segment"]
        else:
            raise ValueError("Expected 'Trajectory' or 'Segment' column in parquet file.")

    # Filter to a subset of MMSIs, if requested
    if mmsi_whitelist is not None:
        # Make sure types match (MMSI often stored as string in your pipeline)
        df["MMSI"] = df["MMSI"].astype(str)
        mmsi_whitelist = [str(m) for m in mmsi_whitelist]
        df = df[df["MMSI"].isin(mmsi_whitelist)]
        print(f"Filtered to {len(mmsi_whitelist)} MMSIs, remaining rows: {len(df)}")

    group_cols = ["MMSI", "Trajectory"]
    df = df.sort_values(group_cols + ["Timestamp"])

    print(f"Total rows after filtering & sorting: {len(df)}")
    print(f"Number of trajectory groups: {df.groupby(group_cols).ngroups}")

    feature_cols = ['UTM_x', 'UTM_y', 'SOG', 'v_east', 'v_north']
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Expected feature column '{col}' in parquet file.")

    sequences = []
    lengths = []

    for keys, g in df.groupby(group_cols):
        feats = g[feature_cols].to_numpy(dtype=np.float32)  # (T, 5)
        T = feats.shape[0]
        lengths.append(T)
        
        if T < seq_len:
            continue

        # sliding windows with stride `step`
        for start in range(0, T - seq_len + 1, step):
            window = feats[start : start + seq_len]  # (seq_len, 5)
            sequences.append(window)

    if len(sequences) == 0:
        print("DEBUG: No sequences created.")
        if lengths:
            print(f"  Num trajectories: {len(lengths)}")
            print(f"  Min length: {min(lengths)}")
            print(f"  Max length: {max(lengths)}")
            print(f"  Mean length: {np.mean(lengths):.1f}")
            print(f"  seq_len required: {seq_len}")
        else:
            print("  No trajectories at all after filtering (maybe MMSI filter too strict?).")
        raise ValueError("No sequences produced. Check seq_len / MMSI filter / data path.")

    trajectories = np.stack(sequences, axis=0)  # (N, seq_len, 5)
    print(f"Built {len(sequences)} trajectory sequences with shape: {trajectories.shape}")
    
    return torch.from_numpy(trajectories).float()

