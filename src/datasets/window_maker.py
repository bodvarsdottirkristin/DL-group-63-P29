import pandas as pd
import pyarrow
import pyarrow.parquet
import os
import random
import numpy as np


def assign_cluster_ids_to_segments(df_segment):
    """Placeholder random cluster assignment."""
    return random.randint(0, 9)


def make_past_future_windows(
        past_min=30, 
        future_min=30, 
        input_path="data/aisdk/processed/aisdk_2025", 
        output_path="data/aisdk/processed/windows_30_30"):
    """
    Splits trajectory data into past and future windows,
    and saves per-segment parquet files partitioned by cluster_id.
    """

    print("Loading input dataset...")
    df = pd.read_parquet(input_path)
    print(f"   → Loaded {len(df):,} rows.")

    df = df.sort_values(["MMSI", "Trajectory", "Timestamp"])
    print(f"   → Sorted by MMSI, Trajectory, Timestamp.\n")

    feature_cols = ['UTM_x', 'UTM_y', 'SOG', 'v_east', 'v_north']

    group_cols = ['MMSI', 'Trajectory']
    grouped = df.groupby(group_cols)

    window_len_total = past_min + future_min

    os.makedirs(output_path, exist_ok=True)

    total_segments = len(grouped)
    print(f"Total trajectory segments: {total_segments}\n")

    processed_segments = 0
    total_windows = 0

    # Iterate each trajectory segment
    for (mmsi, seg_id), g in grouped:

        processed_segments += 1
        T = len(g)

        # Skip short segments
        if T < window_len_total:
            continue

        feats = g[feature_cols].to_numpy(dtype=float)

        # Assign cluster
        cluster_id = assign_cluster_ids_to_segments(g)

        # Number of windows
        num_windows = T - window_len_total + 1
        total_windows += num_windows

        # Minor printout every 50 segments
        if processed_segments % 50 == 0:
            print(f"   → Processed {processed_segments}/{total_segments} segments "
                  f"(+{num_windows} windows, cluster {cluster_id})")

        rows = []
        for past_idx_start in range(num_windows):
            past_idx_end = past_idx_start + past_min
            future_idx_start = past_idx_end
            future_idx_end = future_idx_start + future_min

            past_window = feats[past_idx_start:past_idx_end]
            future_window = feats[future_idx_start:future_idx_end]

            rows.append({
                "past_window": past_window.tolist(),
                "future_window": future_window.tolist(),
                "cluster_id": cluster_id,
                "MMSI": mmsi,
                "Trajectory": seg_id
            })

        # Save this segment
        seg_folder = os.path.join(output_path, f"cluster_id={cluster_id}")
        os.makedirs(seg_folder, exist_ok=True)

        file_name = f"segment_{mmsi}_{seg_id}.parquet"
        file_path = os.path.join(seg_folder, file_name)

        df_seg = pd.DataFrame(rows)
        df_seg.to_parquet(file_path, index=False)

    print("\nDONE!")
    print(f"   → Processed segments: {processed_segments}")
    print(f"   → Total windows generated: {total_windows:,}")
    print(f"   → Output stored under: {output_path}\n")


def load_parquet_files(input_path='data/aisdk/processed/windows_30_30', cluster_id=None):

    if cluster_id is not None:
        # Load only one cluster
        cluster_dir = os.path.join(input_path, f"cluster_id={cluster_id}")
        df = pd.read_parquet(cluster_dir)

    else:
        df_list = []
        for cluster_dir in os.listdir(input_path):

            full_path = os.path.join(input_path, cluster_dir)

            # Only load directories like: cluster_id=0, cluster_id=2, ...
            if not (os.path.isdir(full_path) and cluster_dir.startswith("cluster_id=")):
                continue

            print(f"→ loading {full_path}")
            df_cid = pd.read_parquet(full_path)
            df_list.append(df_cid)

        df = pd.concat(df_list, ignore_index=True)

    df['past_window'] = df["past_window"].apply(lambda w: np.vstack(w))
    df["future_window"] = df["future_window"].apply(lambda w: np.vstack(w))

    past = np.stack(df['past_window'])
    future = np.stack(df['future_window'])


    return past, future, df["cluster_id"]



    



    


    #return past_window, future_window, cluster_id
    