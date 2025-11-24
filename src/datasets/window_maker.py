import pandas as pd
import pyarrow
import pyarrow.parquet
import os
#from src.models.HDBSCAN import assign_cluster_ids_to_segments

# TODO: IMPLEMENT CLUSTERS!!!
# For now, we will assign all segments to cluster 0
# This should be in the HDBSCAN file

def assign_cluster_ids_to_segments(df_segment):
    """
    Given a DataFrame for a trajectory segment, assign a cluster ID
    based on HDBSCAN clustering of the trajectory's latent representation.

    Returns:
        cluster_id: int, cluster label assigned by HDBSCAN
    """
    return random.randint(0, 10)  # Placeholder: assign random cluster between 0 and 9

# TODO: 
# Create a dataloader for this
def make_past_future_windows(
        past_min=30, 
        future_min=30, 
        input_path="data/aisdk/processed/aisdk_2025", 
        output_path="data/aisdk/processed/windows_30_30"):
    
    '''
        Splits trajectory data into past and future windows, 
        past_min and future_min define the length of each window in minutes.
    '''

    feature_cols = ['UTM_x', 'UTM_y', 'SOG', 'v_east', 'v_north']

    df = pd.read_parquet(input_path)
    df = df.sort_values(["MMSI", "Trajectory", "Timestamp"])

    group_cols = ['MMSI', 'Trajectory']
    grouped = df.groupby(group_cols)

    rows = []
    window_len_total = past_min + future_min

    # Iterate each trajectory segment
    for (mmsi, seg_id), g in grouped:
        # g is a DataFrame for this trajectory segment
        
        T = len(g)

        # If trajectorory is too short, skip
        if T < window_len_total:
            continue 

        feats = g[feature_cols].to_numpy(dtype=float)

        # Assign cluster ID to this segment
        cluster_id = assign_cluster_ids_to_segments(g)

        # Determine how many windows we can extract
        num_windows = T - window_len_total + 1

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
                "Segment": seg_id
            })

    # Create dataframe and save as parquet partitioned by cluster_id
    df_windows = pd.DataFrame(rows)
    df_windows.to_parquet(
        output_path,
        index=False,
        partition_cols=["cluster_id"]
    )