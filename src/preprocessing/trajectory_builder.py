import numpy as np
import pandas as pd
import torch

R_EARTH = 6371000.0  # meters

# ----------------------------------------------------------------------
#  Segmentation code (your logic)
# ----------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth in meters.
    Uses the Haversine formula for great-circle distance.
    """
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R_EARTH * c


def calculate_position_variance(latitudes, longitudes):
    """
    Calculate maximum distance from center point (position variance).
    Returns the radius of the smallest circle containing all points.
    """
    if len(latitudes) < 2:
        return 0.0
    
    center_lat = latitudes.mean()
    center_lon = longitudes.mean()
    
    max_distance = 0.0
    for lat, lon in zip(latitudes, longitudes):
        dist = haversine_distance(center_lat, center_lon, lat, lon)
        max_distance = max(max_distance, dist)
    
    return max_distance


def segment_trajectories(
    df,
    sog_threshold=0.514,       # 1 knot in m/s
    position_threshold=50,     # 50 meters
    time_threshold=30,         # 30 minutes
    min_segment_distance=1000, # 1 km
    min_points=10,
):
    """
    Segment ship AIS data into trajectories, splitting when ships stop moving.
    No movement = (SOG < 1 knot) OR (position variance < 50m) for > 30 min
    """

    df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True).copy()
    df['Trajectory'] = -1
    current_traj_id = 0
    
    for mmsi in df['MMSI'].unique():
        ship_indices = df[df['MMSI'] == mmsi].index.tolist()
        if not ship_indices:
            continue

        df.loc[ship_indices[0], 'Trajectory'] = current_traj_id
        
        i = 1
        while i < len(ship_indices):
            curr_idx = ship_indices[i]
            
            # Build buffer to check for stationary period
            stationary_buffer = [curr_idx]
            j = i + 1
            
            while j < len(ship_indices):
                stationary_buffer.append(ship_indices[j])
                
                # Check duration
                start_time = df.loc[stationary_buffer[0], 'Timestamp']
                end_time = df.loc[stationary_buffer[-1], 'Timestamp']
                duration = (end_time - start_time).total_seconds() / 60
                
                if duration >= time_threshold:
                    # Check: (SOG < threshold) OR (position variance < threshold)
                    lats = df.loc[stationary_buffer, 'Latitude'].values
                    lons = df.loc[stationary_buffer, 'Longitude'].values
                    sogs = df.loc[stationary_buffer, 'SOG'].values
                    
                    low_speed = (sogs < sog_threshold).all()
                    pos_variance = calculate_position_variance(lats, lons)
                    
                    if low_speed or pos_variance < position_threshold:
                        # Stationary period confirmed - skip through it
                        while j < len(ship_indices):
                            test_idx = ship_indices[j]
                            test_buffer = stationary_buffer + [test_idx]
                            
                            lats = df.loc[test_buffer, 'Latitude'].values
                            lons = df.loc[test_buffer, 'Longitude'].values
                            sogs = df.loc[test_buffer, 'SOG'].values
                            
                            low_speed = (sogs < sog_threshold).all()
                            pos_variance = calculate_position_variance(lats, lons)
                            
                            if low_speed or pos_variance < position_threshold:
                                stationary_buffer.append(test_idx)
                                j += 1
                            else:
                                break
                        
                        # Mark all stationary points for exclusion
                        df.loc[stationary_buffer, 'Trajectory'] = -2
                        current_traj_id += 1
                        i = j
                        break
                    else:
                        # Not stationary, keep checking
                        j += 1
                else:
                    # Duration not long enough yet
                    j += 1
            else:
                # No stationary period found, assign to current trajectory
                if df.loc[curr_idx, 'Trajectory'] == -1:
                    df.loc[curr_idx, 'Trajectory'] = current_traj_id
                i += 1
        
        current_traj_id += 1
    
    # Remove stationary points (marked as -2)
    df = df[df['Trajectory'] != -2].reset_index(drop=True)
    
    # Filter by minimum requirements
    valid_trajs = []
    for traj_id in df['Trajectory'].unique():
        traj_data = df[df['Trajectory'] == traj_id]
        if len(traj_data) < min_points:
            continue
        
        start = traj_data.iloc[0][['Latitude', 'Longitude']].values
        end = traj_data.iloc[-1][['Latitude', 'Longitude']].values
        distance = haversine_distance(start[0], start[1], end[0], end[1])
        
        if distance >= min_segment_distance:
            valid_trajs.append(traj_id)
    
    df_result = df[df['Trajectory'].isin(valid_trajs)].copy()
    
    if len(df_result) > 0:
        mapping = {old: new for new, old in enumerate(sorted(valid_trajs))}
        df_result['Trajectory'] = df_result['Trajectory'].map(mapping)
    
    return df_result


# ----------------------------------------------------------------------
#  Features → trajectories tensor
# ----------------------------------------------------------------------

def trajectory_to_features(df_traj):
    """
    ONE (MMSI, Trajectory) -> (T, 5) features:
    [dx, dy, SOG, sin(COG), cos(COG)]

    NOTE: SOG is left in original units here; AISDataSet.get_dataloaders
    will normalize all features based on the train split.
    """
    df_traj = df_traj.sort_values("Timestamp")

    lat = np.deg2rad(df_traj["Latitude"].to_numpy())
    lon = np.deg2rad(df_traj["Longitude"].to_numpy())
    lat0, lon0 = lat[0], lon[0]

    x = R_EARTH * (lon - lon0) * np.cos(lat0)
    y = R_EARTH * (lat - lat0)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    sog = df_traj["SOG"].to_numpy().astype(float)

    cog_rad = np.deg2rad(df_traj["COG"].to_numpy())
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)

    feats = np.stack([dx, dy, sog, cog_sin, cog_cos], axis=-1)  # (T, 5)
    return feats


def build_trajectories_from_parquet(
    parquet_path: str,
    seq_len: int = 64,
    step: int = 1,
    mmsi_whitelist = None,
) -> torch.Tensor:
    """
    Load AIS parquet, segment into trajectories, create sliding-window
    sequences of shape (N, seq_len, 5).

    Returns:
        trajectories: torch.FloatTensor (N, seq_len, 5)
    """
    df = pd.read_parquet(parquet_path)

    if "Timestamp" not in df.columns:
        raise ValueError("Expected 'Timestamp' column in parquet file.")

    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])


     # 🔹 Filter to a subset of MMSIs, if requested
    if mmsi_whitelist is not None:
        # Make sure types match (MMSI often stored as string in your pipeline)
        df["MMSI"] = df["MMSI"].astype(str)
        mmsi_whitelist = [str(m) for m in mmsi_whitelist]
        df = df[df["MMSI"].isin(mmsi_whitelist)]
        print(f"Filtered to {len(mmsi_whitelist)} MMSIs, remaining rows: {len(df)}")

    # Decide which column defines trajectories
    if "Segment" in df.columns:
     
        print("Using existing 'Segment' column as trajectory id.")

    group_cols = ["MMSI", "Segment"]
    df = df.sort_values(group_cols + ["Timestamp"])

    print("Total rows after filtering & sorting:", len(df))
    print("Number of trajectory groups:", df.groupby(group_cols).ngroups)

    sequences = []
    lengths = []

    for keys, g in df.groupby(group_cols):
        feats = trajectory_to_features(g)  # (T, 5)
        T = feats.shape[0]
        lengths.append(T)
        if T < seq_len:
            continue

        # sliding windows
        for start in range(0, T - seq_len + 1, step):
            window = feats[start : start + seq_len]  # (seq_len, 5)
            sequences.append(window)

    if len(sequences) == 0:
        print("DEBUG: No sequences created.")
        if lengths:
            print("  Num trajectories:", len(lengths))
            print("  Min length:", min(lengths))
            print("  Max length:", max(lengths))
            print("  seq_len:", seq_len)
        else:
            print("  No trajectories at all after filtering (maybe MMSI filter too strict?).")
        raise ValueError("No sequences produced. Check seq_len / MMSI filter / data path.")

    trajectories = np.stack(sequences, axis=0)  # (N, seq_len, 5)
    print("Built trajectories tensor with shape:", trajectories.shape)
    return torch.from_numpy(trajectories).float()
