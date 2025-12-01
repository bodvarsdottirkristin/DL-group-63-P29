"""
Trajectory Segmentation for Ship AIS Data (Optimized)

Segments continuous ship tracking data into distinct voyages/trajectories.
Trajectories are split when ships stop moving for extended periods.

No movement = (SOG < 1 knot) OR (position variance < 50m) for > 30 min

Optimizations:
- Early SOG checking to avoid expensive distance calculations
- Pre-computed time differences
- Fixed trajectory assignment bug
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth in meters.
    Uses the Haversine formula for great-circle distance.
    """
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_position_variance(latitudes, longitudes):
    """
    Calculate maximum distance from center point (position variance).
    Returns the radius of the smallest circle containing all points.
    Vectorized for speed.
    """
    if len(latitudes) < 2:
        return 0.0
    
    center_lat = latitudes.mean()
    center_lon = longitudes.mean()
    
    # Vectorized haversine
    R = 6371000
    lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
    lat2, lon2 = np.radians(latitudes), np.radians(longitudes)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c
    
    return distances.max()


def segment_trajectories(df, 
                         sog_threshold=0.514,  # 1 knot in m/s
                         position_threshold=50,  # 50 meters
                         time_threshold=30,  # 30 minutes
                         time_gap_threshold=30,  # 30 minutes - split on large gaps (transponder issues)
                         min_segment_distance=1000,  # 1 km
                         min_points=10,
                         max_speed_mps=30.87):  # ~60 knots in m/s (fast military vessels)
    """
    Segment ship AIS data into trajectories, splitting when:
    1. Ships stop moving: (SOG < 1 knot) OR (position variance < 50m) for > 30 min
    2. Large time gaps (>30 min) between consecutive points (transponder issues)
    3. Unrealistic speed jumps between consecutive points (implied speed > max_speed)
    
    Optimized with vectorized operations and efficient detection.
    """
    
    df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True).copy()
    trajectories = np.full(len(df), -1, dtype=np.int32)
    current_traj_id = 0

    count = 0
    
    for mmsi in df['MMSI'].unique():
        mask = df['MMSI'] == mmsi
        ship_indices = np.where(mask)[0]

        count += 1
        print(f"Processing ship {count}")
        
        # Extract arrays once for this ship
        ship_lats = df.loc[mask, 'Latitude'].values
        ship_lons = df.loc[mask, 'Longitude'].values
        ship_sogs = df.loc[mask, 'SOG'].values
        ship_times = df.loc[mask, 'Timestamp'].values
        
        # Pre-compute time differences in minutes (vectorized)
        ship_times_minutes = (ship_times - ship_times[0]) / np.timedelta64(1, 'm')
        time_gaps = np.diff(ship_times_minutes)  # gaps between consecutive points
        
        # Pre-identify time gap splits (vectorized) - where gaps exceed threshold
        gap_splits = np.where(time_gaps >= time_gap_threshold)[0] + 1  # +1 because diff reduces size by 1
        
        # Pre-identify unrealistic speed jumps (vectorized) - efficient check
        speed_jump_splits = []
        if len(ship_lats) > 1:
            # Calculate distances between consecutive points (vectorized)
            distances = haversine_distance(
                ship_lats[:-1], ship_lons[:-1],
                ship_lats[1:], ship_lons[1:]
            )
            # Calculate time differences in seconds
            time_diffs = time_gaps * 60  # convert minutes to seconds
            # Avoid division by zero
            time_diffs = np.maximum(time_diffs, 0.1)
            # Calculate implied speeds in m/s
            implied_speeds = distances / time_diffs
            # Find where speed exceeds maximum realistic speed
            speed_jump_splits = np.where(implied_speeds > max_speed_mps)[0] + 1
        
        # Pre-identify low-speed points for faster checks
        low_speed_mask = ship_sogs < sog_threshold
        
        trajectories[ship_indices[0]] = current_traj_id
        
        i = 1
        n = len(ship_indices)
        gap_split_set = set(gap_splits)  # Convert to set for O(1) lookup
        speed_jump_set = set(speed_jump_splits)  # Convert to set for O(1) lookup
        
        while i < n:
            # Check for large time gap first (cheap O(1) operation)
            if i in gap_split_set:
                # Large time gap detected - start new trajectory
                current_traj_id += 1
                trajectories[ship_indices[i]] = current_traj_id
                i += 1
                continue
            
            # Check for unrealistic speed jump (cheap O(1) operation)
            if i in speed_jump_set:
                # Unrealistic jump detected - start new trajectory
                current_traj_id += 1
                trajectories[ship_indices[i]] = current_traj_id
                i += 1
                continue
            
            # Assign current point to trajectory by default
            if trajectories[ship_indices[i]] == -1:
                trajectories[ship_indices[i]] = current_traj_id
            
            # Use binary search to find furthest point within time threshold
            max_time = ship_times_minutes[i] + time_threshold
            j_max = np.searchsorted(ship_times_minutes, max_time, side='right')
            j_max = min(j_max, n)
            
            # Also limit j_max to next gap split or speed jump to avoid crossing trajectory boundaries
            next_gap_splits = gap_splits[gap_splits > i]
            if len(next_gap_splits) > 0:
                j_max = min(j_max, next_gap_splits[0])
            next_speed_jumps = speed_jump_splits[speed_jump_splits > i]
            if len(next_speed_jumps) > 0:
                j_max = min(j_max, next_speed_jumps[0])
            
            # Quick check: can we even reach the time threshold?
            if j_max <= i + 1:
                i += 1
                continue
            
            # Start checking from first point that satisfies time threshold
            j_start = i + 1
            while j_start < j_max and (ship_times_minutes[j_start] - ship_times_minutes[i]) < time_threshold:
                j_start += 1
            
            if j_start >= j_max:
                i += 1
                continue
            
            # Now check if period [i:j] is stationary for j >= j_start
            j = j_start
            stationary_found = False
            
            # Check if all SOG values in range are low (vectorized)
            if np.all(low_speed_mask[i:j+1]):
                # Extend as far as possible while maintaining low speed
                while j < j_max - 1 and low_speed_mask[j+1]:
                    j += 1
                stationary_end = j + 1
                stationary_found = True
            else:
                # SOG check failed, try position variance
                lats = ship_lats[i:j+1]
                lons = ship_lons[i:j+1]
                pos_variance = calculate_position_variance(lats, lons)
                
                if pos_variance < position_threshold:
                    # Extend the stationary period
                    while j < j_max - 1:
                        lats = ship_lats[i:j+2]
                        lons = ship_lons[i:j+2]
                        pos_variance = calculate_position_variance(lats, lons)
                        
                        if pos_variance < position_threshold:
                            j += 1
                        else:
                            break
                    stationary_end = j + 1
                    stationary_found = True
            
            if stationary_found:
                # Mark stationary points for exclusion
                trajectories[ship_indices[i:stationary_end]] = -2
                current_traj_id += 1
                i = stationary_end
            else:
                i += 1
        
        current_traj_id += 1
    
    # Assign trajectory column and remove stationary points
    df['Trajectory'] = trajectories
    df = df[df['Trajectory'] != -2].reset_index(drop=True)
    
    # Filter by minimum requirements - vectorized approach
    traj_ids = df['Trajectory'].values
    unique_trajs = df['Trajectory'].unique()
    
    # Pre-compute trajectory sizes
    traj_counts = df.groupby('Trajectory').size()
    size_valid = traj_counts >= min_points
    
    # Vectorized distance calculation
    valid_trajs = []
    first_indices = df.groupby('Trajectory').head(1).index
    last_indices = df.groupby('Trajectory').tail(1).index
    
    first_coords = df.loc[first_indices, ['Latitude', 'Longitude']].values
    last_coords = df.loc[last_indices, ['Latitude', 'Longitude']].values
    
    distances = haversine_distance(
        first_coords[:, 0], first_coords[:, 1],
        last_coords[:, 0], last_coords[:, 1]
    )
    
    for idx, traj_id in enumerate(unique_trajs):
        if size_valid[traj_id] and distances[idx] >= min_segment_distance:
            valid_trajs.append(traj_id)
    
    df_result = df[df['Trajectory'].isin(valid_trajs)].copy()
    
    if len(df_result) > 0:
        mapping = {old: new for new, old in enumerate(sorted(valid_trajs))}
        df_result['Trajectory'] = df_result['Trajectory'].map(mapping)
    
    return df_result