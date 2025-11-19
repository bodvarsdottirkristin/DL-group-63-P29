"""
Trajectory Segmentation for Ship AIS Data

Segments continuous ship tracking data into distinct voyages/trajectories.
Trajectories are split when ships stop moving for extended periods.

No movement = (SOG < 1 knot) OR (position variance < 50m) for > 30 min
"""

import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth in meters.
    Uses the Haversine formula for great-circle distance.
    """
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


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


def segment_trajectories(df, 
                         sog_threshold=0.514,  # 1 knot in m/s
                         position_threshold=50,  # 50 meters
                         time_threshold=30,  # 30 minutes
                         min_segment_distance=1000,  # 1 km
                         min_points=10):
    """
    Segment ship AIS data into trajectories, splitting when ships stop moving.
    No movement = (SOG < 1 knot) OR (position variance < 50m) for > 30 min
    """
    
    df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True).copy()
    df['Trajectory'] = -1
    current_traj_id = 0
    
    for mmsi in df['MMSI'].unique():
        ship_indices = df[df['MMSI'] == mmsi].index.tolist()
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
                        # Continue adding points until stationary period ends
                        while j < len(ship_indices):
                            test_idx = ship_indices[j]
                            test_buffer = stationary_buffer + [test_idx]
                            
                            lats = df.loc[test_buffer, 'Latitude'].values
                            lons = df.loc[test_buffer, 'Longitude'].values
                            sogs = df.loc[test_buffer, 'SOG'].values
                            
                            low_speed = (sogs < sog_threshold).all()
                            pos_variance = calculate_position_variance(lats, lons)
                            
                            if low_speed or pos_variance < position_threshold:
                                # Still stationary
                                stationary_buffer.append(test_idx)
                                j += 1
                            else:
                                # Stationary period ended
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