import pandas as pd

def regularize_trajectory(trajectory_df, interval_minutes=5):
    """
    Regularize a single trajectory to fixed time intervals using linear interpolation.
    
    Parameters:
    - trajectory_df: DataFrame for a single trajectory (one MMSI and one Trajectory ID)
    - interval_minutes: Desired time interval in minutes (default: 5 minutes)
    
    Returns:
    - DataFrame with regularized timestamps and interpolated values
    """
    if len(trajectory_df) < 2:
        return trajectory_df
    
    # Sort by timestamp
    trajectory_df = trajectory_df.sort_values('Timestamp').copy()
    
    # Create regular time grid
    start_time = trajectory_df['Timestamp'].min()
    end_time = trajectory_df['Timestamp'].max()
    
    # Generate regular timestamps
    regular_timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f'{interval_minutes}min'
    )
    
    # Create new dataframe with regular timestamps
    regular_df = pd.DataFrame({'Timestamp': regular_timestamps})
    
    # Set timestamp as index for interpolation
    trajectory_df = trajectory_df.set_index('Timestamp')
    
    # Combine original and regular timestamps, removing duplicates
    combined_df = pd.concat([
        trajectory_df,
        regular_df.set_index('Timestamp')
    ]).sort_index()
    
    # Remove duplicate index values (keep first occurrence)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    # Interpolate numeric columns
    numeric_cols = ['Latitude', 'Longitude', 'SOG', 'COG']
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].interpolate(method='linear')
    
    # Forward fill non-numeric columns (MMSI, Trajectory)
    for col in ['MMSI', 'Trajectory']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].ffill()
    
    # Keep only the regular timestamps and reset index
    result = combined_df.reindex(regular_timestamps).reset_index()
    
    # The index should now be a column named 'Timestamp' or 'index'
    if 'index' in result.columns and 'Timestamp' not in result.columns:
        result = result.rename(columns={'index': 'Timestamp'})
    
    return result

def regularize_all_trajectories(df, interval_minutes=5):
    """
    Regularize all trajectories in the dataset.
    
    Parameters:
    - df: DataFrame with MMSI and Trajectory columns
    - interval_minutes: Desired time interval in minutes
    
    Returns:
    - DataFrame with all trajectories regularized
    """
    regularized_trajectories = []
    
    total_trajectories = df['Trajectory'].nunique()
    print(f"Regularizing {total_trajectories} trajectories...")
    
    for i, (traj_id, traj_group) in enumerate(df.groupby('Trajectory')):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_trajectories} trajectories")
        
        regularized = regularize_trajectory(traj_group, interval_minutes)
        regularized_trajectories.append(regularized)
    
    print(f"  Completed all {total_trajectories} trajectories")
    return pd.concat(regularized_trajectories, ignore_index=True)