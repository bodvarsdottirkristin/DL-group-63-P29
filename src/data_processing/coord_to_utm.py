import utm

def to_utm(df, force_zone_number=32):
    """
    Convert latitude and longitude to UTM coordinates.
    
    Parameters:
    - df: DataFrame with 'Latitude' and 'Longitude' columns
    - force_zone_number: UTM zone number to use for all coordinates (default: 32 for Denmark)
    
    Returns:
    - DataFrame with added 'UTM_x', 'UTM_y', 'UTM_zone', 'UTM_letter' columns
    """
    df = df.copy()
    
    # Convert each point to UTM using a fixed zone to avoid jumps at zone boundaries
    utm_coords = df.apply(
        lambda row: utm.from_latlon(
            row['Latitude'], 
            row['Longitude'],
            force_zone_number=force_zone_number
        ),
        axis=1
    )
    
    # Extract UTM components
    df['UTM_x'] = utm_coords.apply(lambda x: x[0])  # Easting (meters)
    df['UTM_y'] = utm_coords.apply(lambda x: x[1])  # Northing (meters)
    df['UTM_zone'] = utm_coords.apply(lambda x: x[2])  # Zone number
    df['UTM_letter'] = utm_coords.apply(lambda x: x[3])  # Zone letter
    
    return df