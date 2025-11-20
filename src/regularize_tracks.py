import pandas as pd
import numpy as np
import pyarrow
import pyarrow.parquet
from scipy.interpolate import PchipInterpolator


def interpolate_continuous_pchip(series: pd.Series, new_index: pd.DatetimeIndex) -> pd.Series:
    '''
    PCHIP interpolation of a numeric time series onto a new DatetimeIndex.
    '''
    
    # Ensure date time index and sort
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    # Drop null values, return nothing if number of values less than 2
    s = s.dropna()
    if len(s) < 2:
        out = pd.Series(index=new_index, dtype=float, name=series.name)
        if len(s) == 1:
            out[:] = s.iloc[0]
        return out
    
    x = s.index.view("int64").astype(float)
    y = s.values.astype(float)

    x_new = pd.DatetimeIndex(new_index).view("int64").astype(float)

    pchip = PchipInterpolator(x, y, extrapolate=False)
    y_new = pchip(x_new)

    return pd.Series(y_new, index=new_index, name=series.name)


def interpolate_angle(series: pd.Series, new_index: pd.DatetimeIndex) -> pd.Series:
    '''
    Time-based interpolation of angular data (degrees),
    using unit-circle projection (sin/cos) to handle wrap-around.
    '''
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    # if only a few indexes
    if len(s) < 2:
        out = pd.Series(index=new_index, dtype=float, name=series.name)
        if len(s) == 1:
            out[:] = float(s.iloc[0]) % 360
        return out

    # Convert to radians
    radians = np.deg2rad(s.values)
    sin_vals = np.sin(radians)
    cos_vals = np.cos(radians)

    df = pd.DataFrame({'sin': sin_vals, 'cos': cos_vals}, index=s.index)

    # Combine original and desired timestamps
    full_index = df.index.union(new_index)
    df = df.reindex(full_index).sort_index()

    df['sin'] = df['sin'].interpolate(method='time')
    df['cos'] = df['cos'].interpolate(method='time')

    df = df.loc[new_index]

    # Convert back to angle
    angles = np.arctan2(df['sin'], df['cos'])
    angles = np.rad2deg(angles)

    # Normalize to [0, 360)
    angles = (angles + 360) % 360

    return pd.Series(angles, index=new_index, name=series.name)


def resample_segment(g):
    '''
    Regularize a single (MMSI, Segment) group to 1-minute frequency.
    Applies angle interpolation for COG/Heading and PCHIP for Lat/Lon/SOG.
    '''

    g = g.copy()

    g["Timestamp"] = pd.to_datetime(g["Timestamp"])
    g = g.sort_values("Timestamp").set_index("Timestamp")

    new_index = pd.date_range(start=g.index.min(), end=g.index.max(), freq='min')
    
    out = pd.DataFrame(index=new_index)

    # Interpolation on an angle
    for col in ["COG", "Heading"]:
        if col in g.columns:
            out[col] = interpolate_angle(g[col].astype(float), new_index)
    
    # Interpolate for continuous data
    for col in ["Latitude", "Longitude", "SOG"]:
        if col in g.columns:
            out[col] = interpolate_continuous_pchip(g[col].astype(float), new_index)

    return out

    # TODO:
        # Are there any missing values that we need to be aware of?


    # TIMESTAMP:    Fixed 1min intervals
    # MMSI:         Constant fill
    # Lat/Lon:      Cupic Spline Interpolation with Haversine constraint
    # SOG:          PCHIP / Linear interpolation
    # COG:          SLERP interpolation
    # Heading:      SLERP interpolation  
    # Segments:     Constant fill 

def regularize_segment(file_path="data/aisdk/interim/initial_clean", out_path="data/aisdk/interim/regularized"):
    '''
    Reads interim parquet data, stored in the 'read_path'
    Regularizes each segment on a 1 minute interval
    Stores the data as parquet files in the 'out' filepath
    '''
    df = pd.read_parquet(file_path)
    df_reg = df.groupby(["MMSI", "Segment"]).apply(resample_segment)
    df_reg = df_reg.reset_index(names=["MMSI", "Segment", "Timestamp"])

    table = pyarrow.Table.from_pandas(df_reg, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI",  # "Date",
                        "Segment",  # "Geocell"
                        ]
    )