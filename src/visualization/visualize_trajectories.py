import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch import nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

import plotly.express as px
from pyproj import Transformer

import os
import joblib

import importlib
import json
import itertools

def visualize_trajectories_efficient(
    df,
    color_by='MMSI',
    title=None,
    zoom=5,
    height=800,
    max_points_total=800_000,
    max_points_per_group=8_000,
    transform_chunk_size=300_000,):

    required = {'UTM_x', 'UTM_y', 'Timestamp', 'MMSI'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    vis_cols = ['MMSI', 'Timestamp', 'UTM_x', 'UTM_y']
    if 'Trajectory' in df.columns:
        vis_cols.insert(0, 'Trajectory')
    if color_by not in vis_cols:
        vis_cols.append(color_by)
    if 'SOG' in df.columns and 'SOG' not in vis_cols:
        vis_cols.append('SOG')

    # Sort to maintain path continuity
    vis_df = df[vis_cols].sort_values(
        ['MMSI'] + (['Trajectory'] if 'Trajectory' in df.columns else []) + ['Timestamp']
    )

    # Chunked UTM -> WGS84 transform to lower peak memory
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    n = len(vis_df)
    lon_out = np.empty(n, dtype='float64')
    lat_out = np.empty(n, dtype='float64')

    utm_x = vis_df['UTM_x'].to_numpy()
    utm_y = vis_df['UTM_y'].to_numpy()

    idx_start = 0
    while idx_start < n:
        idx_end = min(idx_start + transform_chunk_size, n)
        lon_chunk, lat_chunk = transformer.transform(utm_x[idx_start:idx_end], utm_y[idx_start:idx_end])
        lon_out[idx_start:idx_end] = lon_chunk
        lat_out[idx_start:idx_end] = lat_chunk
        idx_start = idx_end

    vis_df = vis_df.assign(Longitude=lon_out, Latitude=lat_out)

    # Per-group downsampling: keep at most max_points_per_group points per group
    group_cols = ['MMSI'] + (['Trajectory'] if 'Trajectory' in vis_df.columns else [])
    def _downsample_group(g):
        if len(g) <= max_points_per_group:
            return g
        stride = max(1, len(g) // max_points_per_group)
        return g.iloc[::stride]

    vis_df = vis_df.groupby(group_cols, as_index=False, sort=False).apply(_downsample_group).reset_index(drop=True)

    # Global cap as a final guardrail
    total_points = len(vis_df)
    if total_points > max_points_total:
        stride = max(1, total_points // max_points_total)
        vis_df = vis_df.iloc[::stride]

    # Title
    if title is None:
        try:
            date_min = pd.to_datetime(vis_df['Timestamp']).min().date()
            date_max = pd.to_datetime(vis_df['Timestamp']).max().date()
            title = f"Ship Trajectories - {date_min} to {date_max}"
        except Exception:
            title = "Ship Trajectories"

    fig = px.line_mapbox(
        vis_df,
        lat="Latitude",
        lon="Longitude",
        color=color_by,
        hover_data=[c for c in ["MMSI", "Timestamp", "SOG"] if c in vis_df.columns],
        zoom=zoom,
        title=title,
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        showlegend=False,
        height=height,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    print(
        f"✓ Visualization complete - colored by '{color_by}'. "
        f"Plotted {len(vis_df)} points across {vis_df.groupby(group_cols).ngroups} trajectories."
    )

    return fig

def visualize_trajectories_to_compare(df, color_by='MMSI', title=None, zoom=5, height=800):
    
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df['UTM_x'].values, df['UTM_y'].values)
    
    if 'Trajectory' in df.columns:
        vis_cols = ['Trajectory', 'Timestamp', 'UTM_x', 'UTM_y']
    else:
        vis_cols = ['Timestamp', 'UTM_x', 'UTM_y']

    if color_by not in vis_cols:
        vis_cols.append(color_by)

    if 'SOG' in df.columns and 'SOG' not in vis_cols:
        vis_cols.append('SOG')
    
    vis_df = df[vis_cols].copy()
    vis_df['Longitude'] = lon
    vis_df['Latitude'] = lat
    
    # Generate title if not provided
    if title is None:
        date_min = vis_df['Timestamp'].min().date()
        date_max = vis_df['Timestamp'].max().date()
        title = f"Ship Trajectories - {date_min} to {date_max}"
    
    # Create visualization with trajectories using lat/lon on a map
    fig = px.line_map(
        vis_df.sort_values('Timestamp'),
        lat="Latitude",
        lon="Longitude",
        color=color_by,
        # hover_data=["Trajectory", "Timestamp", "SOG"] if "SOG" in vis_df.columns else ["MMSI", "Timestamp"],
        zoom=zoom,
        title=title
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        showlegend=False,  # Hide legend since there can be many trajectories
        height=height
    )
    
    print(f"✓ Visualization complete - colored by '{color_by}'")
    
    return fig