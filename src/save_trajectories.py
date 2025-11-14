import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from pathlib import Path


def save_trajectories_geojson():
    root = Path("data/aisdk/processed")
    df = pd.read_parquet(root)
    df = df.sort_values(["MMSI", "Segment", "Timestamp"])

    out_dir = Path("data/aisdk/trajectories_geojson")
    out_dir.mkdir(parents=True, exist_ok=True)

    for (mmsi, segment), g in df.groupby(["MMSI", "Segment"]):
        # take all points in order and build a LineString
        coords = list(zip(g["Longitude"], g["Latitude"]))
        if len(coords) < 2:
            continue  # can't make a line from a single point

        line = LineString(coords)
        gdf = gpd.GeoDataFrame(
            {"MMSI": [mmsi], "Segment": [segment]},
            geometry=[line],
            crs="EPSG:4326",  # lat/lon WGS84
        )

        fname = out_dir / f"MMSI_{mmsi}_segment_{segment}.geojson"
        gdf.to_file(fname, driver="GeoJSON")

    print("Saved GeoJSON trajectories to", out_dir)

    