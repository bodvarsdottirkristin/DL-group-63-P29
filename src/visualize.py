import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def save_all_trajectory_plots_simple(
    data_path: str = "data/aisdk/processed",
    out_dir: str = "plots/trajectories",
):
    """
    Load the processed AIS parquet dataset, and for each (MMSI, Segment)
    save a PNG image of the trajectory (Lon/Lat) to out_dir.
    """
    # Load dataset
    df = pd.read_parquet(data_path)

    # Ensure sorted so trajectories draw correctly
    df = df.sort_values(["MMSI", "Segment", "Timestamp"])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pairs = df[["MMSI", "Segment"]].drop_duplicates()
    print(f"Found {len(pairs)} (MMSI, Segment) pairs")

    for _, row in pairs.iterrows():
        mmsi = row["MMSI"]
        segment = row["Segment"]

        seg_df = df[(df["MMSI"] == mmsi) & (df["Segment"] == segment)].copy()
        if len(seg_df) < 2:
            # too short to be interesting
            continue

        seg_df = seg_df.sort_values("Timestamp")

        # --- compute tight bounding box around this trajectory ---
        lon_min = seg_df["Longitude"].min()
        lon_max = seg_df["Longitude"].max()
        lat_min = seg_df["Latitude"].min()
        lat_max = seg_df["Latitude"].max()

        lon_pad = (lon_max - lon_min) * 0.2 if lon_max > lon_min else 0.1
        lat_pad = (lat_max - lat_min) * 0.2 if lat_max > lat_min else 0.1

        west = lon_min - lon_pad
        east = lon_max + lon_pad
        south = lat_min - lat_pad
        north = lat_max + lat_pad

        # --- create figure ---
        fig, ax = plt.subplots(figsize=(8, 8))

        # optional: faint bbox as a "map" background
        ax.fill(
            [west, east, east, west],
            [south, south, north, north],
            alpha=0.05,
        )

        # plot trajectory
        ax.plot(
            seg_df["Longitude"],
            seg_df["Latitude"],
            marker="o",
            markersize=3,
            linewidth=1.5,
        )

        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"MMSI={mmsi}, Segment={segment}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        # save instead of show
        fname = out_path / f"MMSI_{mmsi}_segment_{segment}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {fname}")
