import pandas as pd
import pyarrow
import pyarrow.parquet
from sklearn.cluster import KMeans
from src.visualize import save_all_trajectory_plots_simple
from src.inspect import inspect_one_segment

RAW_FILES = [f"data/aisdk/raw/aisdk-2025-08-01.csv"] #1 day for running on CPU

def fn(file_path, out_path):
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
        "Ship type": "object",
    }
    usecols = list(dtypes.keys())
    df = pd.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]

    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])

    # We only want Cargo ships
    df = df[df["Ship type"].isin(["Cargo"])]
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")

    def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])

    # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())  # Max allowed timegap

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)

    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    # Clustering
    # --------
    kmeans = KMeans(n_clusters=48, random_state=0)
    kmeans.fit(df[["Latitude", "Longitude"]])

    df["Geocell"] = kmeans.labels_

    centers= kmeans.cluster_centers_
    geocell_centers = pd.DataFrame(
    centers,
    columns=["Center_Latitude", "Center_Longitude"],
    )

    geocell_centers["Geocell"] = geocell_centers.index

    print(geocell_centers.head())
    # --------

    df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
    # Save as parquet file with partitions
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI",  # "Date",
                        "Segment",  # "Geocell"
                        ]
    )


def fn_get_dk_ports(file_path, out_path):
    dtypes = {
        "Harbor": str,
        "Code": str,
        "Location": "object"
    }

    df = pd.read_csv(file_path, sep=";")

    df.columns=['harbor', 'code', 'location']

    df = df[df['code'].str[0:2] == "DK"]

    df.to_csv(out_path, index=False)
    
    print('finished creating csv')

if __name__ == "__main__":
    # Don't need to run this again
    # fn_get_dk_ports('data/port_locodes/raw/port_locodes.csv', 'data/port_locodes/processed/dk_port_locodes.csv')
    
    for f in RAW_FILES:
        fn(f, 'data/aisdk/processed')
    
    save_all_trajectory_plots_simple()

    inspect_one_segment()


