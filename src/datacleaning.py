import pandas
import pyarrow
import pyarrow.parquet
from sklearn.cluster import KMeans



def resample_segment(g):

    # 1) Ensure Timestamp is datetime and sorted
    g["Timestamp"] = pd.to_datetime(g["Timestamp"])
    g = g.sort_values("Timestamp").set_index("Timestamp")

    # 2) Remember static/categorical info for this segment
    static_cols = ["MMSI", "Ship type", "is_cargo", "Segment"]
    static_vals = {}
    for col in static_cols:
        if col in g.columns:
            static_vals[col] = g[col].iloc[0]

    # 3) Select numeric columns and resample to 1-minute bins
    numeric_cols = g.select_dtypes(include="number").columns
    g_num = g[numeric_cols].resample("1T").mean()

    # 4) Interpolate over time for numeric data
    g_num = g_num.interpolate(method="time")

    # 5) Rebuild DataFrame, re-attach static columns
    g_res = g_num.reset_index()  # bring Timestamp back as column
    for col, val in static_vals.items():
        g_res[col] = val

    # TODO: More research on how to interpolate the data!!

    return g_res


def fn(file_path, out_path):
    '''
    in: filepath as str
    out: filepath as str

    '''
    import shutil, os
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

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
    df = pandas.read_csv(file_path, usecols=usecols, dtype=dtypes)

    df["Ship type"] = df["Ship type"].astype("string").str.strip()
    df = df[df["Ship type"] == "Cargo"]

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]

    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pandas.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

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

    #
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    # Clustering
    # kmeans = KMeans(n_clusters=48, random_state=0)
    # kmeans.fit(df[["Latitude", "Longitude"]])
    # df["Geocell"] = kmeans.labels_
    # centers = kmeans.cluster_centers_
    # "Latitude": center[0],
    # "Longitude": center[1],

    # df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")



    # Save as parquet file with partitions
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=["MMSI",  # "Date",
                        "Segment",  # "Geocell"
                        ]
    )

    df.iloc[:1000].to_csv('data/aisdk/processed/processed_ais_data_to_look_at.csv')

    print(df['Ship type'].unique())


def fn_get_dk_ports(file_path, out_path):
    dtypes = {
        "Harbor": str,
        "Code": str,
        "Location": "object"
    }

    df = pandas.read_csv(file_path, sep=";")
    df.columns=['harbor', 'code', 'location']
    df = df[df['code'].str[0:2] == "DK"]
