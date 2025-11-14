import pandas as pd

def inspect_one_segment(
    data_path: str = "data/aisdk/processed",
    max_rows: int = 15,
):
    # Load the processed parquet dataset
    df = pd.read_parquet(data_path)

    print("=== Columns and dtypes ===")
    print(df.dtypes)
    print()

    print("=== First few rows of full df ===")
    print(df.head(10))
    print()

    # Show some unique MMSI / Segment combos
    print("=== Unique (MMSI, Segment) pairs (first 10) ===")
    pairs = df[["MMSI", "Segment"]].drop_duplicates().head(10)
    print(pairs)
    print()

    # Pick the first MMSI+Segment pair
    pair = pairs.iloc[0]
    mmsi = pair["MMSI"]
    segment = pair["Segment"]

    print(f"=== Inspecting MMSI={mmsi}, Segment={segment} ===")

    seg_df = df[(df["MMSI"] == mmsi) & (df["Segment"] == segment)].copy()
    seg_df = seg_df.sort_values("Timestamp")

    print(f"Rows in this segment: {len(seg_df)}")
    print()

    # Show the first few rows for this trajectory
    cols_to_show = [
        col
        for col in ["Timestamp", "MMSI", "Segment", "Latitude", "Longitude", "SOG", "COG", "Geocell"]
        if col in seg_df.columns
    ]

    print("=== Head of this segment ===")
    print(seg_df[cols_to_show].head(max_rows))
    print()

    if "Geocell" in seg_df.columns:
        print("=== Geocell value counts in this segment ===")
        print(seg_df["Geocell"].value_counts().head(10))
        print()

        print("Unique geocells in this segment:", sorted(seg_df["Geocell"].unique()))
