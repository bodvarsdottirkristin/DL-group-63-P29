import pandas as pd
import numpy as np


from src.datacleaning import fn, fn_get_dk_ports

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


def main():
    in_path = "data/aisdk/raw/aisdk-2025-08-29.csv"
    out_path = "data/aisdk/interim/aisdk-2025-08-29"

    df = pd.read_csv("data/aisdk/processed/processed_ais_data_to_look_at.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.groupby(["MMSI", "Segment"]).apply(resample_segment)

    df = df.reset_index(drop=True)

    print(df)
    #fn(in_path, out_path)  




if __name__ == "__main__":
    main()
    
# TODO: do we need to filter on ships going from/to ports in dk