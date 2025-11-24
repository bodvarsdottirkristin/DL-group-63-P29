import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

#from src.datacleaning import fn, fn_get_dk_ports
from src.datasets.window_maker import make_past_future_windows



def main():
    # in_path = "data/aisdk/raw/aisdk-2025-08-29.csv"
    # out_path = "data/aisdk/interim/aisdk-2025-08-29"

    # df = pd.read_csv("data/aisdk/processed/processed_ais_data_to_look_at.csv")
    # #df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # df_new = df.groupby(["MMSI", "Segment"]).apply(resample_segment)

    # #df = resample_segment()

    # print(df_new.head())
    # print(df.head())
    # #fn(in_path, out_path)  

    in_path = "data/aisdk/raw/aisdk-2025-08-29.csv"
    out_path =  "data/aisdk/interim/initial_clean"

    #fn()
    #regularize_segment()

    make_past_future_windows(
        past_min=30,future_min=30,
        input_path="data/aisdk/processed/aisdk_2025", 
        output_path="data/aisdk/processed/windows_30_30")




if __name__ == "__main__":
    main()
    
# TODO: do we need to filter on ships going from/to ports in dk