import pandas
import numpy as np


from src.datacleaning import fn, fn_get_dk_ports


def main():
    in_path = "data/aisdk/raw/aisdk-2025-08-29.csv"
    out_path = "data/aisdk/interim/aisdk-2025-08-29"

    fn(in_path, out_path)

if __name__ == "__main__":
    main()
    


# We have to:
#   Filter so only cargo vessels
#   Only vessels going to ports within denmark