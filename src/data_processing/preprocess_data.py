import pandas as pd
import pyarrow
import pyarrow.parquet
import importlib
import trajectory_segmentation
importlib.reload(trajectory_segmentation)
from coord_to_utm import to_utm
from trajectory_segmentation import segment_trajectories

# -------------------------------------------------------------------
# Config / constants
# -------------------------------------------------------------------

AIS_DTYPES = {
    "MMSI": "object",
    "SOG": float,
    "COG": float,
    "Longitude": float,
    "Latitude": float,
    "# Timestamp": "object",
    "Type of mobile": "object",
    "Ship type": "object",
}
AIS_USECOLS = list(AIS_DTYPES.keys())

DEFAULT_BBOX = (60, 0, 50, 20)  # (north, west, south, east)
KNOTS_TO_MS = 0.514444


# -------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------

def read_zip_to_df(zip_path: str,
                   dtypes: dict = AIS_DTYPES,
                   usecols: list[str] = AIS_USECOLS) -> pd.DataFrame:
    """Read AIS CSV directly from ZIP into a DataFrame."""
    print(f"Reading {zip_path} ...")
    df = pd.read_csv(
        zip_path,
        usecols=usecols,
        dtype=dtypes,
        compression="zip",
    )
    print("Columns in raw DF:", df.columns.tolist())
    return df


def filter_by_ship_type(df: pd.DataFrame, ship_type: str = "Cargo") -> pd.DataFrame:
    """Keep only rows with a given Ship type and drop the column."""
    df = df[df["Ship type"] == ship_type].drop(columns=["Ship type"])
    print(f"Filtered to Ship type == {ship_type} and dropped column.")
    return df


def filter_by_bbox(df: pd.DataFrame,
                   bbox: tuple[float, float, float, float] = DEFAULT_BBOX) -> pd.DataFrame:
    """Filter rows to a geographic bounding box (north, west, south, east)."""
    north, west, south, east = bbox
    df = df[
        (df["Latitude"] <= north)
        & (df["Latitude"] >= south)
        & (df["Longitude"] >= west)
        & (df["Longitude"] <= east)
    ]
    print(f"Applied geographic bounding box {bbox}.")
    return df


def filter_by_mobile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Class A and Class B and drop Type of mobile."""
    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(
        columns=["Type of mobile"]
    )
    print("Filtered to Type of mobile in ['Class A', 'Class B'] and dropped column.")
    return df


def clean_mmsi(df: pd.DataFrame) -> pd.DataFrame:
    """Filter MMSI by length and MID range."""
    # 9-digit MMSI
    df = df[df["MMSI"].str.len() == 9]
    # MID between 200 and 775
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]
    print("Applied MMSI format and MID filters.")
    return df


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Rename '# Timestamp' to 'Timestamp' and parse datetime."""
    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    print("Parsed Timestamp column.")
    return df


def drop_duplicate_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (Timestamp, MMSI) pairs."""
    before = len(df)
    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")
    after = len(df)
    print(f"Dropped {before - after} duplicate (Timestamp, MMSI) rows.")
    return df


def convert_sog_to_ms(df: pd.DataFrame,
                      col: str = "SOG") -> pd.DataFrame:
    """Convert SOG from knots to m/s."""
    df[col] = KNOTS_TO_MS * df[col]
    print(f"Converted {col} from knots to m/s.")
    return df


def save_parquet_partitioned(df: pd.DataFrame,
                             out_path: str,
                             partition_cols: list[str] = ["MMSI", "Segment"]) -> None:
    """Save DataFrame as a partitioned Parquet dataset."""
    print(f"Saving to parquet dataset at {out_path} ...")
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=partition_cols,
    )
    print("Parquet save done.")


# -------------------------------------------------------------------
# High-level pipeline
# -------------------------------------------------------------------

def process_zip(zip_path: str,
                out_path: str,
                ship_type: str = "Cargo",
                bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
                gap_minutes: int = 15) -> pd.DataFrame:
    """
    Full AIS processing pipeline:
    - read from ZIP
    - filter ship type, bbox, mobile class, MMSI
    - parse timestamps, drop duplicates
    - track and segment filtering
    - convert SOG to m/s
    - save as partitioned Parquet
    Returns the final cleaned DataFrame.
    """
    print(f"\n=== Processing {zip_path} ===")
    print(f"Output path: {out_path}")

    # 1) Load
    df = read_zip_to_df(zip_path)

    # 2) Basic filters
    df = filter_by_ship_type(df, ship_type=ship_type)
    df = filter_by_bbox(df, bbox=bbox)
    df = filter_by_mobile_class(df)
    df = clean_mmsi(df)
    df = parse_timestamp(df)
  

    # 4) Unit conversion
    df = convert_sog_to_ms(df, col="SOG")

    # 5) Add geospatial UTM coordinates
    df = to_utm(df)

    print("Final columns:", df.columns.tolist())
    print(f"Rows after filtering: {len(df)}")

    # 5) Save
    save_parquet_partitioned(df, out_path=out_path, partition_cols=["MMSI"])
    print(f"=== Done for {zip_path} ===\n")

    return df
