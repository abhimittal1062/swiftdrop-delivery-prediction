"""Load the SwiftDrop delivery dataset from a local path or its public source."""
from pathlib import Path
import pandas as pd

DATA_URL = (
    "https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA/resolve/main/zomato_cleaned.csv"
)

def load_dataset(path=None) -> pd.DataFrame:
    """Load CSV from `path`; use the public source URL when path is omitted."""
    if path:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip()
    return df
