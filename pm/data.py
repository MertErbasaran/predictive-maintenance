"""Download and load the AI4I 2020 Predictive Maintenance dataset (UCI, CC BY 4.0).

10,000 machine records with sensor readings and a label for whether the
machine failed. Source: https://archive.ics.uci.edu/dataset/601
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "ai4i2020.csv"

# Short, code-friendly column names.
RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
}

# Which kind of failure happened. Only known AFTER a failure, so these must
# never be used as model inputs (that would be data leakage).
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def download(path: Path = CSV_PATH) -> Path:
    """Download the CSV once and cache it in data/."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
    path.write_bytes(archive.read(csv_name))
    return path


def load(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(download(path))
    df = df.rename(columns=RENAME)
    df["type"] = pd.Categorical(df["type"], categories=["L", "M", "H"], ordered=True)
    return df
