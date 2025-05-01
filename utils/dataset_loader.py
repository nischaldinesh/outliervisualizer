import os
import pandas as pd

HERE      = os.path.abspath(os.path.dirname(__file__))      # …/outliervisualizer/utils
REPO_ROOT = os.path.dirname(HERE)                          # …/outliervisualizer
DATA_DIR  = os.path.join(REPO_ROOT, "datasets")            # …/outliervisualizer/datasets

def load_dataset(name: str) -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    # DEBUG:
    print(f"[DEBUG] Looking for dataset at: {csv_path}", flush=True)
    if not os.path.isfile(csv_path):
        raise ValueError(f"Dataset file not found: {csv_path}")
    return pd.read_csv(csv_path)
