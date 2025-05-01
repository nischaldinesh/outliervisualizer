import pandas as pd
import os

HERE = os.path.dirname(os.path.abspath(__file__))           # …/outliervisualizer/utils
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))  # …/outliervisualizer
DATA_DIR = os.path.join(REPO_ROOT, "datasets")

def load_dataset(name: str) -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.isfile(csv_path):
        raise ValueError(f"Dataset file not found: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load dataset {name} from {csv_path}: {e}")

def load_dataset(name):
    try:
        if name == "AirQualityUCI":
            df = pd.read_csv("/datasets/AirQualityUCI.csv", sep=';', decimal=',')
            df = df.iloc[:, :-2]  
            return df
        elif name == "Bank":
            df = pd.read_csv("/datasets/Bank.csv", sep=';')  
            return df
        elif name == "BeijingClimate":
            return pd.read_csv("/datasets/BeijingClimate.csv")
        elif name == "CardioIsomap":
             return pd.read_json("/datasets/CardioIsomap.json")
        elif name == "CoilDensmap":
             return pd.read_json("/datasets/CoilDensmap.json")
        elif name == "Nhanes":
            return pd.read_csv("/datasets/Nhanes.csv")
        elif name == "Students":
            return pd.read_csv("/datasets/Students.csv")
        elif name == "ObesityDataSet":
            return pd.read_csv("/datasets/ObesityDataSet.csv")
        elif name == "Breast-cancer-wisconsin":
            df = pd.read_csv("/datasets/Breast-cancer-wisconsin.data", header=None, na_values='?')
            df.columns = ['id', 'clump_thickness', 'cell_size', 'cell_shape', 'marginal_adhesion',
                          'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli',
                          'mitoses', 'class']
            df.drop(columns=['id'], inplace=True)
            return df
        elif name == "Iris":
            df = pd.read_csv("/datasets/Iris.data", header=None)
            df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
            return df
        elif name == "PIRSensor":
            return pd.read_csv("/datasets/PIRSensor.csv")
        elif name == "Diabetes":
            return pd.read_csv("/datasets/Diabetes.csv")
        else:
            raise ValueError(f"Unknown dataset name: {name}")
    except Exception as e:
        raise ValueError(f"Failed to load dataset {name}: {e}")