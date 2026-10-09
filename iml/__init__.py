import pandas as pd
import numpy as np

from .simple_regressor import SimpleRegressor

def load_powerplant() -> tuple[np.array, np.array]:
    df = pd.read_csv("data/powerplant.csv")
    return df["x"].to_numpy(), df["y"].to_numpy()

def load_ulu() -> np.array:
    return pd.read_csv("data/ulu.csv").to_numpy()
