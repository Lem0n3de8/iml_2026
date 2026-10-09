import pandas as pd
import numpy as np

from .simple_regressor import SimpleRegressor
from .simple_classifier import SimpleClassifier, show_iris_classifier

from sklearn.datasets import load_iris

def load_powerplant() -> tuple[np.array, np.array]:
    df = pd.read_csv("data/powerplant.csv")
    return df["x"].to_numpy(), df["y"].to_numpy()

def load_ulu() -> np.array:
    return pd.read_csv("data/ulu.csv").to_numpy()
