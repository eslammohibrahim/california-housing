"""Load the California Housing dataset and split it into train and test sets."""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Return the dataset as a single DataFrame (features + target)."""
    return fetch_california_housing(as_frame=True).frame


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features that describe each block better than raw totals."""
    df = df.copy()
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop blocks where the target was capped at 5.0 (the data's upper limit).

    Prices above $500k were recorded as exactly 5.0, so the model can't learn
    the real value for them. Removing them keeps the target honest.
    """
    return df[df[TARGET] < 5.0].reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
