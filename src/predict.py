"""Predict the price for one example block with the saved model.

Usage:
    python src/predict.py
"""

from pathlib import Path

import joblib
import pandas as pd

from data import add_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"

EXAMPLE = {
    "MedInc": 5.0,        # median income in the block ($10k units)
    "HouseAge": 20.0,
    "AveRooms": 5.5,
    "AveBedrms": 1.05,
    "Population": 1200.0,
    "AveOccup": 2.8,
    "Latitude": 34.05,    # Los Angeles area
    "Longitude": -118.25,
}


def main() -> None:
    model = joblib.load(MODEL_PATH)
    X = add_features(pd.DataFrame([EXAMPLE]))
    price = model.predict(X)[0] * 100_000
    print(f"Predicted median house value: ${price:,.0f}")


if __name__ == "__main__":
    main()
