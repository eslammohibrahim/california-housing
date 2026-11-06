"""Tests for the data pipeline. They use small hand-made tables, so no download is needed."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import TARGET, add_features, clean, split  # noqa: E402


@pytest.fixture
def blocks() -> pd.DataFrame:
    return pd.DataFrame({
        "MedInc": [3.0, 5.0, 8.0, 2.0, 6.0],
        "HouseAge": [20, 35, 10, 50, 15],
        "AveRooms": [5.0, 6.0, 8.0, 4.0, 6.5],
        "AveBedrms": [1.0, 1.2, 1.6, 1.0, 1.3],
        "Population": [1000, 800, 1200, 500, 900],
        "AveOccup": [2.5, 3.0, 2.0, 4.0, 2.6],
        "Latitude": [34.0, 37.5, 33.9, 36.0, 38.2],
        "Longitude": [-118.2, -122.3, -118.4, -119.8, -121.5],
        TARGET: [1.5, 2.8, 5.0, 0.9, 3.4],
    })


def test_clean_drops_capped_prices(blocks):
    cleaned = clean(blocks)
    assert (cleaned[TARGET] < 5.0).all()
    assert len(cleaned) == 4


def test_add_features_computes_ratios(blocks):
    out = add_features(blocks)
    assert out.loc[0, "RoomsPerPerson"] == pytest.approx(5.0 / 2.5)
    assert out.loc[0, "BedroomRatio"] == pytest.approx(1.0 / 5.0)
    assert "RoomsPerPerson" not in blocks.columns  # the input is not modified


def test_split_keeps_target_out_of_features(blocks):
    X_train, X_test, y_train, y_test = split(blocks, test_size=0.4)
    assert TARGET not in X_train.columns
    assert len(X_train) + len(X_test) == len(blocks)
    assert len(y_test) == len(X_test)


def test_split_is_reproducible(blocks):
    first = split(blocks, test_size=0.4)[1].index.tolist()
    second = split(blocks, test_size=0.4)[1].index.tolist()
    assert first == second
