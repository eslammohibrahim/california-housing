"""Train three models, compare them on a held-out test set, and save the best one."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import RANDOM_STATE, add_features, clean, load_data, split

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "MAE ($)": round(mean_absolute_error(y_test, pred) * 100_000),
        "RMSE ($)": round(np.sqrt(mean_squared_error(y_test, pred)) * 100_000),
        "R2": round(r2_score(y_test, pred), 3),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = add_features(clean(load_data()))
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train: {len(X_train):,} rows  Test: {len(X_test):,} rows\n")

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        model.fit(X_train, y_train)
        scores = evaluate(model, X_test, y_test)
        scores["CV R2 (train)"] = f"{cv.mean():.3f} ± {cv.std():.3f}"
        results[name] = scores
        print(f"{name:<18} {scores}")

    best_name = max(results, key=lambda n: results[n]["R2"])
    best = models[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best, MODEL_DIR / "best_model.joblib")
    (ROOT / "reports" / "results.json").write_text(
        json.dumps({"best_model": best_name, "results": results}, indent=2)
    )

    # Predicted vs actual for the best model
    pred = best.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, s=3, alpha=0.3, color="#3b82f6")
    lim = [0, 5]
    ax.plot(lim, lim, color="#ef4444", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual value ($100k)")
    ax.set_ylabel("Predicted value ($100k)")
    ax.set_title(f"{best_name}: predicted vs actual")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=120)

    # Feature importance from the random forest (easy to read)
    rf = models["Random Forest"]
    imp = sorted(zip(X_train.columns, rf.feature_importances_), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([n for n, _ in imp], [v for _, v in imp], color="#3b82f6")
    ax.set_title("Feature importance (Random Forest)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=120)


if __name__ == "__main__":
    main()
