"""Quick exploratory analysis: summary stats and a few plots saved to reports/figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import TARGET, load_data

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print(f"Rows: {len(df):,}  Columns: {df.shape[1]}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    capped = (df[TARGET] >= 5.0).mean()
    print(f"Blocks with price capped at $500k: {capped:.1%}\n")
    print(df.describe().T.round(2))

    print("\nCorrelation with price:")
    print(df.corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3))

    # Price distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[TARGET], bins=60, color="#3b82f6")
    ax.set_xlabel("Median house value ($100k)")
    ax.set_ylabel("Number of blocks")
    ax.set_title("Price distribution (note the spike at the $500k cap)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_distribution.png", dpi=120)

    # Price on the map
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df[TARGET], s=2, cmap="viridis")
    fig.colorbar(sc, label="Median house value ($100k)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Prices by location")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_map.png", dpi=120)

    # Income vs price
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["MedInc"], df[TARGET], s=2, alpha=0.3, color="#3b82f6")
    ax.set_xlabel("Median income ($10k)")
    ax.set_ylabel("Median house value ($100k)")
    ax.set_title("Income is the strongest single predictor")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "income_vs_price.png", dpi=120)

    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
