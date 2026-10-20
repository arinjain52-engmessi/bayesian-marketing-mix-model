from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = PROJECT_DIR / "synthetic_mmm_data.csv"
TRUTH_FILE = PROJECT_DIR / "synthetic_ground_truth.csv"

CHANNEL_COLUMNS = [
    "google_search_brand",
    "meta_retargeting",
    "google_pmax_demandgen",
    "google_remarketing",
    "google_search_nonbrand_shop",
    "meta_prospecting",
    "youtube",
]

REQUIRED_COLUMNS = [
    "week_start",
    *CHANNEL_COLUMNS,
    "is_event_week",
    "revenue",
]


def main():
    print("=" * 60)
    print("SYNTHETIC MMM DATA VALIDATION")
    print("=" * 60)

    # 1. Load data.
    df = pd.read_csv(DATA_FILE)
    truth = pd.read_csv(TRUTH_FILE)

    print("\n1. DATASET")
    print("-" * 60)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # 2. Check required columns.
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Required columns: PASS")

    # 3. Check missing values.
    missing_count = int(df[REQUIRED_COLUMNS].isna().sum().sum())

    print("\n2. MISSING VALUES")
    print("-" * 60)
    print("Missing values:", missing_count)

    if missing_count:
        raise ValueError("Dataset contains missing values.")

    # 4. Check dates.
    df["week_start"] = pd.to_datetime(df["week_start"])

    dates = df["week_start"]

    duplicate_dates = int(dates.duplicated().sum())

    intervals = dates.sort_values().diff().dropna()
    irregular_intervals = int((intervals != pd.Timedelta(days=7)).sum())

    print("\n3. DATE VALIDATION")
    print("-" * 60)
    print("First date:", dates.min())
    print("Last date:", dates.max())
    print("Duplicate dates:", duplicate_dates)
    print("Irregular weekly intervals:", irregular_intervals)

    # 5. Check numerical values.
    numeric_columns = [
        *CHANNEL_COLUMNS,
        "is_event_week",
        "revenue",
    ]

    numeric_values = df[numeric_columns].to_numpy(dtype=float)

    finite_values = bool(np.isfinite(numeric_values).all())

    print("\n4. NUMERICAL VALIDATION")
    print("-" * 60)
    print("All numerical values finite:", finite_values)
    print("Negative revenue values:", int((df["revenue"] < 0).sum()))

    negative_spend_count = int(
        (df[CHANNEL_COLUMNS] < 0).sum().sum()
    )

    print("Negative spend values:", negative_spend_count)

    # 6. Validate event indicator.
    event_values = set(df["is_event_week"].unique())

    print("\n5. EVENT CONTROL")
    print("-" * 60)
    print("Observed values:", sorted(event_values))
    print("Event weeks:", int(df["is_event_week"].sum()))

    # 7. Validate ground truth.
    print("\n6. PLANTED PARAMETERS")
    print("-" * 60)

    print(truth.to_string(index=False))

    truth_channels = set(truth["channel"])
    data_channels = set(CHANNEL_COLUMNS)

    print(
        "\nGround-truth channel names match:",
        truth_channels == data_channels,
    )

    # 8. Summarise spend and revenue.
    print("\n7. SUMMARY STATISTICS")
    print("-" * 60)

    print(
        df[CHANNEL_COLUMNS + ["revenue"]]
        .describe()
        .round(2)
        .to_string()
    )

    # 9. Final checks.
    checks = {
        "Required columns": not missing_columns,
        "No missing values": missing_count == 0,
        "No duplicate dates": duplicate_dates == 0,
        "Weekly intervals": irregular_intervals == 0,
        "Finite numerical values": finite_values,
        "Non-negative spend": negative_spend_count == 0,
        "Valid ground-truth channels": truth_channels == data_channels,
    }

    print("\n8. FINAL VALIDATION")
    print("-" * 60)

    for check_name, passed in checks.items():
        print(f"{check_name}: {'PASS' if passed else 'FAIL'}")

    if all(checks.values()):
        print("\nDATA VALIDATION PASSED")
    else:
        print("\nDATA VALIDATION FAILED")


if __name__ == "__main__":
    main()