from pathlib import Path

import numpy as np
import pandas as pd

from pymc_marketing.mmm import MMM


PROJECT_DIR = Path(__file__).resolve().parent

POSTERIOR_FILE = PROJECT_DIR / "mmm_posterior.nc"
HOLDOUT_FILE = PROJECT_DIR / "holdout_data.csv"

OUTPUT_FILE = PROJECT_DIR / "holdout_predictions.csv"
METRICS_FILE = PROJECT_DIR / "holdout_metrics.csv"

DATE_COLUMN = "week_start"
TARGET_COLUMN = "revenue"

CHANNEL_COLUMNS = [
    "google_search_brand",
    "meta_retargeting",
    "google_pmax_demandgen",
    "google_remarketing",
    "google_search_nonbrand_shop",
    "meta_prospecting",
    "youtube",
]

CONTROL_COLUMNS = ["is_event_week"]


def calculate_metrics(actual, predicted):
    """Calculate regression metrics on the original revenue scale."""

    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: {len(actual)} actual values and "
            f"{len(predicted)} predictions."
        )

    errors = predicted - actual

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    if np.any(actual == 0):
        mape = np.nan
    else:
        mape = np.mean(np.abs(errors / actual)) * 100

    denominator = np.sum(np.abs(actual))
    wape = (
        np.sum(np.abs(errors)) / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "WAPE_percent": float(wape),
    }


def main():
    print("=" * 65)
    print("MMM HOLDOUT EVALUATION")
    print("=" * 65)

    # 1. Verify files.
    for path in (POSTERIOR_FILE, HOLDOUT_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # 2. Load and validate holdout data.
    holdout = pd.read_csv(HOLDOUT_FILE)
    holdout[DATE_COLUMN] = pd.to_datetime(holdout[DATE_COLUMN])

    holdout = (
        holdout.sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if holdout.empty:
        raise ValueError("Holdout dataset is empty.")

    if holdout[DATE_COLUMN].duplicated().any():
        raise ValueError("Duplicate holdout dates detected.")

    required_columns = (
        [DATE_COLUMN]
        + CHANNEL_COLUMNS
        + CONTROL_COLUMNS
        + [TARGET_COLUMN]
    )

    missing = [
        column for column in required_columns
        if column not in holdout.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if holdout[required_columns].isna().any().any():
        raise ValueError("Missing values detected in required columns.")

    y_actual = holdout[TARGET_COLUMN].to_numpy(dtype=float)

    if not np.isfinite(y_actual).all():
        raise ValueError("Actual revenue contains non-finite values.")

    print("\nHoldout observations:", len(holdout))
    print(
        "Date range:",
        holdout[DATE_COLUMN].min(),
        "to",
        holdout[DATE_COLUMN].max(),
    )

    # 3. Prepare the predictor DataFrame.
    # Keep the date column because the MMM was configured with it.
    # Do not include the target column in prediction inputs.
    X_holdout = holdout[
        [DATE_COLUMN] + CHANNEL_COLUMNS + CONTROL_COLUMNS
    ].copy()

    for column in CHANNEL_COLUMNS + CONTROL_COLUMNS:
        X_holdout[column] = pd.to_numeric(
            X_holdout[column],
            errors="raise",
        )

    print("\nPredictor shape:", X_holdout.shape)
    print("Predictor columns:", X_holdout.columns.tolist())

    # 4. Load the saved fitted model.
    mmm = MMM.load(str(POSTERIOR_FILE))

    print("\nSaved posterior loaded.")

    # 5. Generate predictions using a DataFrame.
    print("\nGenerating posterior predictions...")

    try:
        prediction_result = mmm.predict(
            X_holdout,
            extend_idata=False,
        )

    except Exception as exc:
        raise RuntimeError(
            "\nPrediction failed. No evaluation metrics were produced.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error details: {exc}"
        ) from exc

    # 6. Normalize the returned prediction format.
    if hasattr(prediction_result, "values"):
        predicted = np.asarray(
            prediction_result.values,
            dtype=float,
        ).reshape(-1)
    else:
        predicted = np.asarray(
            prediction_result,
            dtype=float,
        ).reshape(-1)

    # Convert predictions back to the original revenue scale.
    target_scale = float(mmm.scalers["_target"].values)
    predicted = predicted * target_scale

    if len(predicted) != len(y_actual):
        raise ValueError(
            f"Expected {len(y_actual)} predictions but received "
            f"{len(predicted)}. Check the prediction output dimensions."
        )

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain non-finite values.")

    # 7. Calculate holdout metrics.
    metrics = calculate_metrics(y_actual, predicted)

    # 8. Save predictions.
    results = pd.DataFrame({
        DATE_COLUMN: holdout[DATE_COLUMN],
        "actual_revenue": y_actual,
        "predicted_revenue": predicted,
        "prediction_error": predicted - y_actual,
        "absolute_error": np.abs(predicted - y_actual),
    })

    results.to_csv(OUTPUT_FILE, index=False)

    # 9. Save metrics.
    metrics_df = pd.DataFrame([{
        "holdout_observations": len(holdout),
        "first_holdout_date": holdout[DATE_COLUMN].min(),
        "last_holdout_date": holdout[DATE_COLUMN].max(),
        **metrics,
    }])

    metrics_df.to_csv(METRICS_FILE, index=False)

    # 10. Report results.
    print("\n" + "=" * 65)
    print("HOLDOUT EVALUATION RESULTS")
    print("=" * 65)

    for name, value in metrics.items():
        if np.isfinite(value):
            print(f"{name}: {value:,.4f}")
        else:
            print(f"{name}: undefined")

    print("\nFirst five predictions:")
    print(results.head().to_string(index=False))

    print("\nPredictions saved to:", OUTPUT_FILE)
    print("Metrics saved to:", METRICS_FILE)

    print(
        "\nEvaluation finished. Prediction accuracy alone does not "
        "establish causal validity or successful channel ROI recovery."
    )


if __name__ == "__main__":
    main()
