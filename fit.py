from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytensor

# Workaround for the compiler issue in your Windows environment.
# This can slow some computations, but your sampler has run successfully
# with this setting.
pytensor.config.cxx = ""

import arviz as az

from model import build_model, CHANNEL_COLUMNS


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

HOLDOUT_WEEKS = 16

TARGET_COLUMN = "revenue"
DATE_COLUMN = "week_start"
CONTROL_COLUMNS = ["is_event_week"]

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "synthetic_mmm_data.csv"
POSTERIOR_PATH = PROJECT_DIR / "mmm_posterior.nc"
SUMMARY_PATH = PROJECT_DIR / "sampling_summary.csv"
HOLDOUT_PATH = PROJECT_DIR / "holdout_data.csv"
TRAIN_PATH = PROJECT_DIR / "training_data.csv"


# ---------------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# ---------------------------------------------------------

def load_data():
    """Load the weekly synthetic dataset and validate its structure."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        [DATE_COLUMN]
        + CHANNEL_COLUMNS
        + CONTROL_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="raise",
    )

    df = (
        df.sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df[DATE_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate week_start dates were detected."
        )

    # Require weekly observations spaced exactly seven days apart.
    date_differences = df[DATE_COLUMN].diff().dropna()

    if not date_differences.eq(pd.Timedelta(days=7)).all():
        raise ValueError(
            "The dataset does not contain consecutive weekly "
            "observations spaced exactly seven days apart."
        )

    numeric_columns = (
        CHANNEL_COLUMNS
        + CONTROL_COLUMNS
        + [TARGET_COLUMN]
    )

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    if df[numeric_columns].isna().any().any():
        missing_counts = (
            df[numeric_columns]
            .isna()
            .sum()
        )

        raise ValueError(
            "Missing numerical values detected:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    values = df[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Infinite or non-finite numerical values detected."
        )

    if (df[CHANNEL_COLUMNS] < 0).any().any():
        raise ValueError(
            "Advertising spend cannot be negative."
        )

    if (df[TARGET_COLUMN] <= 0).any():
        raise ValueError(
            "Revenue must be positive for this project."
        )

    if not df["is_event_week"].isin([0, 1]).all():
        raise ValueError(
            "is_event_week must contain only 0 and 1."
        )

    if len(df) <= HOLDOUT_WEEKS:
        raise ValueError(
            f"Need more than {HOLDOUT_WEEKS} observations "
            "to create a training set and holdout set."
        )

    print("\nDATA VALIDATION")
    print("=" * 60)
    print("Dataset shape:", df.shape)
    print("Date range:", df[DATE_COLUMN].min(), "to", df[DATE_COLUMN].max())
    print("Missing values:", int(df.isna().sum().sum()))
    print("All numerical values finite:", np.isfinite(values).all())
    print("Weekly frequency: validated")
    print("Channel count:", len(CHANNEL_COLUMNS))

    return df


# ---------------------------------------------------------
# 2. CREATE CHRONOLOGICAL TRAINING AND HOLDOUT SETS
# ---------------------------------------------------------

def split_data(df):
    """Reserve the final 16 weeks for out-of-sample evaluation."""

    train = df.iloc[:-HOLDOUT_WEEKS].copy()
    holdout = df.iloc[-HOLDOUT_WEEKS:].copy()

    if train[DATE_COLUMN].max() >= holdout[DATE_COLUMN].min():
        raise ValueError(
            "Training and holdout periods overlap."
        )

    print("\nDATA SPLIT")
    print("=" * 60)
    print("Training observations:", len(train))
    print("Holdout observations:", len(holdout))
    print(
        "Training dates:",
        train[DATE_COLUMN].min(),
        "to",
        train[DATE_COLUMN].max(),
    )
    print(
        "Holdout dates:",
        holdout[DATE_COLUMN].min(),
        "to",
        holdout[DATE_COLUMN].max(),
    )

    train.to_csv(TRAIN_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)

    print("Training data saved to:", TRAIN_PATH)
    print("Holdout data saved to:", HOLDOUT_PATH)

    return train, holdout


# ---------------------------------------------------------
# 3. FIT MODEL
# ---------------------------------------------------------

def fit_model(train):
    """Fit the MMM using training data only."""

    mmm = build_model()

    # The date, seven channel spends, and event indicator
    # are predictors. Revenue is the target.
    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]

    print("\nMODEL FITTING")
    print("=" * 60)
    print("Predictor shape:", X_train.shape)
    print("Target shape:", y_train.shape)

    mmm.fit(
        X=X_train,
        y=y_train,
        target_accept=0.995,
        random_seed=42,
    )

    if not hasattr(mmm, "idata") or mmm.idata is None:
        raise RuntimeError(
            "Model fitting completed without producing idata."
        )

    # Save posterior samples immediately after fitting.
    mmm.idata.to_netcdf(POSTERIOR_PATH)

    print("\nPosterior saved to:", POSTERIOR_PATH)

    return mmm


# ---------------------------------------------------------
# 4. REPORT SAMPLING DIAGNOSTICS
# ---------------------------------------------------------

def report_diagnostics(mmm):
    """Save parameter summaries and report sampling quality."""

    print("\nSAMPLING DIAGNOSTICS")
    print("=" * 60)

    summary = az.summary(
        mmm.idata,
        round_to=4,
    )

    summary.to_csv(SUMMARY_PATH)

    print("Sampling summary saved to:", SUMMARY_PATH)

    # Report the main model parameters separately from
    # date-level derived contribution variables.
    parameter_names = [
        name
        for name in [
            "adstock_alpha",
            "saturation_lam",
            "saturation_beta",
            "gamma_fourier",
            "gamma_control",
            "intercept_contribution",
            "y_sigma",
        ]
        if name in mmm.idata.posterior.data_vars
    ]

    if parameter_names:
        parameter_summary = az.summary(
            mmm.idata,
            var_names=parameter_names,
            round_to=4,
        )

        print("\nMAIN MODEL PARAMETERS")
        print(parameter_summary.to_string())

    # Inspect the sampler statistics.
    if not hasattr(mmm.idata, "sample_stats"):
        print("WARNING: No sampling statistics are available.")
        return

    stats = mmm.idata.sample_stats

    if "diverging" in stats:
        divergences = int(
            stats["diverging"].sum().item()
        )
    else:
        divergences = None

    print("\nTotal sampling divergences:", divergences)

    if divergences is not None and divergences > 0:
        warnings.warn(
            f"The fit contains {divergences} divergences. "
            "Investigate model parameterization and priors "
            "before trusting the results.",
            RuntimeWarning,
        )

    if "acceptance_rate" in stats:
        mean_acceptance = float(
            stats["acceptance_rate"].mean().item()
        )

        print(
            "Mean acceptance rate:",
            round(mean_acceptance, 4),
        )

    # Evaluate R-hat for the main model parameters.
    if parameter_names:
        rhat = parameter_summary["r_hat"].to_numpy(dtype=float)

        finite_rhat = rhat[np.isfinite(rhat)]

        nonfinite_count = int(
            (~np.isfinite(rhat)).sum()
        )

        print("\nMain parameter R-hat assessment:")

        if finite_rhat.size:
            print(
                "Maximum finite R-hat:",
                round(float(finite_rhat.max()), 4),
            )

        print(
            "Non-finite main parameter R-hat values:",
            nonfinite_count,
        )

        if (
            finite_rhat.size > 0
            and finite_rhat.max() < 1.01
            and nonfinite_count == 0
            and divergences == 0
        ):
            print("Basic sampling gate: PASSED")
        else:
            print("Basic sampling gate: NOT PASSED")

    print("\nFull summary:")
    print(summary)


# ---------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("BAYESIAN MARKETING MIX MODEL")
    print("=" * 60)

    df = load_data()

    train, holdout = split_data(df)

    mmm = fit_model(train)

    report_diagnostics(mmm)

    print("\nFITTING PIPELINE FINISHED.")
    print(
        "This does not mean predictive validation or ROI "
        "recovery has passed."
    )


if __name__ == "__main__":
    main()
