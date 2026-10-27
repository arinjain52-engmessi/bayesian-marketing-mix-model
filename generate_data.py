from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. REPRODUCIBLE CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_WEEKS = 120
HOLDOUT_WEEKS = 16
TRAINING_WEEKS = N_WEEKS - HOLDOUT_WEEKS

START_DATE = "2024-01-01"

ADSTOCK_MAX_LAG = 8

BASELINE_REVENUE = 15_000_000
EVENT_REVENUE_EFFECT = 1_000_000
NOISE_STANDARD_DEVIATION = 300_000

MIN_WEEKLY_SPEND = 20_000
MAX_WEEKLY_SPEND = 100_000

OUTPUT_DIR = Path(__file__).resolve().parent

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. PLANTED CHANNEL PARAMETERS
# ============================================================

# alpha:
#   Advertising carryover/retention parameter.
#
# saturation_lam:
#   Controls how quickly the response approaches saturation.
#
# max_contribution:
#   Asymptotic maximum contribution in revenue units.
#   It is NOT the same thing as ROI.

CHANNELS = {
    "google_search_brand": {
        "alpha": 0.12,
        "saturation_lam": 3.0,
        "max_contribution": 1_500_000,
    },
    "meta_retargeting": {
        "alpha": 0.15,
        "saturation_lam": 2.8,
        "max_contribution": 1_300_000,
    },
    "google_pmax_demandgen": {
        "alpha": 0.30,
        "saturation_lam": 2.5,
        "max_contribution": 1_200_000,
    },
    "google_remarketing": {
        "alpha": 0.18,
        "saturation_lam": 3.1,
        "max_contribution": 1_000_000,
    },
    "google_search_nonbrand_shop": {
        "alpha": 0.18,
        "saturation_lam": 3.4,
        "max_contribution": 1_100_000,
    },
    "meta_prospecting": {
        "alpha": 0.40,
        "saturation_lam": 2.3,
        "max_contribution": 900_000,
    },
    "youtube": {
        "alpha": 0.60,
        "saturation_lam": 2.0,
        "max_contribution": 800_000,
    },
}


# ============================================================
# 3. TRANSFORMATION FUNCTIONS
# ============================================================

def geometric_adstock(
    values: np.ndarray,
    alpha: float,
    l_max: int = ADSTOCK_MAX_LAG,
) -> np.ndarray:
    """Apply normalized geometric adstock with a finite lag window."""

    values = np.asarray(values, dtype=float)

    if not 0 <= alpha < 1:
        raise ValueError("alpha must be between 0 inclusive and 1 exclusive.")

    if l_max < 1:
        raise ValueError("l_max must be at least 1.")

    weights = alpha ** np.arange(l_max, dtype=float)
    weights = weights / weights.sum()

    result = np.zeros_like(values, dtype=float)

    for t in range(len(values)):
        available_lags = min(t + 1, l_max)

        result[t] = sum(
            weights[lag] * values[t - lag]
            for lag in range(available_lags)
        )

    return result


def logistic_saturation(
    values: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Apply the PyMC-Marketing logistic saturation function."""

    values = np.asarray(values, dtype=float)

    if lam <= 0:
        raise ValueError("lam must be positive.")

    exp_term = np.exp(-lam * values)

    return (1.0 - exp_term) / (1.0 + exp_term)


# ============================================================
# 4. GENERATE WEEKLY SPEND AND EVENT DATA
# ============================================================

dates = pd.date_range(
    start=START_DATE,
    periods=N_WEEKS,
    freq="W-MON",
)

df = pd.DataFrame({"week_start": dates})

for channel in CHANNELS:
    df[channel] = rng.uniform(
        low=MIN_WEEKLY_SPEND,
        high=MAX_WEEKLY_SPEND,
        size=N_WEEKS,
    )

df["is_event_week"] = rng.binomial(
    n=1,
    p=0.15,
    size=N_WEEKS,
)


# ============================================================
# 5. GENERATE TRUE CHANNEL CONTRIBUTIONS
# ============================================================

revenue = np.full(
    N_WEEKS,
    BASELINE_REVENUE,
    dtype=float,
)

true_contributions = {
    "week_start": dates,
    "baseline_component": np.full(N_WEEKS, BASELINE_REVENUE),
    "event_component": (
        df["is_event_week"].to_numpy(dtype=float)
        * EVENT_REVENUE_EFFECT
    ),
}

revenue += true_contributions["event_component"]

ground_truth_rows = []

for channel, params in CHANNELS.items():

    spend = df[channel].to_numpy(dtype=float)

    # Match the channel scaling learned from the training period.
    # The final 16 weeks remain reserved for holdout evaluation.
    training_max_spend = float(
        df[channel].iloc[:TRAINING_WEEKS].max()
    )

    # Apply normalized geometric adstock to raw spend.
    adstocked_spend = geometric_adstock(
        spend,
        alpha=params["alpha"],
        l_max=ADSTOCK_MAX_LAG,
    )

    # PyMC-Marketing scales channel data by the training maximum.
    scaled_adstock = adstocked_spend / training_max_spend

    # Match the model's logistic saturation shape.
    saturation_response = logistic_saturation(
        scaled_adstock,
        lam=params["saturation_lam"],
    )

    # Contribution approaches max_contribution as saturation increases.
    contribution = (
        params["max_contribution"] * saturation_response
    )

    true_contributions[channel] = contribution
    revenue += contribution

    total_spend = float(spend.sum())
    total_contribution = float(contribution.sum())

    # This is revenue contribution divided by spend, commonly
    # called ROAS. It is not net profit ROI.
    realized_roas = (
        total_contribution / total_spend
        if total_spend > 0
        else np.nan
    )

    ground_truth_rows.append(
        {
            "channel": channel,
            "adstock_alpha": params["alpha"],
            "saturation_lam": params["saturation_lam"],
            "max_contribution": params["max_contribution"],
            "training_max_spend": training_max_spend,
            "total_spend": total_spend,
            "total_true_contribution": total_contribution,
            "mean_true_contribution": float(contribution.mean()),
            "true_roas": realized_roas,
        }
    )


# ============================================================
# 6. ADD MEASUREMENT NOISE ONCE
# ============================================================

noise = rng.normal(
    loc=0.0,
    scale=NOISE_STANDARD_DEVIATION,
    size=N_WEEKS,
)

revenue_without_noise = revenue.copy()
revenue = revenue + noise

if not np.isfinite(revenue).all():
    raise ValueError("Generated revenue contains non-finite values.")

if (revenue <= 0).any():
    raise ValueError("Generated revenue contains non-positive values.")

df["revenue"] = revenue

true_contributions["revenue_without_noise"] = revenue_without_noise
true_contributions["noise_component"] = noise
true_contributions["revenue"] = revenue


# ============================================================
# 7. SAVE DATASETS AND GROUND TRUTH
# ============================================================

data_path = OUTPUT_DIR / "synthetic_mmm_data.csv"
ground_truth_path = OUTPUT_DIR / "synthetic_ground_truth.csv"
contributions_path = OUTPUT_DIR / "synthetic_true_contributions.csv"
diagnostics_path = OUTPUT_DIR / "synthetic_generation_diagnostics.csv"

df.to_csv(data_path, index=False)

ground_truth = pd.DataFrame(ground_truth_rows)
ground_truth.to_csv(ground_truth_path, index=False)

contributions_df = pd.DataFrame(true_contributions)
contributions_df.to_csv(contributions_path, index=False)

diagnostics = ground_truth[
    [
        "channel",
        "training_max_spend",
        "total_spend",
        "total_true_contribution",
        "mean_true_contribution",
        "true_roas",
    ]
].copy()

diagnostics.to_csv(diagnostics_path, index=False)


# ============================================================
# 8. REPORT GENERATION RESULTS
# ============================================================

print("=" * 68)
print("SYNTHETIC MARKETING MIX DATA GENERATION")
print("=" * 68)

print("\nDATASET")
print("-" * 68)
print("Shape:", df.shape)
print("Date range:", df["week_start"].min(), "to", df["week_start"].max())
print("Training observations:", TRAINING_WEEKS)
print("Holdout observations:", HOLDOUT_WEEKS)
print("Advertising channels:", len(CHANNELS))
print("Missing values:", int(df.isna().sum().sum()))
print("Event weeks:", int(df["is_event_week"].sum()))
print("Mean weekly revenue:", round(float(df["revenue"].mean()), 2))
print("Revenue standard deviation:", round(float(df["revenue"].std()), 2))

print("\nPLANTED CHANNEL PARAMETERS")
print("-" * 68)
print(
    ground_truth[
        [
            "channel",
            "adstock_alpha",
            "saturation_lam",
            "max_contribution",
            "true_roas",
        ]
    ].round(4).to_string(index=False)
)

print("\nFILES SAVED")
print("-" * 68)
print(data_path)
print(ground_truth_path)
print(contributions_path)
print(diagnostics_path)

print("\nDATA GENERATION COMPLETED.")
