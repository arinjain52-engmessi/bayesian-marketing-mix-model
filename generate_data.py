from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_SEED = 42
N_WEEKS = 120

BASELINE_REVENUE = 15_000_000
SATURATION_SCALE = 100_000
EVENT_REVENUE_EFFECT = 1_000_000
NOISE_STANDARD_DEVIATION = 300_000

OUTPUT_DIR = Path(__file__).resolve().parent

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. CHANNEL PARAMETERS
# ============================================================

# ROI: planted revenue coefficient before saturation.
# Decay: fraction of previous adstock carried into the next week.

CHANNELS = {
    "google_search_brand": {
        "roi": 5.5,
        "decay": 0.12,
    },
    "meta_retargeting": {
        "roi": 3.8,
        "decay": 0.15,
    },
    "google_pmax_demandgen": {
        "roi": 3.0,
        "decay": 0.30,
    },
    "google_remarketing": {
        "roi": 3.0,
        "decay": 0.18,
    },
    "google_search_nonbrand_shop": {
        "roi": 2.6,
        "decay": 0.18,
    },
    "meta_prospecting": {
        "roi": 1.5,
        "decay": 0.40,
    },
    "youtube": {
        "roi": 1.0,
        "decay": 0.60,
    },
}


# ============================================================
# 3. GENERATE WEEKLY DATES AND ADVERTISING SPEND
# ============================================================

dates = pd.date_range(
    start="2024-01-01",
    periods=N_WEEKS,
    freq="W-MON",
)

df = pd.DataFrame(
    {
        "week_start": dates,
    }
)

for channel in CHANNELS:
    df[channel] = rng.uniform(
        low=20_000,
        high=100_000,
        size=N_WEEKS,
    )


# ============================================================
# 4. GENERATE MARKETING-EVENT INDICATOR
# ============================================================

df["is_event_week"] = rng.binomial(
    n=1,
    p=0.15,
    size=N_WEEKS,
)


# ============================================================
# 5. CALCULATE ADSTOCK AND CHANNEL CONTRIBUTIONS
# ============================================================

revenue = np.full(
    N_WEEKS,
    BASELINE_REVENUE,
    dtype=float,
)

adstocked_spend = {}
channel_contributions = {}

for channel, parameters in CHANNELS.items():

    spend = df[channel].to_numpy(dtype=float)

    roi = parameters["roi"]
    decay = parameters["decay"]

    # Geometric adstock:
    # current spend + decayed previous adstock.

    adstock = np.zeros(N_WEEKS, dtype=float)

    for t in range(N_WEEKS):

        previous_adstock = (
            adstock[t - 1]
            if t > 0
            else 0.0
        )

        adstock[t] = (
            spend[t]
            + decay * previous_adstock
        )

    adstocked_spend[channel] = adstock

    # Saturation factor between 0 and 1.
    saturation_factor = (
        adstock
        / (adstock + SATURATION_SCALE)
    )

    # Channel contribution.
    #
    # Contribution / adstocked spend
    # = ROI * saturation_factor.
    #
    # Therefore, ROI is the coefficient before saturation;
    # the realized ratio after saturation is lower.

    contribution = (
        roi
        * adstock
        * saturation_factor
    )

    channel_contributions[channel] = contribution

    revenue += contribution


# ============================================================
# 6. ADD EVENT EFFECT ONCE
# ============================================================

event_contribution = (
    df["is_event_week"].to_numpy(dtype=float)
    * EVENT_REVENUE_EFFECT
)

revenue += event_contribution


# ============================================================
# 7. ADD RANDOM MEASUREMENT NOISE ONCE
# ============================================================

noise = rng.normal(
    loc=0.0,
    scale=NOISE_STANDARD_DEVIATION,
    size=N_WEEKS,
)

df["revenue"] = revenue + noise


# ============================================================
# 8. SAVE SYNTHETIC DATA
# ============================================================

data_path = OUTPUT_DIR / "synthetic_mmm_data.csv"

df.to_csv(
    data_path,
    index=False,
)


# ============================================================
# 9. SAVE PLANTED PARAMETERS
# ============================================================

ground_truth = pd.DataFrame.from_dict(
    CHANNELS,
    orient="index",
)

ground_truth.index.name = "channel"

ground_truth_path = (
    OUTPUT_DIR / "synthetic_ground_truth.csv"
)

ground_truth.to_csv(
    ground_truth_path,
)


# ============================================================
# 10. SAVE GENERATOR DIAGNOSTICS
# ============================================================

diagnostics = pd.DataFrame(
    {
        "channel": list(CHANNELS.keys()),
        "mean_spend": [
            df[channel].mean()
            for channel in CHANNELS
        ],
        "mean_adstocked_spend": [
            adstocked_spend[channel].mean()
            for channel in CHANNELS
        ],
        "planted_roi_coefficient": [
            CHANNELS[channel]["roi"]
            for channel in CHANNELS
        ],
        "decay": [
            CHANNELS[channel]["decay"]
            for channel in CHANNELS
        ],
        "mean_channel_contribution": [
            channel_contributions[channel].mean()
            for channel in CHANNELS
        ],
    }
)

diagnostics_path = (
    OUTPUT_DIR / "synthetic_generation_diagnostics.csv"
)

diagnostics.to_csv(
    diagnostics_path,
    index=False,
)


# ============================================================
# 11. PRINT VALIDATION SUMMARY
# ============================================================

print("=" * 65)
print("SYNTHETIC MMM DATA GENERATION")
print("=" * 65)

print("\nDataset shape:", df.shape)

print(
    "Date range:",
    df["week_start"].min(),
    "to",
    df["week_start"].max(),
)

print("\nMissing values:", int(df.isna().sum().sum()))

print(
    "All revenue values finite:",
    bool(np.isfinite(df["revenue"]).all()),
)

print(
    "\nAverage weekly revenue:",
    f"Rs. {df['revenue'].mean():,.2f}",
)

print(
    "Minimum weekly revenue:",
    f"Rs. {df['revenue'].min():,.2f}",
)

print(
    "Maximum weekly revenue:",
    f"Rs. {df['revenue'].max():,.2f}",
)

print("\nPlanted channel parameters:")
print(ground_truth)

print("\nGeneration diagnostics:")
print(diagnostics.to_string(index=False))

print("\nFiles saved:")
print(data_path)
print(ground_truth_path)
print(diagnostics_path)

print("\nSynthetic data generation complete.")