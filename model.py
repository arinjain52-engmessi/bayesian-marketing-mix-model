from pymc_extras.prior import Prior

from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    LogisticSaturation,
)


CHANNEL_COLUMNS = [
    "google_search_brand",
    "meta_retargeting",
    "google_pmax_demandgen",
    "google_remarketing",
    "google_search_nonbrand_shop",
    "meta_prospecting",
    "youtube",
]


def build_model() -> MMM:
    """Build the Bayesian Marketing Mix Model."""

    # Priors informed by the scale and design of our synthetic data.
    model_config = {
        "saturation_lam": Prior(
            "Gamma",
            alpha=4,
            beta=1.5,
            dims="channel",
        ),
        "saturation_beta": Prior(
            "HalfNormal",
            sigma=0.15,
            dims="channel",
        ),
    }

    mmm = MMM(
        date_column="week_start",
        channel_columns=CHANNEL_COLUMNS,
        target_column="revenue",
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        control_columns=["is_event_week"],
        yearly_seasonality=2,
        model_config=model_config,
        sampler_config={
            "draws": 1000,
            "tune": 1500,
            "chains": 4,
            "target_accept": 0.995,
        },
    )

    return mmm


if __name__ == "__main__":
    model = build_model()
    print("MMM object created successfully.")
    print(model)