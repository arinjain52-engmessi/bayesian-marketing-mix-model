# Bayesian Marketing Mix Model

A Bayesian Marketing Mix Modeling (MMM) project to investigate how advertising expenditure across multiple channels relates to revenue using synthetic weekly marketing data.

## Project Overview

Marketing Mix Modeling uses statistical methods to estimate the relationship between marketing activity and business outcomes.

This project explores a Bayesian approach to modeling advertising channel effects, incorporating carryover effects and diminishing returns. It uses synthetic data with known planted parameters to enable future evaluation of whether the model can recover the underlying relationships.

## Objectives

- Build a Bayesian MMM using Python and PyMC-Marketing.
- Model advertising carryover effects using geometric adstock.
- Represent diminishing returns using logistic saturation.
- Validate and prepare weekly marketing data.
- Evaluate predictions on a chronological holdout period.
- Investigate channel contribution and return-on-investment estimation.

## Model Design

The current model configuration includes:

- **Seven advertising channels:** Google Search, Meta Retargeting, Google Performance Max, Google Remarketing, Google Search Nonbrand, Meta Prospecting, and YouTube.
- **Geometric adstock:** Represents the potential carryover of advertising effects over time.
- **Logistic saturation:** Represents diminishing marginal returns as advertising exposure increases.
- **Event control:** Includes an indicator for special marketing-event weeks.
- **Bayesian estimation:** Uses posterior sampling to estimate model parameters and quantify uncertainty.

## Dataset

The repository includes synthetic weekly data with 120 observations covering January 2024 through April 2026.

The data contains:
- Weekly advertising expenditure across seven channels.
- A marketing-event indicator.
- Revenue as the target variable.
- A separate file containing planted channel parameters for subsequent model evaluation.

> **Note:** The data is synthetic and does not represent an actual company's performance.

## Technology Stack

- Python
- pandas
- NumPy
- PyMC
- PyMC-Marketing
- ArviZ

## Repository Structure

```text
bayesian-marketing-mix-model/
├── model.py
├── fit.py
├── generate_data.py
├── validate_data.py
├── synthetic_mmm_data.csv
├── synthetic_ground_truth.csv
├── .gitignore
└── README.md
```

## Current Status

- [x] Synthetic dataset generated.
- [x] Dataset validation completed successfully.
- [x] Initial Bayesian MMM configuration implemented.
- [x] Core project files published on GitHub.
- [ ] End-to-end model fitting and sampling diagnostics verified.
- [ ] Holdout revenue predictions evaluated.
- [ ] Recovery of planted channel effects evaluated.
- [ ] Channel contribution and ROI estimates validated.

## Next Steps

1. Verify end-to-end model fitting and posterior sampling.
2. Generate predictions for the final 16 weeks held out from training.
3. Evaluate predictive performance using MAE and RMSE.
4. Compare estimated channel effects with the planted synthetic parameters.
5. Document model limitations and the uncertainty in the estimates.

## Reproducibility and Limitations

This project uses synthetic data rather than real company marketing data. Passing data validation confirms the dataset's structural and numerical integrity; it does not establish predictive accuracy or causal validity.

Model performance, parameter recovery, and channel-level ROI estimates remain to be evaluated.
