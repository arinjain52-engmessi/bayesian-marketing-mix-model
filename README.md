# Bayesian Marketing Mix Model

A Bayesian Marketing Mix Model built with Python and PyMC-Marketing to estimate advertising channel contributions and study advertising carryover, diminishing returns, and revenue prediction.

## Project Features

- Generate 120 weeks of synthetic marketing data.
- Model seven advertising channels.
- Apply geometric adstock and logistic saturation.
- Account for marketing events and yearly seasonality.
- Validate data quality and split data chronologically.
- Fit a Bayesian model using Markov chain Monte Carlo (MCMC).
- Inspect posterior estimates and sampling diagnostics.
- Evaluate holdout predictions against simple baseline forecasts.
- Compare estimated channel contributions with synthetic ground truth.

## Technology Stack

Python, NumPy, pandas, PyMC, PyMC-Marketing, and ArviZ.

## Project Files

- `generate_data.py` — generates synthetic data.
- `validate_data.py` — validates the dataset.
- `model.py` — defines the Bayesian model.
- `fit.py` — fits the model and reports sampling diagnostics.
- `evaluate_model.py` — evaluates holdout predictions.
- `requirements.txt` — lists dependencies.

## Running the Project

Install the dependencies:

    pip install -r requirements.txt

Run the scripts from the project directory:

    python generate_data.py
    python validate_data.py
    python fit.py
    python evaluate_model.py

The workflow uses 104 weekly observations for training and 16 for holdout evaluation.

## Latest Holdout Results

The latest recorded evaluation on 16 synthetic holdout observations reported:

| Metric | Bayesian MMM | Training-mean baseline |
|---|---:|---:|
| MAE | Rs. 218,368 | Rs. 416,962 |
| RMSE | Rs. 321,210 | Rs. 526,177 |
| WAPE | 1.0655% | 2.0346% |

These results are specific to the synthetic dataset. Predictive accuracy alone does not establish causal validity or confirm that every channel's return on ad spend has been recovered correctly.

The model's sampling diagnostics and channel-level contribution recovery require separate assessment.

## Limitations

- The data is synthetic, not real business data.
- The holdout set contains only 16 observations.
- Channel attribution and ROAS recovery require additional validation.

## Author

Arin Jain