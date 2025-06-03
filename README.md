## Overview

This repository implements a hybrid sales‐forecasting agent that combines Facebook Prophet for baseline trends and a small LSTM network for residual correction. The core component is `forecasting_agent.py`, which defines:

* **`ResidualLSTM`**: a PyTorch module that takes a rolling window of residuals + static features and outputs a single residual adjustment.
* **`SalesDataset`**: a PyTorch `Dataset` that builds sequence windows from historical sales (residuals + engineered features).
* **`ForecastingAgent`**: orchestrates the pipeline. For each SKU (identified by `tcin`), it:

  1. Fits a Prophet model on historical `date` & `units_sold`.
  2. Computes Prophet residuals (`residual = actual − yhat`).
  3. Trains the LSTM on those residuals (using `SalesDataset`).
  4. Offers a `predict(df_future, hist_df)` method that returns a DataFrame with columns `yhat_prophet`, `yhat_residual`, and `yhat_final` (sum of the two).

---

## Installation

1. Clone or unzip this repo so that `forecasting_agent.py` is in your Python path.
2. Create a new virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies (see `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

   * Required packages include: `pandas`, `numpy`, `scikit-learn`, `prophet`, `torch`, `matplotlib`.

---

## Usage

### 1. Import & Initialize

```python
from forecasting_agent import ForecastingAgent

agent = ForecastingAgent(
    prophet_params={"weekly_seasonality": True, "yearly_seasonality": True},
    lstm_params={
        "input_size": 10,     # (1 residual + 9 features)
        "hidden_size": 32,
        "num_layers": 1,
        "output_size": 1,
        "lr": 1e-3,
        "epochs": 5,
        "batch_size": 64
    },
    seq_len=8
)
```

### 2. Fit on Historical Data

* Prepare a DataFrame `df_hist` with columns:

  ```
  date (datetime),
  tcin (string or int),
  units_sold (numeric),
  day_of_week,
  week_of_year,
  month,
  region_id,
  category_id,
  new_launch_flag,
  promo,
  state_holiday_id,
  school_holiday_id
  ```
* Call:

  ```python
  df_resid = agent.fit(df_hist)
  ```

  This:

  * Fits Prophet per `tcin`.
  * Appends a `prophet_pred` column to `df_hist`.
  * Computes residuals and trains the LSTM on those residuals.

### 3. Generate Future Predictions

* Prepare a “future” DataFrame `df_future` with the same columns as `df_hist` **except** for `units_sold`. It must include the date range for which you want forecasts.
* Call:

  ```python
  df_preds = agent.predict(
      df_future,
      df_hist.assign(prophet_pred=df_resid["prophet_pred"])
  )
  ```

  * `df_preds` will contain columns:

    ```
    date, tcin, yhat_prophet, yhat_residual, yhat_final
    ```

    where `yhat_final = yhat_prophet + yhat_residual`.

### 4. Example Script

An end‐to‐end example is provided in `train_and_evaluate_rossmann.py`, which:

* Loads `train.csv` & `store.csv`.
* Merges & prepares the Rossmann dataset (`forecast_agent_data_rossmann.py`).
* Splits into train/validation.
* Fits the `ForecastingAgent` and saves a validation report (`rossmann_validation_report.csv`).
* Generates and plots sample forecasts (`plot_performance.py`).

To run:

```bash
python train_and_evaluate_rossmann.py
```

Make sure the CSV files (`train.csv`, `store.csv`) are in the same directory or update paths accordingly.

---

## Directory Structure

```
ForecastingAgent-main/
├── forecasting_agent.py              # Core module (ResidualLSTM, SalesDataset, ForecastingAgent)
├── forecast_agent_data_rossmann.py   # Data‐prep utilities for Rossmann example
├── train_and_evaluate_rossmann.py    # Sample script that trains & validates on Rossmann dataset
├── plot_performance.py               # Helper to visualize errors & predictions
├── inventory_copilot_interface.py    # (Optional) Streamlit app / OpenAI integration for inventory queries
├── requirements.txt                  # pip dependencies
├── train.csv, store.csv              # Raw Rossmann data for examples
├── rossmann_validation_report.csv    # Output report from example script
└── results/
    ├── error_distribution.png
    └── store_1_forecast.png
```

---

## Key Points

* The hybrid approach relies on Prophet to capture long‐term/seasonal trends and the LSTM to learn short‐term patterns (residuals).
* You can adjust:

  * `seq_len` (number of historical weeks for each LSTM window).
  * LSTM hyperparameters (`hidden_size`, `num_layers`, learning rate, epochs, batch size).
  * Prophet hyperparameters (e.g., `seasonality_mode="multiplicative"`).
* If a given `tcin` has insufficient data (`< seq_len + 1`), the LSTM is skipped and only Prophet’s forecast is returned.

---

## Dependencies

Listed in `requirements.txt`:

```
pandas
numpy
scikit-learn
prophet
torch
matplotlib
streamlit
openai
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Notes

* Ensure your historical DataFrame uses `datetime` dtype for the `date` column (Prophet expects `ds`).
* The LSTM expects all engineered features (`day_of_week`, `week_of_year`, `month`, etc.) to be numeric (ints or floats).
* For custom datasets, replicate the same feature engineering steps you see in `forecast_agent_data_rossmann.py`.
