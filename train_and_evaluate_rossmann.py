# train_and_evaluate_rossmann.py

import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

from forecast_agent_data_rossmann import load_and_prep_data
from forecasting_agent import ForecastingAgent

def run_tests():
    # Run all tests under the tests folder with names starting with test_
    result = subprocess.run(
        ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Tests failed. Aborting training and evaluation.")
        sys.exit(1)

if __name__ == "__main__":
    # Run test pre-checks
    run_tests()
    
    agg_df = load_and_prep_data()

    # 3.1: Snapshot the last date in our aggregated Rossmann data
    last_hist_date = agg_df["date"].max()
    cutoff_date = last_hist_date - timedelta(weeks=4)

    # 3.2: Train/Validation split
    train_df = agg_df[agg_df["date"] <= cutoff_date].copy()
    val_df = agg_df[agg_df["date"] > cutoff_date].copy()

    # 3.3: Fit the hybrid agent on train_df
    print("Fitting ForecastingAgent on Rossmann training data (up to {})...".format(cutoff_date.date()))
    agent = ForecastingAgent(
        prophet_params={"weekly_seasonality": True, "yearly_seasonality": True},
        lstm_params={
            "input_size": 10,
            "hidden_size": 64,  # Increased hidden size for better accuracy
            "num_layers": 2,   # Increased number of layers for better feature extraction
            "output_size": 1,
            "lr": 5e-3,        # Increased learning rate for faster convergence
            "epochs": 3,       # Reduced epochs for faster training
            "batch_size": 128  # Increased batch size for faster training
        },
        seq_len=8
    )
    agent.fit(train_df)

    # 3.4: After fitting, we need train_df with prophet_pred → so we can compute
    #      residuals in predict. Compute once and attach to train_df:
    train_with_pred = agent.compute_residuals(train_df)
    train_with_pred["prophet_pred"] = train_with_pred["prophet_pred"]

    # 3.5: Prepare val “future” features (no units_sold)
    future_features = val_df[[
        "date", "tcin",
        "day_of_week", "week_of_year", "month",
        "region_id", "category_id", "new_launch_flag",
        "promo", "state_holiday_id", "school_holiday_id"
    ]].copy()

    # 3.6: Run inference on 4-week window
    print("Running inference on validation set ({} → {})...".format(cutoff_date.date(), last_hist_date.date()))
    preds = agent.predict(future_features, hist_df=train_with_pred)

    # 3.7: Merge preds with actuals
    eval_df = val_df.merge(
        preds,
        on=["date", "tcin"],
        how="left"
    )
    eval_df["abs_error"] = (eval_df["units_sold"] - eval_df["yhat_final"]).abs()
    eval_df["mape"] = (eval_df["abs_error"] / eval_df["units_sold"].replace(0, np.nan)).clip(upper=1)

    # 3.8: Print summary metrics
    mae = eval_df["abs_error"].mean()
    mape = eval_df["mape"].mean() * 100
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation MAPE: {mape:.2f}%")

    # 3.9: Save a validation report for stakeholders
    report = eval_df[[
        "date", "tcin", "units_sold", "yhat_prophet", "yhat_residual", "yhat_final"
    ]]
    report.to_csv("rossmann_validation_report.csv", index=False)
    print("Saved validation report → rossmann_validation_report.csv")
