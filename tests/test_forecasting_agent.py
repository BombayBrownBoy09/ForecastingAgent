import unittest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from forecasting_agent import ForecastingAgent, ResidualLSTM, SalesDataset

# File: /Users/bhargav/ForecastingAgent/test_forecasting_agent.py
import torch.nn as nn

class TestForecastingAgent(unittest.TestCase):
    def setUp(self):
        self.seq_len = 3
        # Create dataset for tcin "A"
        dates_A = pd.date_range(start="2022-01-01", periods=10, freq='D')
        data_A = {
            "date": dates_A,
            "tcin": ["A"] * 10,
            "units_sold": np.arange(1, 11),  # 1 to 10
            "day_of_week": [d.weekday() for d in dates_A],
            "week_of_year": [d.isocalendar()[1] for d in dates_A],
            "month": [d.month for d in dates_A],
            "region_id": [1] * 10,
            "category_id": [1] * 10,
            "new_launch_flag": [0] * 10,
            "promo": [0] * 10,
            "state_holiday_id": [0] * 10,
            "school_holiday_id": [0] * 10
        }
        df_A = pd.DataFrame(data_A)
        
        # Create dataset for tcin "B"
        dates_B = pd.date_range(start="2022-01-05", periods=8, freq='D')
        data_B = {
            "date": dates_B,
            "tcin": ["B"] * 8,
            "units_sold": np.arange(11, 19),  # 11 to 18
            "day_of_week": [d.weekday() for d in dates_B],
            "week_of_year": [d.isocalendar()[1] for d in dates_B],
            "month": [d.month for d in dates_B],
            "region_id": [2] * 8,
            "category_id": [2] * 8,
            "new_launch_flag": [0] * 8,
            "promo": [1] * 8,
            "state_holiday_id": [0] * 8,
            "school_holiday_id": [1] * 8
        }
        df_B = pd.DataFrame(data_B)
        
        # Combine datasets
        self.df = pd.concat([df_A, df_B], ignore_index=True)
    
    def test_fit_prophet_and_compute_residuals(self):
        agent = ForecastingAgent(seq_len=self.seq_len)
        agent.fit_prophet(self.df)
        self.assertIn("A", agent.prophet_models)
        self.assertIn("B", agent.prophet_models)
        df_resid = agent.compute_residuals(self.df)
        self.assertIn("prophet_pred", df_resid.columns)
        self.assertIn("residual", df_resid.columns)

    def test_fit(self):
        lstm_params = {
            "input_size": 10,
            "hidden_size": 4,
            "num_layers": 1,
            "output_size": 1,
            "lr": 1e-3,
            "epochs": 1,
            "batch_size": 2
        }
        agent = ForecastingAgent(seq_len=self.seq_len, lstm_params=lstm_params)
        df_resid = agent.fit(self.df)
        self.assertIn("prophet_pred", df_resid.columns)
        self.assertTrue(len(agent.residual_models) > 0)

    def test_predict(self):
        lstm_params = {
            "input_size": 10,
            "hidden_size": 4,
            "num_layers": 1,
            "output_size": 1,
            "lr": 1e-3,
            "epochs": 1,
            "batch_size": 2
        }
        agent = ForecastingAgent(seq_len=self.seq_len, lstm_params=lstm_params)
        # Fit both Prophet and LSTM models
        df_train = self.df.copy()
        df_train = agent.fit(df_train)
        
        # Create dummy future features for tcin "A"
        future_dates = pd.date_range(start="2022-01-12", periods=2, freq='D')
        future_data = {
            "date": future_dates,
            "tcin": ["A"] * 2,
            "day_of_week": [d.weekday() for d in future_dates],
            "week_of_year": [d.isocalendar()[1] for d in future_dates],
            "month": [d.month for d in future_dates],
            "region_id": [1] * 2,
            "category_id": [1] * 2,
            "new_launch_flag": [0] * 2,
            "promo": [0] * 2,
            "state_holiday_id": [0] * 2,
            "school_holiday_id": [0] * 2
        }
        df_future = pd.DataFrame(future_data)
        preds = agent.predict(df_future, hist_df=self.df)
        self.assertFalse(preds.empty)
        self.assertTrue({"date", "tcin", "yhat_prophet", "yhat_residual", "yhat_final"}.issubset(set(preds.columns)))

    def test_predict_no_model(self):
        # Test predict for tcin with no fitted Prophet model
        agent = ForecastingAgent(seq_len=self.seq_len)
        agent.fit_prophet(self.df)
        future_dates = pd.date_range(start="2022-01-15", periods=2, freq='D')
        future_data = {
            "date": future_dates,
            "tcin": ["C"] * 2,  # tcin "C" not in training data
            "day_of_week": [d.weekday() for d in future_dates],
            "week_of_year": [d.isocalendar()[1] for d in future_dates],
            "month": [d.month for d in future_dates],
            "region_id": [3] * 2,
            "category_id": [3] * 2,
            "new_launch_flag": [0] * 2,
            "promo": [0] * 2,
            "state_holiday_id": [0] * 2,
            "school_holiday_id": [0] * 2
        }
        df_future = pd.DataFrame(future_data)
        preds = agent.predict(df_future, hist_df=self.df)
        self.assertTrue(preds.empty)
    
    def test_sales_dataset_rolling_windows(self):
        # Test that SalesDataset creates the correct number of rolling windows
        dataset = SalesDataset(self.df, seq_len=self.seq_len)
        # For each group, windows = len(group) - seq_len.
        expected_windows = 0
        for _, group in self.df.groupby("tcin"):
            expected_windows += max(0, len(group) - self.seq_len)
        self.assertEqual(len(dataset), expected_windows)
        # Check shape of first sample
        if len(dataset) > 0:
            seq_x, seq_y = dataset[0]
            self.assertEqual(seq_x.shape, (self.seq_len, 10))

    def test_residual_lstm_forward(self):
        # Test a forward pass in ResidualLSTM; mimic an input shape.
        input_size = 10
        hidden_size = 4
        seq_len = self.seq_len
        model = ResidualLSTM(input_size=input_size, hidden_size=hidden_size)
        # Create dummy batch: (batch, seq_len, input_size)
        dummy_input = torch.rand((2, seq_len, input_size))
        output = model(dummy_input)
        # Output should be of shape (batch,)
        self.assertEqual(output.shape, (2,))

# Integration Instructions:
# To integrate these tests into your CI/CD pipeline, ensure the tests directory is included in your
# source repository. Then, add a step in your pipeline configuration (e.g., GitHub Actions, Azure Pipelines)
# to run the command:
#
#     python -m unittest discover -s /Users/bhargav/ForecastingAgent -p "test_*.py"
#
# This will run all tests prefixed with "test_". Additionally, you can invoke tests as a pre-check
# in your train_and_evaluate_rossmann.py script before training.
  
if __name__ == "__main__":
    unittest.main()