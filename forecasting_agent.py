import pandas as pd
import numpy as np

from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 2.1: Residual‐predicting LSTM
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)

# 2.2: Dataset that creates rolling windows for LSTM training
class SalesDataset(Dataset):
    def __init__(self, df, seq_len=8, target_col="units_sold"):
        self.seq_len = seq_len
        self.data = []
        grouped = df.groupby("tcin")
        for tcin, group in grouped:
            group_sorted = group.sort_values("date")
            # Use numeric holiday‐ID columns
            features = group_sorted[
                [
                    "units_sold",
                    "day_of_week",
                    "week_of_year",
                    "month",
                    "region_id",
                    "category_id",
                    "new_launch_flag",
                    "promo",
                    "state_holiday_id",    
                    "school_holiday_id"   
                ]
            ].values.astype(np.float32)
            for i in range(len(features) - seq_len):
                seq_x = features[i : i + seq_len]
                seq_y = features[i + seq_len, 0]  # next units_sold
                self.data.append((seq_x, seq_y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_x, seq_y = self.data[idx]
        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32)
        )

# 2.3: The Hybrid ForecastingAgent
class ForecastingAgent:
    def __init__(self, prophet_params=None, lstm_params=None, seq_len=8):
        self.prophet_params = prophet_params or {}
        self.lstm_params = lstm_params or {
            "input_size": 10,     # units_sold + 9 engineered features
            "hidden_size": 32,
            "num_layers": 1,
            "output_size": 1,
            "lr": 1e-3,
            "epochs": 5,
            "batch_size": 64
        }
        self.seq_len = seq_len
        self.prophet_models = {}   # {tcin: Prophet()}
        self.residual_models = {}  # {tcin: ResidualLSTM()}

    def fit_prophet(self, df):
        for tcin, group in df.groupby("tcin"):
            prophet_df = group[["date", "units_sold"]].rename(
                columns={"date": "ds", "units_sold": "y"}
            )
            m = Prophet(**self.prophet_params)
            m.fit(prophet_df)
            self.prophet_models[tcin] = m

    def compute_residuals(self, df):
        df = df.copy()
        df["prophet_pred"] = 0.0
        for tcin, group in df.groupby("tcin"):
            m = self.prophet_models.get(tcin)
            if m is not None:
                tmp = group[["date"]].rename(columns={"date": "ds"})
                yhat = m.predict(tmp)["yhat"].values
                mask = df["tcin"] == tcin
                df.loc[mask, "prophet_pred"] = yhat
        df["residual"] = df["units_sold"] - df["prophet_pred"]
        return df

    def fit_lstm_on_residuals(self, df_resid):
        for tcin, group in df_resid.groupby("tcin"):
            if len(group) < self.seq_len + 1:
                continue
            ds = group.copy().reset_index(drop=True)
            ds_features = ds[
                [
                    "residual", "day_of_week", "week_of_year", "month",
                    "region_id", "category_id", "new_launch_flag",
                    "promo", "state_holiday_id", "school_holiday_id"
                ]
            ].rename(columns={"residual": "units_sold"})
            ds_features["tcin"] = tcin
            ds_features["date"] = ds["date"]
            # Create dataset and dataloader
            dataset = SalesDataset(ds_features, seq_len=self.seq_len)
            loader = DataLoader(dataset, batch_size=self.lstm_params["batch_size"], shuffle=True)

            model = ResidualLSTM(
                input_size=self.lstm_params["input_size"],
                hidden_size=self.lstm_params["hidden_size"],
                num_layers=self.lstm_params["num_layers"],
                output_size=self.lstm_params["output_size"]
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lstm_params["lr"])
            criterion = nn.MSELoss()

            for epoch in range(self.lstm_params["epochs"]):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
            self.residual_models[tcin] = model

    def fit(self, df):
        # Fit Prophet models per tcin
        self.fit_prophet(df)
        # Compute residuals and add to DataFrame
        df_resid = self.compute_residuals(df)
        # Fit LSTM on residuals
        self.fit_lstm_on_residuals(df_resid)
        return df_resid

    def predict(self, df_future, hist_df):
        # Ensure hist_df has prophet_pred; if not, compute it.
        if "prophet_pred" not in hist_df.columns:
            hist_df = self.compute_residuals(hist_df)

        all_preds = []
        for tcin, group in df_future.groupby("tcin"):
            m = self.prophet_models.get(tcin)
            if m is None:
                continue
            tmp = group[["date"]].rename(columns={"date": "ds"})
            yhat_prophet = m.predict(tmp)["yhat"].values

            # Default residual correction is zero.
            yhat_resid = np.zeros_like(yhat_prophet)
            lstm_model = self.residual_models.get(tcin)
            if lstm_model is not None:
                hist_group = hist_df[hist_df["tcin"] == tcin].sort_values("date")
                if len(hist_group) >= self.seq_len:
                    last_window = hist_group.iloc[-self.seq_len:]
                    resid_hist = (last_window["units_sold"] - last_window["prophet_pred"]).values
                    features = last_window[
                        [
                            "day_of_week", "week_of_year", "month",
                            "region_id", "category_id", "new_launch_flag", "promo",
                            "state_holiday_id", "school_holiday_id"
                        ]
                    ].values
                    # Create input for LSTM: concatenate residual history with features.
                    lstm_input = np.concatenate([resid_hist.reshape(-1, 1), features], axis=1)
                    lstm_input = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0)
                    # Predict residual correction for one step
                    pred_resid = lstm_model(lstm_input).detach().numpy()
                    # Broadcast prediction to all future rows
                    yhat_resid = np.full_like(yhat_prophet, pred_resid[0])
            yhat_final = yhat_prophet + yhat_resid

            forecast_df = group.copy()
            forecast_df["yhat_prophet"] = yhat_prophet
            forecast_df["yhat_residual"] = yhat_resid
            forecast_df["yhat_final"] = yhat_final
            all_preds.append(forecast_df)
        return pd.concat(all_preds) if all_preds else pd.DataFrame()
