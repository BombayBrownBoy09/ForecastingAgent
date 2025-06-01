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
        # Remove any unsupported Prophet kwargs (verbose, mcmc_samples, etc.)
        self.prophet_params = prophet_params or {}
        # You can still pass valid Prophet args here (e.g. seasonality_mode), but do not include `verbose` or `mcmc_samples`.

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
            m = Prophet(**self.prophet_params)  # no verbose/mcmc params
            m.fit(prophet_df)
            self.prophet_models[tcin] = m

    def compute_residuals(self, df):
        df = df.copy()
        df["prophet_pred"] = 0.0
        for tcin, group in df.groupby("tcin"):
            m = self.prophet_models[tcin]
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
            ds_features["region_id"] = ds["region_id"]

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
                total_loss = 0.0
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            self.residual_models[tcin] = model

    def fit(self, df):
        # 1) Fit Prophet per store
        self.fit_prophet(df)
        # 2) Compute residuals
        df_resid = self.compute_residuals(df)
        # 3) Fit LSTM on residuals
        self.fit_lstm_on_residuals(df_resid)
        return df_resid  # return with prophet_pred included

    def predict(self, df_future, hist_df):
        """
        df_future: same columns as agg_df minus 'units_sold'
        hist_df: original historical agg_df with 'prophet_pred' column
        """
        all_preds = []
        for tcin, group in df_future.groupby("tcin"):
            # 1) Prophet baseline
            m = self.prophet_models.get(tcin, None)
            if m is None:
                continue
            tmp = group[["date"]].rename(columns={"date": "ds"})
            yhat_base = m.predict(tmp)["yhat"].values

            # 2) LSTM residual if available
            yhat_resid = np.zeros_like(yhat_base)
            lstm_model = self.residual_models.get(tcin, None)
            if lstm_model is not None:
                hist_group = hist_df[hist_df["tcin"] == tcin].sort_values("date")
                if len(hist_group) >= self.seq_len:
                    last_window = hist_group.iloc[-self.seq_len :]
                    resid_hist = (last_window["units_sold"] - last_window["prophet_pred"]).values
                    features = last_window[
                        [
                            "day_of_week", "week_of_year", "month",
                            "region_id", "category_id", "new_launch_flag",
                            "promo", "state_holiday_id", "school_holiday_id"
                        ]
                    ].values
                    stacked = np.concatenate([resid_hist.reshape(-1, 1), features], axis=1)  # (seq_len, 10)
                    inp = torch.tensor(stacked.reshape(1, self.seq_len, 10), dtype=torch.float32)
                    with torch.no_grad():
                        yhat_resid_val = lstm_model(inp).numpy()
                    yhat_resid[0] = yhat_resid_val  # only first step

            yhat_final = yhat_base + yhat_resid
            out = group.copy()
            out["yhat_prophet"] = yhat_base
            out["yhat_residual"] = yhat_resid
            out["yhat_final"] = yhat_final
            all_preds.append(out[["date", "tcin", "yhat_prophet", "yhat_residual", "yhat_final"]])

        return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
