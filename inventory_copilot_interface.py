import os
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import streamlit as st

from datetime import timedelta
import openai  # for the “Ask Copilot” feature

# ─────────────────────────────────────────────────────────────────────────────
# 1) IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from forecasting_agent import ForecastingAgent  
from forecast_agent_data_rossmann import load_and_prep_data

# ─────────────────────────────────────────────────────────────────────────────
# 2) COPILOT QUERY HELPER (runs at runtime)
# ─────────────────────────────────────────────────────────────────────────────

def ask_copilot(question: str, context: str) -> str:
    """
    Send a natural‐language question + context to OpenAI’s Chat Completion v1 API
    (compatible with openai‐python >= 1.0.0) and return the answer.
    You must have OPENAI_API_KEY set in your environment.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY", "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")
    if not openai.api_key:
        return "🔴 ERROR: OPENAI_API_KEY not set."

    system_message = (
        "You are an AI assistant for inventory analysts. "
        "Given sales history and forecast context, provide actionable inventory guidance."
    )
    user_message = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    try:
        # New v1 interface:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.5,
            max_tokens=400
        )
        # `response.choices[0].message.content` still holds the generated text
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"🔴 ERROR calling OpenAI: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 3) STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Inventory Copilot Dashboard", layout="wide")
st.title("🔮 Inventory Decision Support")

# Sidebar: upload + TCIN list + question
with st.sidebar:
    st.header("1. Upload or Use Default Data")
    uploaded = st.file_uploader("Upload your sales CSV", type=["csv"])
    st.markdown("---")

    st.header("2. Ask Copilot")
    question = st.text_area(
        "Type your inventory question (e.g. “When should I reorder this product?”)",
        height=100
    )
    st.button("Get Recommendation", key="ask_button")

# 3.1) Load & prepare data (if user uploaded, use that; else default to train.csv)
if uploaded is not None:
    tmp = pd.read_csv(uploaded, parse_dates=["Date"])
    # Assume it has same schema as Rossmann’s train.csv
    tmp.to_csv("temp_upload.csv", index=False)
    os.rename("temp_upload.csv", "train.csv")
agg_df = load_and_prep_data()

# 3.2) Let user pick a TCIN
unique_tcins = sorted(agg_df["tcin"].unique())
selected_tcin = st.selectbox("Pick a TCIN (store)", unique_tcins)

# Filter historical data for that TCIN
prod_hist = agg_df[agg_df["tcin"] == selected_tcin].sort_values("date")

# 3.3) Show Historical Sales
st.write(f"### Historical Sales for TCIN {selected_tcin}")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(prod_hist["date"], prod_hist["units_sold"], marker="o", linestyle="-", label="Actual Sales")
ax1.set_xlabel("Date")
ax1.set_ylabel("Units Sold")
ax1.set_title(f"TCIN {selected_tcin} Sales History")
ax1.grid(True)
st.pyplot(fig1)

# 3.4) Train & Forecast button
if st.button("🔄 Train & Forecast"):
    # Split last 4 weeks as validation; train on everything before that
    cutoff_date = prod_hist["date"].max() - timedelta(weeks=4)
    train_df = prod_hist[prod_hist["date"] <= cutoff_date].copy()
    val_df   = prod_hist[prod_hist["date"] >  cutoff_date].copy()

    st.info("Fitting ForecastingAgent (Prophet + LSTM)… this may take a moment.")
    agent = ForecastingAgent(
        prophet_params={"weekly_seasonality": True, "yearly_seasonality": True},
        lstm_params={"input_size": 10, "hidden_size": 32, "num_layers": 1, "output_size": 1, "lr":1e-3, "epochs":3, "batch_size":64},
        seq_len=8
    )
    train_with_pred = agent.fit(train_df)  # returns train_df with "prophet_pred"
    st.success("✅ Model training complete!")

    # 3.5) Prepare “future” features = next 4 weeks
    last_row = prod_hist.iloc[-1]
    future_dates = pd.date_range(start=prod_hist["date"].max() + pd.Timedelta(days=1), periods=28, freq="D")

    future_df = pd.DataFrame({
        "date": future_dates,
        "tcin": selected_tcin,
        "day_of_week": future_dates.dayofweek,
        "week_of_year": future_dates.isocalendar().week.astype(int),
        "month": future_dates.month,
        "region_id": last_row["region_id"],
        "category_id": last_row["category_id"],
        "new_launch_flag": 0,
        "promo": last_row["promo"],
        "state_holiday_id": last_row["state_holiday_id"],
        "school_holiday_id": last_row["school_holiday_id"]
    })

    st.info("Running inference on next 4 weeks…")
    preds = agent.predict(future_df, hist_df=train_with_pred)
    st.success("✅ Forecasting complete!")

    # 3.6) Merge historical + forecasted for plotting
    merged = pd.concat([prod_hist, preds], ignore_index=True, sort=False)
    merged.sort_values("date", inplace=True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    # Plot actual history
    hist_mask = merged["date"] <= prod_hist["date"].max()
    ax2.plot(
        merged.loc[hist_mask, "date"],
        merged.loc[hist_mask, "units_sold"],
        label="Actual Sales", marker="o", linestyle="-", color="blue"
    )
    # Plot forecast
    fut_mask = merged["date"] > prod_hist["date"].max()
    ax2.plot(
        merged.loc[fut_mask, "date"],
        merged.loc[fut_mask, "yhat_final"],
        label="Forecasted Sales", marker="x", linestyle="--", color="orange"
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Units Sold")
    ax2.set_title(f"TCIN {selected_tcin}: Actual + 4‐Week Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # 3.7) Show a small forecast table
    st.write("#### Next 4 Weeks Forecast")
    st.dataframe(
        merged.loc[fut_mask, ["date", "yhat_prophet", "yhat_residual", "yhat_final"]]
        .rename(columns={
            "yhat_prophet": "Prophet Baseline",
            "yhat_residual": "LSTM Residual",
            "yhat_final": "Final Forecast"
        })
    )

    # 3.8) Save forecast CSV and offer download
    csv_filename = f"forecast_tcin_{selected_tcin}.csv"
    merged.loc[fut_mask, ["date", "yhat_final"]].to_csv(csv_filename, index=False)
    st.success(f"Saved forecast to `{csv_filename}`")

    # Provide a download link via base64
    csv_bytes = open(csv_filename, "rb").read()
    b64 = base64.b64encode(csv_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_filename}">⬇️ Download {csv_filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

    # 3.9) If the user typed a question, ask Copilot now
    if question.strip():
        recent_actual = prod_hist.tail(7)[["date", "units_sold"]].to_dict("records")
        recent_forecast = merged.loc[fut_mask].head(7)[["date", "yhat_final"]].to_dict("records")
        context = (
            f"Recent 7 days actual: {recent_actual}\n"
            f"Next 7 days forecast: {recent_forecast}"
        )
        with st.spinner("🔮 Getting Copilot recommendation…"):
            recommendation = ask_copilot(question, context)
        st.write("### Copilot Recommendation")
        st.write(recommendation)

# If user clicks “Get Recommendation” without training
elif question.strip():
    st.warning("❗ Please click “🔄 Train & Forecast” before asking Copilot.")

else:
    st.info("Press “🔄 Train & Forecast” to run the forecasting pipeline.")

# ─────────────────────────────────────────────────────────────────────────────
# 4) FOOTER: Data Source + Model Info
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.write("**Data Source:** Kaggle Rossmann `train.csv` (merged with `store.csv`).")
st.write("**Models:** Prophet (weekly & yearly seasonality) + LSTM residual.")
st.write("**Copilot:** OpenAI GPT‐3.5 under the hood for natural‐language Q&A.")
