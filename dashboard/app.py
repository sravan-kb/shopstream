import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "revenue_forecast_rf.pkl"

# Debug (remove later)
st.write(f"Model Path: {MODEL_PATH}")
st.write(f"Exists: {MODEL_PATH.exists()}")

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Revenue Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Revenue Forecasting")

st.write("Predict daily revenue using the trained Random Forest model.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", value=2027)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)

    day_of_week = st.selectbox(
        "Day of Week",
        [0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ][x]
    )

with col2:

    lag1 = st.number_input("Yesterday Revenue", value=8200.0)

    lag7 = st.number_input("Revenue 7 Days Ago", value=7900.0)

    lag30 = st.number_input("Revenue 30 Days Ago", value=7600.0)

    rolling7 = st.number_input("7-Day Average Revenue", value=8100.0)

    rolling30 = st.number_input("30-Day Average Revenue", value=7900.0)

    rolling_std7 = st.number_input("7-Day Std Dev", value=900.0)

quarter = ((month - 1) // 3) + 1
is_weekend = 1 if day_of_week >= 5 else 0

if st.button("Predict Revenue", use_container_width=True):

    sample = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "Day_of_Week": [day_of_week],
        "Quarter": [quarter],
        "Is_Weekend": [is_weekend],
        "Lag_1": [lag1],
        "Lag_7": [lag7],
        "Lag_30": [lag30],
        "Rolling_Mean_7": [rolling7],
        "Rolling_Mean_30": [rolling30],
        "Rolling_Std_7": [rolling_std7]
    })

    prediction = model.predict(sample)[0]

    st.success(f"Predicted Revenue: ₹ {prediction:,.2f}")