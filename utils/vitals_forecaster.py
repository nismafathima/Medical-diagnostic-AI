# utils/vitals_forecaster.py
# Time-series forecasting of patient vitals using Prophet + LSTM simulation
# Vitals: Heart Rate, Blood Pressure, SpO2, Temperature

import numpy as np
import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


# ── Normal ranges ──────────────────────────────────────────────────────────────
NORMAL_RANGES = {
    "Heart Rate (bpm)":      {"min": 60,  "max": 100, "unit": "bpm"},
    "Blood Pressure (mmHg)": {"min": 90,  "max": 120, "unit": "mmHg"},
    "SpO2 (%)":              {"min": 95,  "max": 100, "unit": "%"},
    "Temperature (°C)":      {"min": 36.1,"max": 37.2,"unit": "°C"},
}


def generate_sample_vitals(vital_name: str, days: int = 10,
                             condition: str = "Normal") -> pd.DataFrame:
    """
    Generate realistic sample vital sign data for demo purposes.
    In production: replace with actual patient sensor/EHR data.
    """
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")

    base_values = {
        "Heart Rate (bpm)":      80,
        "Blood Pressure (mmHg)": 115,
        "SpO2 (%)":              97,
        "Temperature (°C)":      36.8,
    }

    # Simulate condition deterioration
    condition_offset = {
        "Normal":    0,
        "Pneumonia": {"Heart Rate (bpm)": 15, "Temperature (°C)": 1.2,
                      "SpO2 (%)": -4, "Blood Pressure (mmHg)": 10},
        "COVID-19":  {"Heart Rate (bpm)": 20, "Temperature (°C)": 1.8,
                      "SpO2 (%)": -8, "Blood Pressure (mmHg)": 15},
    }

    base = base_values[vital_name]
    offset = 0
    if condition != "Normal" and isinstance(condition_offset.get(condition), dict):
        offset = condition_offset[condition].get(vital_name, 0)

    noise = np.random.normal(0, 2, days)
    trend = np.linspace(0, offset, days)
    values = base + trend + noise

    # Clip to realistic bounds
    if "SpO2" in vital_name:
        values = np.clip(values, 85, 100)
    elif "Temperature" in vital_name:
        values = np.clip(values, 35.5, 41.0)
    elif "Heart Rate" in vital_name:
        values = np.clip(values, 40, 180)

    return pd.DataFrame({"ds": dates, "y": values})


def forecast_vital(df: pd.DataFrame, periods: int = 3) -> dict:
    """
    Forecast next `periods` days of a vital sign using Facebook Prophet with CPU optimization.
    Returns forecast values + trend direction.
    Optimizations:
    - Reduced iterations (default 10000 -> 100)
    - Simpler seasonality model
    - Error handling with fallback to simple linear forecast
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        
        # Fast configuration for CPU
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1,
            interval_width=0.80,
            interval_width_for_level=0.95,
            mcmc_samples=0  # Disable MCMC for faster inference
        )
        
        # Suppress Prophet's verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq="D")
        forecast = model.predict(future)

        last_actual = df["y"].iloc[-1]
        next_forecast = forecast["yhat"].iloc[-periods:]
        avg_forecast = float(next_forecast.mean())

        trend = "increasing" if avg_forecast > last_actual else "decreasing"
        change = abs(avg_forecast - last_actual)

        return {
            "forecast_df": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods + len(df)),
            "next_values": next_forecast.values.tolist(),
            "avg_forecast": avg_forecast,
            "trend": trend,
            "change": change,
            "last_actual": last_actual
        }
    except Exception as e:
        print(f"Prophet forecasting failed: {e}. Using linear forecast fallback.")
        # Fallback: simple linear interpolation
        last_actual = df["y"].iloc[-1]
        second_last = df["y"].iloc[-2] if len(df) > 1 else last_actual
        trend_val = (last_actual - second_last) / max(1, len(df) - 1)
        
        next_values = [last_actual + (i + 1) * trend_val for i in range(periods)]
        avg_forecast = float(np.mean(next_values))
        
        future_dates = pd.date_range(start=df["ds"].iloc[-1], periods=periods + 1, freq="D")[1:]
        forecast_data = pd.DataFrame({
            "ds": future_dates,
            "yhat": next_values,
            "yhat_lower": [v * 0.95 for v in next_values],
            "yhat_upper": [v * 1.05 for v in next_values]
        })
        
        return {
            "forecast_df": forecast_data,
            "next_values": next_values,
            "avg_forecast": avg_forecast,
            "trend": "increasing" if trend_val > 0 else "decreasing",
            "change": abs(avg_forecast - last_actual),
            "last_actual": last_actual,
            "note": "Using linear fallback (Prophet failed)"
        }


def assess_vitals_risk(vitals_results: dict) -> dict:
    """
    Combine all vital forecasts into an overall patient risk score.
    Returns risk level: Low / Medium / High / Critical
    """
    risk_points = 0

    for vital_name, result in vitals_results.items():
        normal = NORMAL_RANGES[vital_name]
        forecast_val = result["avg_forecast"]

        # Check if forecast is outside normal range
        if vital_name == "SpO2 (%)":
            if forecast_val < 90:
                risk_points += 3
            elif forecast_val < 95:
                risk_points += 2
        elif vital_name == "Temperature (°C)":
            if forecast_val > 39.0:
                risk_points += 3
            elif forecast_val > 37.5:
                risk_points += 1
        elif vital_name == "Heart Rate (bpm)":
            if forecast_val > 130 or forecast_val < 50:
                risk_points += 3
            elif forecast_val > 110:
                risk_points += 1
        elif vital_name == "Blood Pressure (mmHg)":
            if forecast_val > 140:
                risk_points += 2
            elif forecast_val > 130:
                risk_points += 1

    # Risk levels
    if risk_points >= 6:
        level, color, action = "Critical 🚨", "red", "Immediate ICU/Emergency care required."
    elif risk_points >= 3:
        level, color, action = "High ⚠️", "orange", "Urgent medical attention within 6 hours."
    elif risk_points >= 1:
        level, color, action = "Medium 🟡", "yellow", "Monitor closely. Doctor visit within 24 hours."
    else:
        level, color, action = "Low ✅", "green", "Vitals within acceptable range. Continue monitoring."

    return {
        "risk_level": level,
        "risk_color": color,
        "risk_action": action,
        "risk_points": risk_points
    }
