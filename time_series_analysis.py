"""
CIMIS TEMPERATURE FORECAST - 6 MONTH TRAIN / 2 DAY PREDICT
Optimized for MacBook Air M2 performance
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# 1. DATA LOADING
def load_and_prepare():
    """Load and preprocess data with 6-month focus"""
    df = pd.read_csv('daily.csv', parse_dates=['Date'])
    
    # Average temperatures from multiple stations per day
    daily_temp = df.groupby('Date')['Avg Air Temp (C)'].mean().reset_index()
    
    # Filter last 6 months + 2 days
    cutoff_date = daily_temp['Date'].max() - pd.DateOffset(months=6)
    return daily_temp[daily_temp['Date'] >= cutoff_date].set_index('Date')

# 2. FORECASTING
def forecast_2days(temp_series):
    """6 months training → 2 days prediction"""
    # Split data (last 2 days for testing)
    train = temp_series.iloc[:-2]
    test = temp_series.iloc[-2:]
    
    # ARIMA model (simplified for short-term forecast)
    model = ARIMA(train, order=(3,1,1))  # Best for 6-month daily data
    model_fit = model.fit()
    
    # Generate and evaluate forecast
    forecast = model_fit.forecast(steps=2)
    mae = mean_absolute_error(test, forecast)
    
    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(train[-30:], label='Training (last 30 days)')  # Show recent trend
    plt.plot(test, 'bo-', label='Actual')
    plt.plot(test.index, forecast, 'ro-', label='Forecast')
    plt.title(f'2-Day Temperature Forecast\nMAE: {mae:.1f}°C')
    plt.legend()
    plt.grid(True)
    plt.savefig('2day_forecast.png', dpi=120, bbox_inches='tight')
    
    print("\n=== FORECAST RESULTS ===")
    print(f"Training Period: {train.index[0].date()} to {train.index[-1].date()}")
    print(f"Predicted Dates: {test.index[0].date()} to {test.index[-1].date()}")
    for date, actual, pred in zip(test.index, test, forecast):
        print(f"{date.date()}: {actual:.1f}°C (Actual) → {pred:.1f}°C (Predicted)")

# MAIN EXECUTION
if __name__ == "__main__":
    print("\n=== 6-MONTH TEMPERATURE FORECAST ===")
    temp_data = load_and_prepare()['Avg Air Temp (C)'].dropna()
    
    if len(temp_data) >= 180:  # ~6 months of data
        forecast_2days(temp_data)
    else:
        print(f"Need 180+ days (current: {len(temp_data)})")
    
    print("\n=== VISUALIZATION SAVED TO 2day_forecast.png ===")