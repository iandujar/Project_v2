# ============================================================
# STREAMLIT DASHBOARD
# WALMART MULTI-SERIES FORECASTING
# ============================================================

# Run with:
# streamlit run app.py

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Walmart Forecast Dashboard",
    layout="wide",
)

st.title("Walmart Multi-Series Forecasting Dashboard")

st.markdown("""
This dashboard compares:

- Single Exponential Smoothing
- ARIMA
- Random Forest

for Walmart department sales forecasting.
""")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df_all = pd.read_csv("train_2.csv")

    df_all['Date'] = pd.to_datetime(df_all['Date'])

    return df_all


df_all = load_data()
df= df_all[
    (df_all['Store'] == 1) &
    (df_all['Dept'].isin([1, 2, 3]))
]


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Dashboard Controls")


forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    min_value=4,
    max_value=26,
    value=12,
)

selected_depts = st.sidebar.multiselect(
    "Select Departments",
    sorted(df['Dept'].unique()),
    default=[1, 2, 3],
)

# Model selector (allow one or multiple)
model_options = ["ETS", "ARIMA", "Random Forest"]
selected_models = st.sidebar.multiselect(
    "Models to show",
    options=model_options,
    default=model_options,
)

if len(selected_models) == 0:
    st.sidebar.warning("Select at least one model to display")


target_series = st.sidebar.selectbox(
    "Select Target Series",
    options=["Total_Sales"] + [f"Dept_{d}" for d in selected_depts],
)


# ============================================================
# PREPARE DATA
# ============================================================

# Ensure store_id variable exists (default to 1)
store_id = 1

df_store = df[df['Store'] == store_id]

weekly_sales = (
    df_store.groupby(['Date', 'Dept'])['Weekly_Sales']
    .sum()
    .reset_index()
)

sales_pivot = (
    weekly_sales[weekly_sales['Dept'].isin(selected_depts)]
    .pivot(index='Date', columns='Dept', values='Weekly_Sales')
)

sales_pivot.columns = [f"Dept_{c}" for c in sales_pivot.columns]

sales_pivot = sales_pivot.fillna(0)

sales_pivot['Total_Sales'] = sales_pivot.sum(axis=1)


# ============================================================
# SHOW DATA
# ============================================================

st.subheader("Sales Data")

st.dataframe(sales_pivot.head())


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

train = sales_pivot.iloc[:-forecast_horizon]
test = sales_pivot.iloc[-forecast_horizon:]


# ============================================================
# ETS (Simple Exponential Smoothing) - safe fit
# ============================================================

from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def fit_ets_safe(series):
    try:
        model = SimpleExpSmoothing(series, initialization_method="estimated").fit()
    except TypeError:
        model = SimpleExpSmoothing(series).fit()
    return model

if len(train[target_series].dropna()) < 2:
    ets_forecast = pd.Series([np.nan] * forecast_horizon, index=test.index)
else:
    ets_model = fit_ets_safe(train[target_series])
    raw_ets = ets_model.forecast(forecast_horizon)
    ets_forecast = pd.Series(np.asarray(raw_ets).ravel(), index=test.index)


# ============================================================
# ARIMA
# ============================================================

arima_model = ARIMA(
    train[target_series],
    order=(2,1,1),
).fit()

arima_forecast = arima_model.forecast(forecast_horizon)


# ============================================================
# RANDOM FOREST
# ============================================================

def build_features(series, lags=[1,2,3,4,8,12,26,52]):

    out = pd.DataFrame({'y': series})

    # -------------------
    # LAGS
    # -------------------
    for L in lags:
        out[f'lag_{L}'] = series.shift(L)

    # -------------------
    # CALENDAR
    # -------------------
    out['month'] = series.index.month
    out['weekofyear'] = series.index.isocalendar().week.astype(int)
    out['quarter'] = series.index.quarter

    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)

    out['week_sin'] = np.sin(2 * np.pi * out['weekofyear'] / 52)
    out['week_cos'] = np.cos(2 * np.pi * out['weekofyear'] / 52)

    # -------------------
    # HOLIDAYS (simple proxies)
    # -------------------
    out['is_christmas'] = (series.index.month == 12).astype(int)
    out['is_november'] = (series.index.month == 11).astype(int)

    # -------------------
    # ROLLING FEATURES
    # -------------------
    for w in [4, 8, 12, 26]:
        out[f'roll_mean_{w}'] = series.shift(1).rolling(w).mean()
        out[f'roll_std_{w}'] = series.shift(1).rolling(w).std()

    # -------------------
    # TREND
    # -------------------
    t = np.arange(len(series))
    out['smooth_time'] = t / (len(series) - 1)
    out['smooth_time_sq'] = out['smooth_time'] ** 2

    out['diff_1'] = series.diff()
    out['diff_2'] = series.diff().diff()

    # -------------------
    # MOMENTUM
    # -------------------
    out['momentum_4'] = series / series.shift(4)
    out['momentum_12'] = series / series.shift(12)

    return out.dropna()


feature_df = build_features(train[target_series])

X_train = feature_df.drop(columns='y')
y_train = feature_df['y']

rf_model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)


history = train[target_series].copy()

# Forecast using RF: generate features from rolling history so feature set matches training
rf_preds = []
for i in range(forecast_horizon):
    feat = build_features(history)
    if feat.empty:
        raise ValueError('Not enough history to build features for RF forecasting')

    last_row = feat.drop(columns='y').iloc[-1:]
    last_row = last_row.reindex(columns=X_train.columns, fill_value=0)

    pred = rf_model.predict(last_row)[0]
    rf_preds.append(pred)

    next_index = history.index[-1] + pd.Timedelta(weeks=1)
    history.loc[next_index] = pred

rf_predictions = pd.Series(rf_preds, index=test.index)


# ============================================================
# FORECAST PLOT (show selected models only)
# ============================================================

st.subheader("Forecast Comparison")

fig, ax = plt.subplots(figsize=(15,6))

ax.plot(train.index, train[target_series], label='Train', color='#444444')
ax.plot(test.index, test[target_series], label='Test', color='#222222')

if 'ETS' in selected_models:
    ax.plot(test.index, ets_forecast, label='ETS Forecast')
if 'ARIMA' in selected_models:
    ax.plot(test.index, arima_forecast, label='ARIMA Forecast')
if 'Random Forest' in selected_models:
    ax.plot(test.index, rf_predictions, label='RF Forecast')

ax.set_title(f"Forecast Comparison - {target_series}")
ax.legend()
st.pyplot(fig)


# ============================================================
# ACCURACY METRICS
# ============================================================

st.subheader("Model Accuracy")

actual = test[target_series]

# helper for MAPE (ignore zero actuals)
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# helper for wMAPE
def wmape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom * 100

results = pd.DataFrame({
    'Model': ['ETS', 'ARIMA', 'Random Forest'],
    'MAE': [
        mean_absolute_error(actual, ets_forecast),
        mean_absolute_error(actual, arima_forecast),
        mean_absolute_error(actual, rf_predictions),
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(actual, ets_forecast)),
        np.sqrt(mean_squared_error(actual, arima_forecast)),
        np.sqrt(mean_squared_error(actual, rf_predictions)),
    ],
    'MAPE': [
        mape(actual, ets_forecast),
        mape(actual, arima_forecast),
        mape(actual, rf_predictions),
    ],
    'wMAPE': [
        wmape(actual, ets_forecast),
        wmape(actual, arima_forecast),
        wmape(actual, rf_predictions),
    ]
})

st.dataframe(results)


# ============================================================
# FORECAST TABLE (kept for downloads but not displayed)
# ============================================================

forecast_table = pd.DataFrame({
    'Actual': actual.values,
    'ETS': ets_forecast.values,
    'ARIMA': arima_forecast.values,
    'RandomForest': rf_predictions.values,
}, index=test.index)


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

st.subheader("Residual Diagnostics")

# ARIMA residuals (show histogram only if selected)
if 'ARIMA' in selected_models:
    residuals = arima_model.resid

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(residuals.dropna(), bins=25)
    ax.set_title("ARIMA Residual Distribution")
    st.pyplot(fig)

# ETS residuals (histogram only)
if 'ETS' in selected_models:
    try:
        residuals_ets = actual - ets_forecast
    except Exception:
        residuals_ets = pd.Series([np.nan]*len(actual), index=actual.index)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(residuals_ets.dropna(), bins=25)
    ax.set_title("ETS Resid Distribution")
    st.pyplot(fig)

# Random Forest residuals (histogram only)
if 'Random Forest' in selected_models:
    try:
        residuals_rf = actual - rf_predictions
    except Exception:
        residuals_rf = pd.Series([np.nan]*len(actual), index=actual.index)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(residuals_rf.dropna(), bins=25)
    ax.set_title("Random Forest Resid Distribution")
    st.pyplot(fig)


# ============================================================
# FEATURE IMPORTANCE (table only; plot removed)
# ============================================================

st.subheader("Random Forest Feature Importance")

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.dataframe(importance_df)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.subheader("Download Results")

csv = forecast_table.to_csv().encode('utf-8')

st.download_button(
    label="Download Forecast Results",
    data=csv,
    file_name='forecast_results.csv',
    mime='text/csv',
)

