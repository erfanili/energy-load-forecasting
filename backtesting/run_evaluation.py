#backtesting/run_evaluation.py
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from visualization.plot_forecast import plot_forecast_timescales

from ingestion.ingest_raw import load_single_csv, load_all_regions
from ingestion.validate_schema import validate, clean

from feature_engineering.feature_pipeline import build_features, FeatureConfig
from models.classical.ets_model import train_and_forecast_ets
from backtesting.evaluate import evaluate_forecast

# --- Load and clean raw AEP data ---
df_raw = load_all_regions("data/raw")
df = clean(validate(df_raw))
# df = validate(df_raw)
df = df.iloc[:100000]
# breakpoint()

# --- Feature engineering ---
config = FeatureConfig(
    lags=[1, 24, 168],
    rolling_windows=[24, 168],
    use_boxcox=True,
    use_stl=False
)
# 1. Build features
df_feat = build_features(df, config)
horizon = 1200
# 2. Select training data (up to some cutoff)
region = "AEP"
cutoff = df_feat[df_feat["region"] == region]["timestamp"].max() - pd.Timedelta(hours=48)
df_train = df_feat[(df_feat["region"] == region) & (df_feat["timestamp"] <= cutoff)]
df_train = df_train.iloc[:9600]
# 3. Train + forecast
df_forecast = train_and_forecast_ets(
    df=df_train,
    region=region,
    seasonal_periods=24,
    horizon=horizon
)
# 4. Get actual future values to compare
df_eval = df[df["region"] == region]

df_actual = df_eval[
    (df_eval["timestamp"] > cutoff) &
    (df_eval["timestamp"] <= cutoff + pd.Timedelta(hours=horizon))
]

# 5. Merge (optional, for plotting)
df_merged = df_actual.merge(
    df_forecast,
    on=["timestamp", "region"],
    how="inner"
)

# 6. Proper evaluation (actual vs forecast)
metrics = evaluate_forecast(
    df_actual=df_actual,
    df_forecast=df_forecast
)

# 7. Proper plotting (merged frame)
plot_forecast_timescales(
    df_merged=df_merged,
    region=region,
    model_name="ETS_baseline",
    output_dir="plots"
)


# --- Print results ---
print("ETS Evaluation for AEP:")
for k, v in metrics.items():
    print(f"{k.upper()}: {v:.2f}")
# computer reference scale

df_aep = df[df["region"] == "AEP"]
mean_load = df_aep["load_mw"].mean()
peak_load = df_aep["load_mw"].max()

print("\n--- Load Scale for AEP ---")
print(f"Average Load: {mean_load:.2f} MW")
print(f"Peak Load:    {peak_load:.2f} MW")

# Print scaled errors
print("\n--- Scaled Error Ratios ---")
print(f"MAE / Avg Load:  {metrics['mae'] / mean_load * 100:.2f}%")
print(f"RMSE / Avg Load: {metrics['rmse'] / mean_load * 100:.2f}%")