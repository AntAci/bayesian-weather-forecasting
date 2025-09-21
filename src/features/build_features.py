import pandas as pd
import numpy as np
from pathlib import Path

MAX_LAG = 14

def build_features_for_station(csv_path: str, out_dir="data/features"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # Basic sanity
    for col in ["TMAX (°C)","TMIN (°C)","PRCP (mm)"]:
        assert col in df.columns, f"Missing column {col}"
    
    # Numeric coercion safety
    for c in ["TMAX (°C)","TMIN (°C)","PRCP (mm)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Target: next-day TMAX
    df["y_tmax_next"] = df["TMAX (°C)"].shift(-1)

    # Lags 
    for L in range(1, MAX_LAG + 1):
        df[f"TMAX_lag{L}"] = df["TMAX (°C)"].shift(L)
        df[f"TMIN_lag{L}"] = df["TMIN (°C)"].shift(L)
        df[f"PRCP_lag{L}"] = df["PRCP (mm)"].shift(L)

    # Rolling stats (only past info)
    df["TMAX_roll7_mean"] = df["TMAX (°C)"].shift(1).rolling(7, min_periods=5).mean()
    df["TMAX_roll7_min"]  = df["TMAX (°C)"].shift(1).rolling(7, min_periods=5).min()
    df["TMAX_roll7_max"]  = df["TMAX (°C)"].shift(1).rolling(7, min_periods=5).max()
    df["PRCP_roll7_mean"] = df["PRCP (mm)"].shift(1).rolling(7, min_periods=5).mean()

    # Seasonality
    doy = df["DATE"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*doy/365.25)
    df["cos_doy"] = np.cos(2*np.pi*doy/365.25)

    # build feature list once 
    feat_cols = ["sin_doy","cos_doy","TMAX_roll7_mean","TMAX_roll7_min",
                 "TMAX_roll7_max","PRCP_roll7_mean"]
    for L in range(1, MAX_LAG + 1):
        feat_cols += [f"TMAX_lag{L}", f"TMIN_lag{L}", f"PRCP_lag{L}"]

    # Keep geo for reference 
    keep_cols = ["DATE","LATITUDE","LONGITUDE","ELEVATION","y_tmax_next",
                 "TMAX (°C)","TMIN (°C)","PRCP (mm)","is_trace"] + feat_cols
    df = df[keep_cols]

    #  require target + ALL features 
    df = df.dropna(subset=["y_tmax_next"] + feat_cols).reset_index(drop=True)

    # Save
    station_name = Path(csv_path).stem 
    out_path = out_dir / f"{station_name}.parquet"
    try:
        df.to_parquet(out_path, index=False)
        print(f"Saved features to{out_path} ({len(df):,} rows)")
    except ImportError:
        # Fallback to CSV
        out_path = out_dir / f"{station_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved features to {out_path} ({len(df):,} rows) [CSV fallback]")
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to per-station CSV (e.g., data/processed/stations/oxford.csv)")
    ap.add_argument("--out", default="data/features", help="Output folder for features")
    args = ap.parse_args()
    build_features_for_station(args.csv, args.out)
