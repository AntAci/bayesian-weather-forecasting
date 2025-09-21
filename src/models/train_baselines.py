import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAIN_END, VAL_END = 2012, 2018  

def year_split(df: pd.DataFrame):
    y = df["DATE"].dt.year
    tr = df[y <= TRAIN_END]
    va = df[(y > TRAIN_END) & (y <= VAL_END)]
    te = df[y > VAL_END]
    return tr, va, te

def climatology_predict(train_df: pd.DataFrame, target_col="y_tmax_next"):
    tmp = train_df.copy()
    doy_next = (tmp["DATE"] + pd.Timedelta(days=1)).dt.dayofyear
    tmp["DOY_NEXT"] = doy_next
    doy_mean = tmp.groupby("DOY_NEXT")[target_col].mean()

    # fill any missing bins (e.g., leap day) by simple neighbors
    doy_mean = doy_mean.reindex(range(1, 367)).interpolate().bfill().ffill()

    def pred_fun(df):
        d = (df["DATE"] + pd.Timedelta(days=1)).dt.dayofyear
        return d.map(doy_mean).values
    return pred_fun

def persistence_predict():
    def pred_fun(df):
        # predict next-day TMAX using today's TMAX ( = y_{t-1})
        return df["TMAX_lag1"].values 
    return pred_fun

def ridge_predictor(train_df, val_df, feature_cols, target_col="y_tmax_next"):
    # Tune Ridge α on validation set
    alphas = [0.0, 0.1, 1.0, 3.0, 10.0]
    best = None; best_alpha = None
    Xtr, ytr = train_df[feature_cols].values, train_df[target_col].values
    for a in alphas:
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a, random_state=0))])
        model.fit(Xtr, ytr)
        if len(val_df) == 0:
            score = model.score(Xtr, ytr)  # fallback 
        else:
            score = -mean_squared_error(val_df[target_col].values, model.predict(val_df[feature_cols].values))
        if (best is None) or (score > best):
            best, best_alpha = score, a
            best_model = model
    def pred_fun(df):
        return best_model.predict(df[feature_cols].values)
    return pred_fun, best_alpha

def evaluate_split(df, pred_fun, target_col="y_tmax_next", split_name="test"):
    y_true = df[target_col].values
    y_pred = pred_fun(df)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae= mean_absolute_error(y_true, y_pred)
    return {"split": split_name, "RMSE": rmse, "MAE": mae, "n": len(df)}

def save_preds(df, pred_fun, split_name, out_dir, station, model_name="ridge"):
    """Save predictions for plots and calibration."""
    df_pred = df[["DATE","y_tmax_next"]].copy()
    df_pred["y_pred"] = pred_fun(df)
    df_pred = df_pred.rename(columns={"y_tmax_next": "y_true"})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(Path(out_dir)/f"{station}_{model_name}_{split_name}.csv", index=False)

def run_for_station(features_path: str, out_dir="results/baselines"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = Path("results/preds"); preds_dir.mkdir(parents=True, exist_ok=True)
    station = Path(features_path).stem  # Define station early
    # Handle both parquet and CSV files
    if features_path.endswith('.parquet'):
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path, parse_dates=['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)

    # Build feature list (lags + rolls + seasonality)
    feature_cols = ["sin_doy","cos_doy","TMAX_roll7_mean","TMAX_roll7_min","TMAX_roll7_max","PRCP_roll7_mean"]
    for L in range(1, 15):
        feature_cols += [f"TMAX_lag{L}", f"TMIN_lag{L}", f"PRCP_lag{L}"]

    # Time splits
    tr, va, te = year_split(df)

    # Climatology 
    climo = climatology_predict(tr)
    m_climo = []
    for part, name in [(tr,"train"), (va,"val"), (te,"test")]:
        if len(part) > 0:
            m_climo.append(evaluate_split(part, climo, split_name=name))
            # Save predictions for plotting
            save_preds(part, climo, name, preds_dir, station, "climatology")

    # Persistence 
    persist = persistence_predict()
    m_persist = []
    for part, name in [(tr,"train"), (va,"val"), (te,"test")]:
        if len(part) > 0:
            m_persist.append(evaluate_split(part, persist, split_name=name))
            # Save predictions for plotting
            save_preds(part, persist, name, preds_dir, station, "persistence")

    # Ridge
    ridge, best_alpha = ridge_predictor(tr, va, feature_cols)
    print(f"[{station}] Best Ridge alpha: {best_alpha}")
    m_ridge = []
    for part, name in [(tr,"train"), (va,"val"), (te,"test")]:
        if len(part) > 0:
            m_ridge.append(evaluate_split(part, ridge, split_name=name))
            # Save predictions for plotting
            save_preds(part, ridge, name, preds_dir, station, "ridge")

    # Collect + save
    metrics = pd.concat([
        pd.DataFrame(m_climo).assign(model="climatology"),
        pd.DataFrame(m_persist).assign(model="persistence"),
        pd.DataFrame(m_ridge).assign(model="ridge"),
    ], ignore_index=True)
    station = Path(features_path).stem
    out_csv = out_dir / f"{station}_metrics.csv"
    metrics.to_csv(out_csv, index=False)
    print(f"[{station}] saved metrics → {out_csv}")
    print(metrics.pivot(index="model", columns="split", values="RMSE").round(3))
    
    # residual check 
    if len(te) > 0:
        te_residuals = te.copy()
        te_residuals['y_pred'] = ridge(te)
        te_residuals['residual'] = te_residuals['y_tmax_next'] - te_residuals['y_pred']
        te_residuals['month'] = te_residuals['DATE'].dt.month
        monthly_mae = te_residuals.groupby('month')['residual'].apply(lambda x: np.abs(x).mean())
        print(f"[{station}] Monthly MAE on test (Ridge): {monthly_mae.round(2).to_dict()}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to station features parquet")
    ap.add_argument("--out", default="results/baselines", help="Output directory for metrics")
    args = ap.parse_args()
    run_for_station(args.features, args.out)
