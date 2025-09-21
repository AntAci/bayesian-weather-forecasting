# src/viz/plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_time_window(pred_csvs, out_path, start=None, end=None, title=None):
    """
    pred_csvs: dict {model_name: path_to_csv} where CSV has DATE,y_true,y_pred
    """
    dfs = {}
    for name, p in pred_csvs.items():
        df = pd.read_csv(p, parse_dates=["DATE"]).sort_values("DATE")
        if start: df = df[df["DATE"] >= pd.to_datetime(start)]
        if end:   df = df[df["DATE"] <= pd.to_datetime(end)]
        dfs[name] = df

    # use any df to get true series (assume aligned)
    base = next(iter(dfs.values()))
    plt.figure(figsize=(10,4))
    plt.plot(base["DATE"], base["y_true"], label="Truth", linewidth=1.5, color='black')
    for name, df in dfs.items():
        plt.plot(df["DATE"], df["y_pred"], label=name, linewidth=1, alpha=0.8)
    plt.xlabel("Date"); plt.ylabel("TMAX next day (°C)")
    if title: plt.title(title)
    plt.legend(); plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()

def plot_residuals_vs_doy(pred_csv, out_path, bins=12, title=None):
    df = pd.read_csv(pred_csv, parse_dates=["DATE"]).sort_values("DATE")
    df["resid"] = df["y_pred"] - df["y_true"]
    df["DOY"]   = df["DATE"].dt.dayofyear
    # bin by month-ish
    edges = np.linspace(1, 366, bins+1)
    idx = np.digitize(df["DOY"], edges) - 1
    df["bin"] = idx
    means = df.groupby("bin")["resid"].mean()
    centers = 0.5*(edges[:-1] + edges[1:])
    plt.figure(figsize=(8,4))
    # Ensure we only plot the bins that have data
    valid_bins = means.index[means.index < len(centers)]
    plt.bar(centers[valid_bins], means[valid_bins].values, width=(edges[1]-edges[0])*0.9)
    plt.axhline(0, linewidth=1, color='red', alpha=0.7)
    plt.xlabel("Day of year (binned)"); plt.ylabel("Mean residual (°C)")
    if title: plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()

def plot_scatter_with_rmse(pred_csv, out_path, title=None):
    from sklearn.metrics import mean_squared_error
    df = pd.read_csv(pred_csv)
    rmse = mean_squared_error(df["y_true"], df["y_pred"], squared=False)
    mae = np.mean(np.abs(df["y_true"] - df["y_pred"]))
    plt.figure(figsize=(5,5))
    plt.scatter(df["y_true"], df["y_pred"], s=6, alpha=0.6)
    lims = [min(df["y_true"].min(), df["y_pred"].min()),
            max(df["y_true"].max(), df["y_pred"].max())]
    plt.plot(lims, lims, 'r--', alpha=0.8)
    plt.xlabel("True TMAX next day (°C)")
    plt.ylabel("Predicted TMAX next day (°C)")
    if title: plt.title(f"{title}\nRMSE={rmse:.2f}°C, MAE={mae:.2f}°C")
    else: plt.title(f"RMSE={rmse:.2f}°C, MAE={mae:.2f}°C")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()

def plot_composite_summary(pred_csvs, out_path, station, split, title=None):
    """Create a 2x2 composite figure with all plot types."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1) Time series (top left)
    ax1 = axes[0, 0]
    dfs = {}
    for name, p in pred_csvs.items():
        df = pd.read_csv(p, parse_dates=["DATE"]).sort_values("DATE")
        dfs[name] = df
    
    base = next(iter(dfs.values()))
    ax1.plot(base["DATE"], base["y_true"], label="Truth", linewidth=1.5, color='black')
    for name, df in dfs.items():
        ax1.plot(df["DATE"], df["y_pred"], label=name, linewidth=1, alpha=0.8)
    ax1.set_xlabel("Date"); ax1.set_ylabel("TMAX next day (°C)")
    ax1.legend(); ax1.set_title("Time Series Comparison")
    
    # 2) Scatter plot for Ridge (top right)
    ax2 = axes[0, 1]
    ridge_df = pd.read_csv(pred_csvs["ridge"])
    rmse = np.sqrt(np.mean((ridge_df["y_true"] - ridge_df["y_pred"])**2))
    mae = np.mean(np.abs(ridge_df["y_true"] - ridge_df["y_pred"]))
    ax2.scatter(ridge_df["y_true"], ridge_df["y_pred"], s=6, alpha=0.6)
    lims = [min(ridge_df["y_true"].min(), ridge_df["y_pred"].min()),
            max(ridge_df["y_true"].max(), ridge_df["y_pred"].max())]
    ax2.plot(lims, lims, 'r--', alpha=0.8)
    ax2.set_xlabel("True TMAX (°C)"); ax2.set_ylabel("Predicted TMAX (°C)")
    ax2.set_title(f"Ridge: RMSE={rmse:.2f}°C, MAE={mae:.2f}°C")
    
    # 3) Residuals by DOY (bottom left)
    ax3 = axes[1, 0]
    df = pd.read_csv(pred_csvs["ridge"], parse_dates=["DATE"])
    df["resid"] = df["y_pred"] - df["y_true"]
    df["DOY"] = df["DATE"].dt.dayofyear
    edges = np.linspace(1, 366, 13)
    idx = np.digitize(df["DOY"], edges) - 1
    df["bin"] = idx
    means = df.groupby("bin")["resid"].mean()
    centers = 0.5*(edges[:-1] + edges[1:])
    valid_bins = means.index[means.index < len(centers)]
    ax3.bar(centers[valid_bins], means[valid_bins].values, width=(edges[1]-edges[0])*0.9)
    ax3.axhline(0, linewidth=1, color='red', alpha=0.7)
    ax3.set_xlabel("Day of year"); ax3.set_ylabel("Mean residual (°C)")
    ax3.set_title("Seasonal Bias (Ridge)")
    
    # 4) Model comparison (bottom right)
    ax4 = axes[1, 1]
    model_names = []
    rmse_values = []
    for name, p in pred_csvs.items():
        df = pd.read_csv(p)
        rmse = np.sqrt(np.mean((df["y_true"] - df["y_pred"])**2))
        model_names.append(name.title())
        rmse_values.append(rmse)
    
    bars = ax4.bar(model_names, rmse_values, alpha=0.7)
    ax4.set_ylabel("RMSE (°C)")
    ax4.set_title("Model Comparison")
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, rmse_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.suptitle(f"{station.upper()} — {split.upper()} Split Analysis", fontsize=14)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()
