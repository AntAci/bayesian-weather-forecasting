import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, binom

# your utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from metrics.calibration import (
    pit_values, event_probability, reliability_curve, prediction_intervals
)

def read_csv(path):
    return pd.read_csv(path, parse_dates=["DATE"])

def _require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def p_quantile_from_train(train_csv, q=0.90):
    df = read_csv(train_csv)
    _require_cols(df, ["y_true"], "train_csv")
    return float(np.quantile(df["y_true"].dropna().values, q))

def plot_pit(test_csv, out_png, title):
    df = read_csv(test_csv)
    _require_cols(df, ["y_true","mu","sigma"], "test_csv")
    pit = pit_values(df["y_true"].values, df["mu"].values, df["sigma"].values)

    # KS test for uniformity
    stat, pval = kstest(pit, 'uniform')
    print(f"[PIT] KS stat={stat:.3f}, p={pval:.3f}")

    plt.figure(figsize=(7,4))
    plt.hist(pit, bins=20, density=True, edgecolor="black")
    plt.axhline(1.0, linestyle="--", linewidth=2, label="Perfect (uniform)")
    plt.xlabel("PIT value"); plt.ylabel("Density")
    plt.title(title + f" - PIT (KS p={pval:.3f})")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight", dpi=160); plt.close()

def plot_reliability(test_csv, threshold, out_png, title, n_bins=10):
    df = read_csv(test_csv)
    _require_cols(df, ["y_true","mu","sigma"], "test_csv")
    p = event_probability(df["mu"].values, df["sigma"].values, threshold, side=">")
    y_event = (df["y_true"].values > threshold).astype(int)

    p_bin, obs_bin, counts = reliability_curve(p, y_event, n_bins=n_bins)

    # Binomial confidence intervals
    lower = np.zeros_like(obs_bin); upper = np.zeros_like(obs_bin); alpha=0.05
    for i, n in enumerate(counts.astype(int)):
        if n > 0:
            lower[i] = binom.ppf(alpha/2, n, obs_bin[i]) / n
            upper[i] = binom.ppf(1-alpha/2, n, obs_bin[i]) / n

    plt.figure(figsize=(5.8,5.8))
    plt.fill_between(p_bin, lower, upper, alpha=0.15, label="Binomial CI")
    plt.plot(p_bin, obs_bin, "o-", label="Reliability")
    plt.plot([0,1],[0,1], "--", label="Perfect")
    for x,y,c in zip(p_bin, obs_bin, counts):
        if c > 0:
            plt.annotate(f"n={int(c)}", (x,y), xytext=(5,5), textcoords="offset points", fontsize=8)
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(title + f" - Heat: TMAX > {threshold:.1f}°C")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=160); plt.close()

def plot_strip(test_csv, out_png, days=45, title=""):
    df = read_csv(test_csv).sort_values("DATE").tail(days)
    _require_cols(df, ["DATE","y_true","mu","sigma"], "test_csv")
    lo90, hi90 = prediction_intervals(df["mu"].values, df["sigma"].values, alpha=0.1)

    plt.figure(figsize=(10,4))
    plt.fill_between(df["DATE"], lo90, hi90, alpha=0.3, label="90% PI")
    plt.plot(df["DATE"], df["mu"].values, label="Predicted μ")
    plt.plot(df["DATE"], df["y_true"].values, label="Truth", linewidth=1)
    plt.ylabel("TMAX (°C)"); plt.title(title + f" - last {days} days")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight", dpi=160); plt.close()

def plot_scatter_rmse(test_csv, out_png, title):
    df = read_csv(test_csv)
    _require_cols(df, ["y_true","mu"], "test_csv")
    y = df["y_true"].values
    yhat = df["mu"].values
    rmse = np.sqrt(np.mean((y - yhat)**2))

    plt.figure(figsize=(5.8,5.8))
    plt.scatter(y, yhat, s=8, alpha=0.6, label="points")
    # 45° line
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    plt.plot(lims, lims, "--", label="y = μ")
    # RMSE band (±RMSE around identity)
    low = np.array(lims) - rmse; high = np.array(lims) + rmse
    plt.plot(lims, high, ":", label=f"+RMSE ({rmse:.2f})")
    plt.plot(lims, low, ":", label=f"-RMSE ({rmse:.2f})")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Truth (°C)"); plt.ylabel("Predicted μ (°C)")
    plt.title(title + f" - scatter (RMSE={rmse:.2f})")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=160); plt.close()

def plot_residuals_vs_doy(test_csv, out_png, title):
    """Optional: residuals against seasonality (day-of-year)."""
    df = read_csv(test_csv)
    _require_cols(df, ["DATE","y_true","mu"], "test_csv")
    resid = df["y_true"].values - df["mu"].values
    doy = df["DATE"].dt.dayofyear.values
    order = np.argsort(doy)

    plt.figure(figsize=(9,3.8))
    plt.scatter(doy, resid, s=6, alpha=0.45)
    # smooth trend
    smooth = pd.Series(resid).rolling(21, min_periods=10, center=True).mean().values
    plt.plot(np.sort(doy), smooth[order], lw=2, alpha=0.9, label="21-day rolling mean")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Day of year"); plt.ylabel("Residual (°C)")
    plt.title(title + " - residuals vs DOY"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight", dpi=160); plt.close()

def plot_multi_panel(test_csv, threshold, out_png, title, days=45):
    """Create a 2x2 multi-panel figure with all diagnostics."""
    df = read_csv(test_csv)
    _require_cols(df, ["DATE","y_true","mu","sigma"], "test_csv")
    
    # Prepare data
    pit = pit_values(df["y_true"].values, df["mu"].values, df["sigma"].values)
    p = event_probability(df["mu"].values, df["sigma"].values, threshold, side=">")
    y_event = (df["y_true"].values > threshold).astype(int)
    p_bin, obs_bin, counts = reliability_curve(p, y_event, n_bins=10)
    
    # Binomial CIs
    lower = np.zeros_like(obs_bin); upper = np.zeros_like(obs_bin); alpha=0.05
    for i, n in enumerate(counts.astype(int)):
        if n > 0:
            lower[i] = binom.ppf(alpha/2, n, obs_bin[i]) / n
            upper[i] = binom.ppf(1-alpha/2, n, obs_bin[i]) / n
    
    # Strip data
    strip_df = df.sort_values("DATE").tail(days)
    lo90, hi90 = prediction_intervals(strip_df["mu"].values, strip_df["sigma"].values, alpha=0.1)
    
    # Scatter data
    y = df["y_true"].values
    yhat = df["mu"].values
    # Remove any NaN values
    mask = ~(np.isnan(y) | np.isnan(yhat))
    y = y[mask]
    yhat = yhat[mask]
    rmse = np.sqrt(np.mean((y - yhat)**2))
    
    # Residuals data
    resid = df["y_true"].values - df["mu"].values
    doy = df["DATE"].dt.dayofyear.values
    # Apply same mask
    resid = resid[mask]
    doy = doy[mask]
    order = np.argsort(doy)
    smooth = pd.Series(resid).rolling(21, min_periods=10, center=True).mean().values
    
    # Handle NaN values in smooth
    smooth = np.nan_to_num(smooth, nan=0.0)
    
    # KS test
    stat, pval = kstest(pit, 'uniform')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title + " - Uncertainty Diagnostics", fontsize=16)
    
    # PIT histogram (top left)
    ax1 = axes[0, 0]
    ax1.hist(pit, bins=20, density=True, edgecolor="black")
    ax1.axhline(1.0, linestyle="--", linewidth=2, label="Perfect (uniform)")
    ax1.set_xlabel("PIT value"); ax1.set_ylabel("Density")
    ax1.set_title(f"PIT (KS p={pval:.3f})")
    ax1.legend()
    
    # Reliability diagram (top right)
    ax2 = axes[0, 1]
    ax2.fill_between(p_bin, lower, upper, alpha=0.15, label="Binomial CI")
    ax2.plot(p_bin, obs_bin, "o-", label="Reliability")
    ax2.plot([0,1],[0,1], "--", label="Perfect")
    for x,y,c in zip(p_bin, obs_bin, counts):
        if c > 0:
            ax2.annotate(f"n={int(c)}", (x,y), xytext=(5,5), textcoords="offset points", fontsize=8)
    ax2.set_xlim(0,1); ax2.set_ylim(0,1)
    ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Observed frequency")
    ax2.set_title(f"Heat: TMAX > {threshold:.1f}°C")
    ax2.grid(alpha=0.3); ax2.legend()
    
    # Time series strip (bottom left)
    ax3 = axes[1, 0]
    ax3.fill_between(strip_df["DATE"], lo90, hi90, alpha=0.3, label="90% PI")
    ax3.plot(strip_df["DATE"], strip_df["mu"].values, label="Predicted μ")
    ax3.plot(strip_df["DATE"], strip_df["y_true"].values, label="Truth", linewidth=1)
    ax3.set_ylabel("TMAX (°C)"); ax3.set_title(f"Last {days} days")
    ax3.legend()
    
    # Scatter plot (bottom right)
    ax4 = axes[1, 1]
    ax4.scatter(y, yhat, s=8, alpha=0.6, label="points")
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    ax4.plot(lims, lims, "--", label="y = μ")
    low = np.array(lims) - rmse; high = np.array(lims) + rmse
    ax4.plot(lims, high, ":", label=f"+RMSE ({rmse:.2f})")
    ax4.plot(lims, low, ":", label=f"-RMSE ({rmse:.2f})")
    ax4.set_xlim(lims); ax4.set_ylim(lims)
    ax4.set_xlabel("Truth (°C)"); ax4.set_ylabel("Predicted μ (°C)")
    ax4.set_title(f"Scatter (RMSE={rmse:.2f})")
    ax4.grid(alpha=0.3); ax4.legend()
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="train preds CSV (for p90)")
    ap.add_argument("--test_csv",  required=True, help="test preds CSV")
    ap.add_argument("--outdir",    required=True)
    ap.add_argument("--title",     default="")
    ap.add_argument("--days",      type=int, default=45)
    ap.add_argument("--threshold", type=float, default=None,
                    help="if given, use this °C threshold instead of train p90")
    ap.add_argument("--multi_panel", action="store_true", 
                    help="create single 2x2 multi-panel figure instead of separate plots")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # threshold
    thr = args.threshold if args.threshold is not None else p_quantile_from_train(args.train_csv, 0.90)

    if args.multi_panel:
        # Single multi-panel figure
        plot_multi_panel(args.test_csv, thr, outdir / "uncertainty_diagnostics.png", args.title, args.days)
    else:
        # Individual plots
        plot_pit(args.test_csv, outdir / "pit.png", args.title)
        plot_reliability(args.test_csv, thr, outdir / "reliability.png", args.title)
        plot_strip(args.test_csv, outdir / "strip.png", args.days, args.title)
        plot_scatter_rmse(args.test_csv, outdir / "scatter_rmse.png", args.title)
        plot_residuals_vs_doy(args.test_csv, outdir / "resid_doy.png", args.title)

if __name__ == "__main__":
    main()
