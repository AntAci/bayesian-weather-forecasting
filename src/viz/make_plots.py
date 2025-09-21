import argparse
from pathlib import Path
from plots import plot_time_window, plot_residuals_vs_doy, plot_scatter_with_rmse, plot_composite_summary

def main(station: str, split: str, preds_dir="results/preds", out_dir="results/plots"):
    preds_dir = Path(preds_dir)
    out_dir = Path(out_dir) / station  
    # expected filenames
    files = {
        "climatology": preds_dir / f"{station}_climatology_{split}.csv",
        "persistence": preds_dir / f"{station}_persistence_{split}.csv",
        "ridge": preds_dir / f"{station}_ridge_{split}.csv",
    }
    
    # Check which files exist
    available_files = {k: v for k, v in files.items() if v.exists()}
    
    print(f"Found {len(available_files)} prediction files for {station} {split}")
    
    #time window (last 60 days of the split)
    plot_time_window(
        {k: str(v) for k,v in available_files.items()},
        out_path=str(out_dir / f"{station}_{split}_timeseries.png"),
        title=f"{station.upper()} — {split.upper()} (truth vs baselines)"
    )
    print(f"  ✓ Time series plot: {station}_{split}_timeseries.png")
    
    # residuals vs DOY 
    best_model = "ridge" if "ridge" in available_files else list(available_files.keys())[0]
    plot_residuals_vs_doy(
        str(available_files[best_model]),
        out_path=str(out_dir / f"{station}_{split}_resid_by_doy.png"),
        title=f"{station.upper()} — {split.upper()} ({best_model} residuals vs DOY)"
    )
    print(f"Residuals plot: {station}_{split}_resid_by_doy.png")
    
    # scatter with RMSE using best model
    plot_scatter_with_rmse(
        str(available_files[best_model]),
        out_path=str(out_dir / f"{station}_{split}_scatter.png"),
        title=f"{station.upper()} — {split.upper()} ({best_model})"
    )
    print(f"Scatter plot: {station}_{split}_scatter.png")
    
    #Composite summary
    if len(available_files) > 1:
        plot_composite_summary(
            {k: str(v) for k,v in available_files.items()},
            out_path=str(out_dir / f"{station}_{split}_summary.png"),
            station=station, split=split
        )
        print(f"Composite summary: {station}_{split}_summary.png")
    
    print(f"\nAll plots saved to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True, help="e.g., oxford or wisley")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--preds_dir", default="results/preds")
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()
    main(args.station, args.split, args.preds_dir, args.out_dir)
