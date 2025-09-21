"""
Bayesian Neural Network Training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import yaml
import json
import random
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our calibration utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from metrics.calibration import (
    gaussian_nll, gaussian_crps, prediction_intervals, 
    combine_normal_moments, crps_from_samples, interval_coverage, pit_values
)
from utils.config import load_config_with_base
from sklearn.preprocessing import StandardScaler

def nll_gauss_torch(y, mu, sigma):
    """Gaussian NLL in PyTorch for training."""
    eps = 1e-8
    sigma = torch.clamp(sigma, min=eps)
    return 0.5 * (torch.log(2 * torch.pi * torch.ones_like(sigma)) + 2*torch.log(sigma) + ((y - mu)**2) / (sigma**2))

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def read_table(path):
    """Read table with auto-detection of format."""
    p = Path(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p, parse_dates=['DATE'])

def mask_for_split(df, split, train_end, val_end):
    """Get boolean mask for data split."""
    y = df['DATE'].dt.year
    if split == 'train': 
        return (y <= train_end)
    if split == 'val':   
        return (y > train_end) & (y <= val_end)
    if split == 'test':  
        return (y > val_end)
    raise ValueError(f"Unknown split: {split}")

def standardize_splits(X_train, X_val, X_test):
    """Standardize features using train statistics."""
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.numpy())
    X_val_np = scaler.transform(X_val.numpy()) if len(X_val) > 0 else np.empty((0, X_train.shape[1]))
    X_test_np = scaler.transform(X_test.numpy()) if len(X_test) > 0 else np.empty((0, X_train.shape[1]))
    
    return (torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(X_val_np, dtype=torch.float32),
            torch.tensor(X_test_np, dtype=torch.float32),
            scaler)

def summarize_and_save(mu, sigma, y, out_json):
    """Compute and save comprehensive metrics."""
    nll = float(gaussian_nll(y, mu, sigma))
    crps = float(gaussian_crps(y, mu, sigma))
    
    # 90% intervals
    lo90, hi90 = prediction_intervals(mu, sigma, alpha=0.1)
    cov90, w90 = interval_coverage(y, lo90, hi90)
    
    # 50% intervals
    lo50, hi50 = prediction_intervals(mu, sigma, alpha=0.5)
    cov50, w50 = interval_coverage(y, lo50, hi50)
    
    # PIT values
    pit = pit_values(y, mu, sigma)
    
    metrics = {
        'nll': float(nll), 
        'crps': float(crps),
        'cov90': float(cov90), 
        'width90': float(w90),
        'cov50': float(cov50), 
        'width50': float(w50),
        'pit_mean': float(pit.mean()), 
        'pit_std': float(pit.std())
    }
    
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)

class HeteroscedasticMLP(nn.Module):
    """
    MLP with heteroscedastic head for uncertainty estimation.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        
        # Build trunk
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Heteroscedastic heads
        self.mu_head = nn.Linear(prev_dim, 1)
        self.logvar_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        # Trunk forward pass
        h = self.trunk(x)
        
        # Mean prediction
        mu = self.mu_head(h).squeeze(-1)
        
        # Variance prediction (logvar -> sigma via softplus)
        logvar = self.logvar_head(h).squeeze(-1)
        sigma = torch.nn.functional.softplus(0.5 * logvar) + 1e-6
        
        return mu, sigma

class EarlyStopping:
    """Early stopping based on validation CRPS."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_score: float):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def prepare_data(df: pd.DataFrame, feature_cols: List[str], target_col: str, 
                train_end: int, val_end: int) -> Tuple[torch.Tensor, ...]:
    """
    Prepare data tensors for training.
    """
    # Filter data
    train_data = df[df['DATE'].dt.year <= train_end].copy()
    val_data = df[(df['DATE'].dt.year > train_end) & (df['DATE'].dt.year <= val_end)].copy()
    test_data = df[df['DATE'].dt.year > val_end].copy()
    
    print(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Extract features and targets
    def extract_split(data):
        if len(data) == 0:
            return torch.empty(0, len(feature_cols)), torch.empty(0)
        
        X = torch.tensor(data[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(data[target_col].values, dtype=torch.float32)
        return X, y
    
    X_train, y_train = extract_split(train_data)
    X_val, y_val = extract_split(val_data)
    X_test, y_test = extract_split(test_data)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        mu, sigma = model(X)
        loss = nll_gauss_torch(y, mu, sigma).mean()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate model and return NLL and CRPS."""
    model.eval()
    all_mus, all_sigmas, all_ys = [], [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            mu, sigma = model(X)
            
            all_mus.append(mu.cpu().numpy())
            all_sigmas.append(sigma.cpu().numpy())
            all_ys.append(y.cpu().numpy())
    
    mus = np.concatenate(all_mus)
    sigmas = np.concatenate(all_sigmas)
    ys = np.concatenate(all_ys)
    
    nll = gaussian_nll(ys, mus, sigmas)
    crps = gaussian_crps(ys, mus, sigmas)
    
    return nll, crps

def mc_predict(model: nn.Module, X: torch.Tensor, T: int = 100, device: torch.device = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout prediction.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()  # Keep dropout active
    mus, vars_ = [], []
    
    with torch.no_grad():
        for _ in range(T):
            mu, sigma = model(X.to(device))
            mus.append(mu.cpu().numpy())
            vars_.append((sigma.cpu().numpy())**2)
    
    mus = np.stack(mus, 0)
    vars_ = np.stack(vars_, 0)
    
    # Combine moments
    mu_bar, sigma_bar = combine_normal_moments(mus, vars_)
    
    return mu_bar, sigma_bar

def ensemble_predict(model_paths: List[str], X: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensemble prediction from multiple saved models.
    """
    mus, vars_ = [], []
    
    for model_path in model_paths:
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device)
        model = HeteroscedasticMLP(ckpt['input_dim'], ckpt['hidden'], ckpt['dropout']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        with torch.no_grad():
            mu, sigma = model(X.to(device))
            mus.append(mu.cpu().numpy())
            vars_.append((sigma.cpu().numpy())**2)
    
    mus = np.stack(mus, 0)
    vars_ = np.stack(vars_, 0)
    
    # Combine moments
    mu_bar, sigma_bar = combine_normal_moments(mus, vars_)
    
    return mu_bar, sigma_bar

def save_predictions(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, 
                    split: str, station: str, output_dir: Path, y_true: np.ndarray = None):
    """
    Save predictions with prediction intervals and metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Assert alignment before saving
    assert len(df) == len(mu), f"Length mismatch for {split}: {len(df)} vs {len(mu)}"
    assert len(mu) == len(sigma), f"Length mismatch for {split}: {len(mu)} vs {len(sigma)}"
    
    # Create prediction dataframe
    pred_df = df[['DATE']].copy()
    if y_true is not None:
        pred_df['y_true'] = y_true
        assert len(y_true) == len(mu), f"Length mismatch for {split}: {len(y_true)} vs {len(mu)}"
    else:
        pred_df['y_true'] = df['y_tmax_next'] if 'y_tmax_next' in df.columns else df['y_true']
    pred_df['mu'] = mu
    pred_df['sigma'] = sigma
    
    # Add prediction intervals
    for q in [0.5, 0.9]:
        lo, hi = prediction_intervals(mu, sigma, alpha=1-q)
        pred_df[f'lo{int(q*100)}'] = lo
        pred_df[f'hi{int(q*100)}'] = hi
    
    # Save predictions
    output_path = output_dir / f"{station}_{split}_bnn.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")
    
    # Save metrics
    metrics_path = output_dir / f"{station}_{split}_metrics.json"
    summarize_and_save(mu, sigma, pred_df['y_true'].values, metrics_path)
    print(f"Saved metrics: {metrics_path}")

def train_single_model(config: Dict, X_train: torch.Tensor, y_train: torch.Tensor,
                      X_val: torch.Tensor, y_val: torch.Tensor, device: torch.device,
                      model_save_path: str = None) -> nn.Module:
    """
    Train a single BNN model.
    """
    # Create model
    model = HeteroscedasticMLP(
        input_dim=X_train.shape[1],
        hidden_dims=config['hidden'],
        dropout=config['dropout']
    ).to(device)
    
    # Optimiser
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                             pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                           pin_memory=torch.cuda.is_available())
    
    # Training loop
    early_stopping = EarlyStopping(patience=config['patience'])
    best_model_state = None
    
    print(f"Training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_nll, val_crps = evaluate_model(model, val_loader, device)
        
        # Early stopping check
        early_stopping(val_crps)
        if early_stopping.best_score == val_crps:
            best_model_state = model.state_dict().copy()
        
        if epoch % 5 == 0:  # Print every 5 epochs
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val NLL={val_nll:.4f}, Val CRPS={val_crps:.4f}")
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model checkpoint
    if model_save_path:
        checkpoint = {
            'state_dict': model.state_dict(),
            'input_dim': X_train.shape[1],
            'hidden': config['hidden'],
            'dropout': config['dropout']
        }
        torch.save(checkpoint, model_save_path)
        print(f"Saved model: {model_save_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Bayesian Neural Network')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--features', required=True, help='Path to features CSV/Parquet')
    parser.add_argument('--output', default='results/bnn', help='Output directory')
    parser.add_argument('--device', default='auto', help='Device (cpu/cuda/auto)')
    args = parser.parse_args()
    
    # Load config
    config = load_config_with_base(args.config)
    print(f"Loaded config: {args.config}")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data with auto-detection
    df = read_table(args.features)
    print(f"Loaded data: {len(df)} rows")
    
    # Validate columns
    missing = [c for c in config['feature_cols'] + [config['target_col']] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        df, config['feature_cols'], config['target_col'], 
        config['train_end'], config['val_end']
    )
    
    # Standardize features
    X_train, X_val, X_test, scaler = standardize_splits(X_train, X_val, X_test)
    
    # Get station name
    station = Path(args.features).stem
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    joblib.dump(scaler, output_dir / f"{station}_scaler.joblib")
    print(f"Saved scaler: {station}_scaler.joblib")
    
    # Save config and feature list for reproducibility
    with open(output_dir / f"{station}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / f"{station}_features.json", "w") as f:
        json.dump(config['feature_cols'], f, indent=2)
    print(f"Saved config and features: {station}_config.json, {station}_features.json")
    
    if config['mode'] == 'mc':
        # Monte Carlo Dropout
        print("\nMonte Carlo Dropout Training")
        
        # Set random seed
        set_seed(config['seed'])
        
        # Train model
        model_save_path = output_dir / f"{station}_mc_model.pth"
        model = train_single_model(config, X_train, y_train, X_val, y_val, device, model_save_path)
        
        # MC Predictions
        print("Generating MC predictions...")
        
        for split_name, X_split, y_split in [('train', X_train, y_train), 
                                           ('val', X_val, y_val), 
                                           ('test', X_test, y_test)]:
            if len(X_split) == 0:
                continue
                
            print(f"{split_name} split...")
            mu, sigma = mc_predict(model, X_split, T=config['mc_samples'], device=device)
            
            # Create split dataframe using helper with alignment check
            m = mask_for_split(df, split_name, config['train_end'], config['val_end'])
            split_df = df.loc[m].copy().reset_index(drop=True)
            assert len(split_df) == len(mu), f"Length mismatch for {split_name}: {len(split_df)} vs {len(mu)}"
            
            save_predictions(split_df, mu, sigma, split_name, station, output_dir, y_split.numpy())
    
    elif config['mode'] == 'ensemble':
        # Ensemble training
        print("\nEnsemble Training")
        
        model_paths = []
        
        for i, seed in enumerate(config['seeds']):
            print(f"\nTraining ensemble member {i+1}/{len(config['seeds'])} (seed={seed})")
            
            # Set seed
            set_seed(seed)
            
            # Train model
            model_save_path = output_dir / f"{station}_ens_{i+1}_model.pth"
            model = train_single_model(config, X_train, y_train, X_val, y_val, device, model_save_path)
            model_paths.append(model_save_path)
        
        # Ensemble predictions
        print("Generating ensemble predictions...")
        
        for split_name, X_split, y_split in [('train', X_train, y_train), 
                                           ('val', X_val, y_val), 
                                           ('test', X_test, y_test)]:
            if len(X_split) == 0:
                continue
                
            print(f"  {split_name} split...")
            mu, sigma = ensemble_predict(model_paths, X_split, device)
            
            # Create split dataframe using helper with alignment check
            m = mask_for_split(df, split_name, config['train_end'], config['val_end'])
            split_df = df.loc[m].copy().reset_index(drop=True)
            assert len(split_df) == len(mu), f"Length mismatch for {split_name}: {len(split_df)} vs {len(mu)}"
            
            save_predictions(split_df, mu, sigma, split_name, station, output_dir, y_split.numpy())
    
    else:
        raise ValueError(f"Unknown mode: {config['mode']}")
    
    print(f"\nTraining complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
