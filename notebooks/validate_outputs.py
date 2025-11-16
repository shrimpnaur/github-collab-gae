#!/usr/bin/env python3
"""
Automated validation script for GAE pipeline outputs.
Checks all files and metrics for correctness.

Usage:
    python3 validate_outputs.py [--data-root data]
"""

import json
import sys
import os
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check(condition, message, error_msg=""):
    """Helper to print check results."""
    if condition:
        print(f"{GREEN}✓{RESET} {message}")
        return True
    else:
        print(f"{RED}✗{RESET} {message}")
        if error_msg:
            print(f"  {error_msg}")
        return False

def validate_file_exists(path, name):
    """Check if file exists."""
    exists = os.path.isfile(path)
    size = f"({os.path.getsize(path) / 1024:.1f} KB)" if exists else ""
    return check(exists, f"File exists: {name} {size}", f"Missing: {path}")

def validate_pyg_data(data_root):
    """Validate graph_data.pt structure."""
    print(f"\n{YELLOW}Validating graph_data.pt{RESET}")
    print("-" * 50)
    
    path = os.path.join(data_root, "processed", "graph_data.pt")
    if not validate_file_exists(path, "graph_data.pt"):
        return False
    
    try:
        d = torch.load(path, weights_only=False)
        
        # Check keys
        check("data" in d and "nodes" in d, 
              f"Keys present: {list(d.keys())}")
        
        # Check structure
        num_nodes = len(d['nodes'])
        num_edges = d['data'].edge_index.shape[1]
        x_shape = d['data'].x.shape
        
        print(f"  Nodes: {num_nodes}")
        print(f"  Edge pairs (both directions): {num_edges}")
        print(f"  Node features shape: {x_shape}")
        
        # Validate
        checks = [
            (d['data'].edge_index.shape[0] == 2, 
             "Edge index has 2 rows"),
            (x_shape[0] == num_nodes, 
             f"Features match nodes ({x_shape[0]} == {num_nodes})"),
            (num_edges % 2 == 0, 
             f"Even number of edges (undirected): {num_edges}"),
            (x_shape[1] > 0, 
             f"Node features are non-empty: {x_shape[1]} dims"),
        ]
        
        all_passed = all(c[0] for c in checks)
        for cond, msg in checks:
            check(cond, msg)
        
        return all_passed
    except Exception as e:
        check(False, "Load and validate graph_data.pt", str(e))
        return False

def validate_baseline_metrics(data_root):
    """Validate baseline_metrics.json."""
    print(f"\n{YELLOW}Validating baseline_metrics.json{RESET}")
    print("-" * 50)
    
    path = os.path.join(data_root, "processed", "baseline_metrics.json")
    if not validate_file_exists(path, "baseline_metrics.json"):
        return False
    
    try:
        with open(path) as f:
            metrics = json.load(f)
        
        num_baselines = len(metrics.get('baselines', []))
        print(f"  Baselines: {num_baselines}")
        
        all_valid = True
        for baseline in metrics.get('baselines', []):
            method = baseline.get('method', 'Unknown')
            auc = baseline.get('auc', -1)
            ap = baseline.get('ap', -1)
            
            valid_auc = 0.0 <= auc <= 1.0
            valid_ap = 0.0 <= ap <= 1.0
            
            check(valid_auc and valid_ap,
                  f"{method:20s} AUC={auc:.4f} AP={ap:.4f}")
            
            all_valid = all_valid and valid_auc and valid_ap
        
        return all_valid
    except Exception as e:
        check(False, "Load and validate baseline metrics", str(e))
        return False

def validate_gae_metrics(data_root):
    """Validate gae_metrics.json."""
    print(f"\n{YELLOW}Validating gae_metrics.json{RESET}")
    print("-" * 50)
    
    path = os.path.join(data_root, "processed", "gae_metrics.json")
    if not validate_file_exists(path, "gae_metrics.json"):
        return False
    
    try:
        with open(path) as f:
            metrics = json.load(f)
        
        auc = metrics.get('auc', -1)
        ap = metrics.get('ap', -1)
        epochs = metrics.get('epochs', 0)
        device = metrics.get('device', 'Unknown')
        
        print(f"  AUC: {auc:.4f}")
        print(f"  AP:  {ap:.4f}")
        print(f"  Epochs: {epochs}")
        print(f"  Device: {device}")
        
        checks = [
            (0.0 <= auc <= 1.0, f"Valid AUC: {auc:.4f}"),
            (0.0 <= ap <= 1.0, f"Valid AP: {ap:.4f}"),
            (epochs > 0, f"Epochs > 0: {epochs}"),
            (auc > 0.0, "AUC > 0.0 (model learned)"),
            (ap > 0.0, "AP > 0.0 (model learned)"),
        ]
        
        all_passed = True
        for cond, msg in checks:
            check(cond, msg)
            all_passed = all_passed and cond
        
        return all_passed
    except Exception as e:
        check(False, "Load and validate GAE metrics", str(e))
        return False

def validate_training_logs(data_root):
    """Validate gae_training_logs.json."""
    print(f"\n{YELLOW}Validating gae_training_logs.json{RESET}")
    print("-" * 50)
    
    path = os.path.join(data_root, "processed", "gae_training_logs.json")
    if not validate_file_exists(path, "gae_training_logs.json"):
        return False
    
    try:
        with open(path) as f:
            logs = json.load(f)
        
        num_epochs = len(logs)
        initial_loss = logs[0]['loss']
        final_loss = logs[-1]['loss']
        
        print(f"  Epochs logged: {num_epochs}")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        
        improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
        print(f"  Loss reduction: {improvement:.1f}%")
        
        checks = [
            (num_epochs > 0, f"Non-empty logs: {num_epochs} epochs"),
            (all(isinstance(log.get('epoch'), int) for log in logs), 
             "All epochs are integers"),
            (all(isinstance(log.get('loss'), (int, float)) for log in logs), 
             "All losses are numbers"),
            (initial_loss > 0 and final_loss > 0, 
             "Losses are positive"),
        ]
        
        all_passed = True
        for cond, msg in checks:
            check(cond, msg)
            all_passed = all_passed and cond
        
        return all_passed
    except Exception as e:
        check(False, "Load and validate training logs", str(e))
        return False

def validate_embeddings(data_root):
    """Validate gae_embeddings.npy."""
    print(f"\n{YELLOW}Validating gae_embeddings.npy{RESET}")
    print("-" * 50)
    
    path = os.path.join(data_root, "processed", "gae_embeddings.npy")
    if not validate_file_exists(path, "gae_embeddings.npy"):
        return False
    
    try:
        emb = np.load(path)
        
        print(f"  Shape: {emb.shape}")
        print(f"  Dtype: {emb.dtype}")
        print(f"  Min: {emb.min():.6f}")
        print(f"  Max: {emb.max():.6f}")
        print(f"  Mean: {emb.mean():.6f}")
        
        checks = [
            (len(emb.shape) == 2, f"2D array: shape {emb.shape}"),
            (emb.shape[0] > 0 and emb.shape[1] > 0, 
             f"Non-empty: {emb.shape}"),
            (emb.dtype == np.float32 or emb.dtype == np.float64, 
             f"Float dtype: {emb.dtype}"),
        ]
        
        all_passed = True
        for cond, msg in checks:
            check(cond, msg)
            all_passed = all_passed and cond
        
        return all_passed
    except Exception as e:
        check(False, "Load and validate embeddings", str(e))
        return False

def validate_predictions(data_root):
    """Validate predicted_links_top50.csv."""
    print(f"\n{YELLOW}Validating predicted_links_top50.csv{RESET}")
    print("-" * 50)
    
    path = os.path.join(data_root, "processed", "predicted_links_top50.csv")
    if not validate_file_exists(path, "predicted_links_top50.csv"):
        return False
    
    try:
        df = pd.read_csv(path)
        
        required_cols = ['u', 'v', 'score', 'u_idx', 'v_idx']
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        checks = [
            (len(df) > 0, f"Non-empty: {len(df)} predictions"),
            (all(col in df.columns for col in required_cols), 
             f"All required columns: {required_cols}"),
            (df['score'].min() >= 0, 
             f"Non-negative scores: min={df['score'].min():.4f}"),
            (all(isinstance(idx, (int, np.integer)) for idx in df['u_idx']), 
             "u_idx are integers"),
            (all(isinstance(idx, (int, np.integer)) for idx in df['v_idx']), 
             "v_idx are integers"),
        ]
        
        all_passed = True
        for cond, msg in checks:
            check(cond, msg)
            all_passed = all_passed and cond
        
        # Show sample
        print(f"\n  Top 3 predictions:")
        for idx, row in df.head(3).iterrows():
            print(f"    {row['u']:20s} <-> {row['v']:20s} score={row['score']:.4f}")
        
        return all_passed
    except Exception as e:
        check(False, "Load and validate predictions", str(e))
        return False

def validate_model(data_root):
    """Validate gae_model.pt."""
    print(f"\n{YELLOW}Validating gae_model.pt{RESET}")
    print("-" * 50)
    
    path = os.path.join(data_root, "processed", "gae_model.pt")
    if not validate_file_exists(path, "gae_model.pt"):
        return False
    
    try:
        state_dict = torch.load(path, weights_only=True)
        
        num_params = len(state_dict)
        print(f"  Parameters: {num_params}")
        print(f"  Layer names: {list(state_dict.keys())[:3]}...")
        
        checks = [
            (isinstance(state_dict, dict), "State dict is a dictionary"),
            (num_params > 0, f"Non-empty: {num_params} parameters"),
        ]
        
        all_passed = True
        for cond, msg in checks:
            check(cond, msg)
            all_passed = all_passed and cond
        
        return all_passed
    except Exception as e:
        check(False, "Load and validate model", str(e))
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Validate GAE pipeline outputs"
    )
    parser.add_argument('--data-root', default='data',
                        help='Data root directory (default: data)')
    args = parser.parse_args()
    
    print(f"\n{YELLOW}{'='*50}")
    print("GAE Pipeline Output Validation")
    print(f"{'='*50}{RESET}")
    print(f"Data root: {args.data_root}")
    
    # Run all validations
    results = {
        "PyG Data": validate_pyg_data(args.data_root),
        "Baseline Metrics": validate_baseline_metrics(args.data_root),
        "GAE Metrics": validate_gae_metrics(args.data_root),
        "Training Logs": validate_training_logs(args.data_root),
        "Embeddings": validate_embeddings(args.data_root),
        "Predictions": validate_predictions(args.data_root),
        "Model": validate_model(args.data_root),
    }
    
    # Summary
    print(f"\n{YELLOW}{'='*50}")
    print("Summary")
    print(f"{'='*50}{RESET}")
    
    for test, passed in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"{test:20s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n{GREEN}✓ All validations passed!{RESET}")
        print("\nNext steps:")
        print("  1. jupyter notebook notebooks/gae_quick_demo.ipynb")
        print("  2. Compare baseline vs GAE results")
        print("  3. Run full training: python3 scripts/train_gae.py")
        return 0
    else:
        print(f"\n{RED}✗ Some validations failed{RESET}")
        print("\nTroubleshooting:")
        print("  - Check data/processed/ directory exists and is writable")
        print("  - Ensure graph_data.pt was created by prepare_pyg_data.py")
        print("  - Verify all scripts completed without errors")
        return 1

if __name__ == '__main__':
    sys.exit(main())
