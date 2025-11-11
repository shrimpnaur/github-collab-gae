#!/bin/bash

# Smoke Test Suite for GAE Link Prediction Pipeline
# This script validates all major components end-to-end
# Run: bash test_smoke.sh

set -e  # Exit on any error

echo "=================================="
echo "🧪 GAE Link Prediction Smoke Tests"
echo "=================================="
echo ""

# Configuration
DATA_ROOT="data"
VENV_DIR="venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        return 1
    fi
}

# Helper function for file size
file_size() {
    if [ -f "$1" ]; then
        du -h "$1" | awk '{print $1}'
    else
        echo "N/A"
    fi
}

# Test 1: Check virtual environment
echo -e "${YELLOW}[Test 1]${NC} Virtual Environment Setup"
echo "---"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}✗${NC} Virtual environment not found at $VENV_DIR"
    echo "   Create it with: python3 -m venv venv"
    exit 1
fi
echo -e "${GREEN}✓${NC} Virtual environment found"
source $VENV_DIR/bin/activate
python3 -c "import torch; print(f'   torch version: {torch.__version__}')"
python3 -c "import torch_geometric; print(f'   PyG version: {torch_geometric.__version__}')"
echo ""

# Test 2: Check required input files
echo -e "${YELLOW}[Test 2]${NC} Input Data Files"
echo "---"
check_file "$DATA_ROOT/processed/github_collab_graph_clean.gexf" || {
    echo -e "${RED}Error:${NC} Missing GEXF file. Run run_pipeline.py first."
    exit 1
}
check_file "$DATA_ROOT/processed/edges.csv" && echo "   File size: $(file_size $DATA_ROOT/processed/edges.csv)"
check_file "$DATA_ROOT/processed/nodes.csv" && echo "   File size: $(file_size $DATA_ROOT/processed/nodes.csv)"
echo ""

# Test 3: Prepare PyG Data
echo -e "${YELLOW}[Test 3]${NC} Prepare PyG Data (prepare_pyg_data.py)"
echo "---"
echo "Running: python3 scripts/prepare_pyg_data.py --data-root $DATA_ROOT"
python3 scripts/prepare_pyg_data.py --data-root "$DATA_ROOT"
echo ""

# Test 3a: Verify PyG data file
echo -e "${YELLOW}[Test 3a]${NC} Validate graph_data.pt"
echo "---"
check_file "$DATA_ROOT/processed/graph_data.pt" || exit 1
echo "   File size: $(file_size $DATA_ROOT/processed/graph_data.pt)"
echo ""

# Test 3b: Inspect PyG data structure
echo -e "${YELLOW}[Test 3b]${NC} Inspect graph_data.pt Structure"
echo "---"
python3 << 'PYEOF'
import torch
import sys

try:
    d = torch.load("data/processed/graph_data.pt", weights_only=False)
    print(f"   Keys: {list(d.keys())}")
    print(f"   Number of nodes: {len(d['nodes'])}")
    print(f"   Node features shape: {d['data'].x.shape}")
    print(f"   Edge index shape: {d['data'].edge_index.shape}")
    
    # Validate
    num_nodes = len(d['nodes'])
    num_edges_undirected = d['data'].edge_index.shape[1] // 2
    
    print(f"   Expected: (2, 2*|E|) where |E|={num_edges_undirected}")
    print(f"   Actual: {d['data'].edge_index.shape}")
    
    if d['data'].edge_index.shape[0] != 2:
        raise ValueError(f"Edge index should have 2 rows, got {d['data'].edge_index.shape[0]}")
    
    if d['data'].x.shape[0] != num_nodes:
        raise ValueError(f"Features should match nodes: {d['data'].x.shape[0]} != {num_nodes}")
    
    print(f"\n   ✓ Data structure is valid!")
except Exception as e:
    print(f"   ✗ Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Test 4: Run Baselines
echo -e "${YELLOW}[Test 4]${NC} Baseline Link Prediction (baselines_link_pred.py)"
echo "---"
echo "Running: python3 scripts/baselines_link_pred.py --data-root $DATA_ROOT"
python3 scripts/baselines_link_pred.py --data-root "$DATA_ROOT"
echo ""

# Test 4a: Verify baseline metrics
echo -e "${YELLOW}[Test 4a]${NC} Validate baseline_metrics.json"
echo "---"
check_file "$DATA_ROOT/processed/baseline_metrics.json" || exit 1
python3 << 'PYEOF'
import json
import sys

try:
    with open("data/processed/baseline_metrics.json") as f:
        metrics = json.load(f)
    
    print(f"   Baselines evaluated: {len(metrics.get('baselines', []))}")
    for baseline in metrics.get('baselines', []):
        auc = baseline.get('auc', -1)
        ap = baseline.get('ap', -1)
        method = baseline.get('method', 'Unknown')
        print(f"   {method:20s} AUC={auc:.4f} AP={ap:.4f}")
        
        # Validate
        if not (0.0 <= auc <= 1.0):
            raise ValueError(f"Invalid AUC for {method}: {auc}")
        if not (0.0 <= ap <= 1.0):
            raise ValueError(f"Invalid AP for {method}: {ap}")
    
    print(f"\n   ✓ Baseline metrics valid!")
except Exception as e:
    print(f"   ✗ Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Test 5: Quick GAE Training
echo -e "${YELLOW}[Test 5]${NC} GAE Training (train_gae.py --sample)"
echo "---"
echo "Running: python3 scripts/train_gae.py --data-root $DATA_ROOT --sample --seed 42"
python3 scripts/train_gae.py --data-root "$DATA_ROOT" --sample --seed 42
echo ""

# Test 5a: Verify GAE outputs
echo -e "${YELLOW}[Test 5a]${NC} Validate GAE Output Files"
echo "---"
echo "Checking model artifacts:"
check_file "$DATA_ROOT/processed/gae_model.pt" && echo "   Size: $(file_size $DATA_ROOT/processed/gae_model.pt)"
check_file "$DATA_ROOT/processed/gae_embeddings.npy" && echo "   Size: $(file_size $DATA_ROOT/processed/gae_embeddings.npy)"
check_file "$DATA_ROOT/processed/gae_metrics.json" && echo "   Size: $(file_size $DATA_ROOT/processed/gae_metrics.json)"
check_file "$DATA_ROOT/processed/gae_training_logs.json" && echo "   Size: $(file_size $DATA_ROOT/processed/gae_training_logs.json)"
check_file "$DATA_ROOT/processed/predicted_links_top50.csv" && echo "   Size: $(file_size $DATA_ROOT/processed/predicted_links_top50.csv)"
check_file "$DATA_ROOT/processed/layout_positions.json" && echo "   Size: $(file_size $DATA_ROOT/processed/layout_positions.json)"
echo ""

# Test 5b: Validate GAE metrics
echo -e "${YELLOW}[Test 5b]${NC} Validate gae_metrics.json"
echo "---"
python3 << 'PYEOF'
import json
import sys

try:
    with open("data/processed/gae_metrics.json") as f:
        metrics = json.load(f)
    
    auc = metrics.get('auc')
    ap = metrics.get('ap')
    epochs = metrics.get('epochs')
    device = metrics.get('device')
    
    print(f"   AUC: {auc:.4f}")
    print(f"   AP:  {ap:.4f}")
    print(f"   Epochs: {epochs}")
    print(f"   Device: {device}")
    
    # Validate
    if auc is None or ap is None:
        raise ValueError("Missing AUC or AP in metrics")
    if not (0.0 <= auc <= 1.0):
        raise ValueError(f"Invalid AUC: {auc}")
    if not (0.0 <= ap <= 1.0):
        raise ValueError(f"Invalid AP: {ap}")
    
    print(f"\n   ✓ GAE metrics valid!")
except Exception as e:
    print(f"   ✗ Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Test 5c: Validate predictions
echo -e "${YELLOW}[Test 5c]${NC} Validate predicted_links_top50.csv"
echo "---"
python3 << 'PYEOF'
import pandas as pd
import sys

try:
    df = pd.read_csv("data/processed/predicted_links_top50.csv")
    print(f"   Predicted edges: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    required_cols = ['u', 'v', 'score', 'u_idx', 'v_idx']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    print(f"\n   Top 5 predictions:")
    for idx, row in df.head(5).iterrows():
        print(f"   {row['u']:20s} <-> {row['v']:20s} score={row['score']:.4f}")
    
    # Validate
    if df['score'].min() < 0:
        raise ValueError(f"Negative score found: {df['score'].min()}")
    
    print(f"\n   ✓ Predictions valid!")
except Exception as e:
    print(f"   ✗ Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Test 6: Summary
echo -e "${YELLOW}[Test 6]${NC} Summary Report"
echo "---"
echo "Input files:"
ls -lh $DATA_ROOT/processed/github_collab_graph_clean.gexf
echo ""
echo "Generated files:"
ls -lh $DATA_ROOT/processed/*.pt $DATA_ROOT/processed/*.npy $DATA_ROOT/processed/*_metrics.json $DATA_ROOT/processed/predicted_links_top50.csv 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
echo ""

# Final summary
echo -e "${GREEN}=================================="
echo "✓ All Tests Passed!"
echo "==================================${NC}"
echo ""
echo "📊 What was tested:"
echo "   [1] Virtual environment and dependencies"
echo "   [2] Input graph data files"
echo "   [3] PyG data preparation (prepare_pyg_data.py)"
echo "   [4] PyG data structure validation"
echo "   [5] Baseline link prediction (baselines_link_pred.py)"
echo "   [6] Baseline metrics validation"
echo "   [7] GAE training with --sample mode (5 epochs)"
echo "   [8] GAE model artifacts"
echo "   [9] GAE metrics validation"
echo "  [10] Predicted links validation"
echo ""
echo "🚀 Next steps:"
echo "   1. View results: jupyter notebook notebooks/gae_quick_demo.ipynb"
echo "   2. Full training: python3 scripts/train_gae.py --data-root data"
echo "   3. See README.md for detailed instructions"
echo ""
