# 🧪 Smoke Testing Guide

## Overview

This guide walks through **manual smoke tests** to validate the entire pipeline end-to-end. Run in this exact order.

## Prerequisites

1. **Virtual environment activated**
   ```bash
   source venv/bin/activate
   ```

2. **Graph data exists**
   ```bash
   ls -lh data/processed/github_collab_graph_clean.gexf
   ```
   
   If missing, first run:
   ```bash
   python3 scripts/run_pipeline.py --repo owner/repo --commit_limit 100
   ```

---

## Test Suite

### Test 1: Prepare PyG Data

**Purpose**: Convert GEXF graph to PyTorch Geometric Data format

**Command**:
```bash
python3 scripts/prepare_pyg_data.py --data-root data
```

**Expected Output**:
```
Loaded graph with 19 nodes, 60 edges
Saved data/processed/graph_data.pt
Graph data: num_nodes=19, num_edges=120
```

**Validation**:
```bash
# Check file was created
ls -lh data/processed/graph_data.pt

# Inspect structure
python3 << 'PY'
import torch
d = torch.load("data/processed/graph_data.pt", weights_only=False)
print("✓ Keys:", list(d.keys()))
print("✓ Nodes:", len(d['nodes']))
print("✓ Node features shape:", d['data'].x.shape)
print("✓ Edge index shape:", d['data'].edge_index.shape)
print("✓ Expected edges: (2, 2*60)=(2, 120), Actual:", d['data'].edge_index.shape)

# Validate
assert d['data'].edge_index.shape[0] == 2, "Edge index must have 2 rows"
assert d['data'].x.shape[0] == len(d['nodes']), "Features must match nodes"
print("\n✓ Data structure is valid!")
PY
```

**Success Criteria**:
- ✅ File `data/processed/graph_data.pt` exists (50-500 KB)
- ✅ `num_nodes` matches GEXF graph
- ✅ `edge_index.shape[1] == 2 * num_edges` (both directions)
- ✅ `x.shape[0] == num_nodes` (features per node)

---

### Test 2: Run Baseline Methods

**Purpose**: Compute Jaccard, Adamic-Adar, and Preferential Attachment baselines

**Command**:
```bash
python3 scripts/baselines_link_pred.py --data-root data
```

**Expected Output**:
```
Loaded graph: n_nodes=19, n_edges=60
Jaccard: AUC=0.7123, AP=0.6234
Adamic-Adar: AUC=0.7456, AP=0.6789
PreferentialAttachment: AUC=0.6890, AP=0.5912

Saved baseline metrics to data/processed/baseline_metrics.json
```

**Validation**:
```bash
# Inspect metrics
python3 << 'PY'
import json

with open("data/processed/baseline_metrics.json") as f:
    metrics = json.load(f)

print("✓ Baselines evaluated:", len(metrics['baselines']))
for baseline in metrics['baselines']:
    method = baseline['method']
    auc = baseline['auc']
    ap = baseline['ap']
    print(f"  {method:20s} AUC={auc:.4f} AP={ap:.4f}")
    
    # Validate
    assert 0.0 <= auc <= 1.0, f"Invalid AUC: {auc}"
    assert 0.0 <= ap <= 1.0, f"Invalid AP: {ap}"

print("\n✓ All baseline metrics valid!")
PY
```

**Success Criteria**:
- ✅ All 3 baselines print AUC and AP
- ✅ AUC and AP are between 0.0 and 1.0
- ✅ File `data/processed/baseline_metrics.json` exists
- ✅ AUC/AP values are not all ~0.5 (indicates working method)

**Troubleshooting**:
```
If all AUC/AP are ~0.5 or very low:
  → Holdout set is too small (increase graph size)
  → Try reducing holdout_ratio in script
  → Graph may be too sparse for meaningful baselines

If script errors:
  → Check canonicalization: edges should use tuple(sorted((u,v)))
  → Check train/test labels: should match holdout_edges set exactly
```

---

### Test 3: Quick GAE Training (Sample Mode)

**Purpose**: Train GAE for 5 epochs to validate model pipeline

**Command**:
```bash
python3 scripts/train_gae.py --data-root data --sample --seed 42
```

**Expected Output**:
```
Training GAE: epochs=5, seed=42, data_root=data
Loaded data: num_nodes=19, num_node_features=2
Training GAE on device: cpu
Epoch 001 | loss = 0.1234
Epoch 005 | loss = 0.0987

GAE results -> AUC: 0.7234, AP: 0.6543
Saved top 50 predicted links to data/processed/predicted_links_top50.csv
Saved model to data/processed/gae_model.pt
Saved embeddings to data/processed/gae_embeddings.npy
Saved training logs to data/processed/gae_training_logs.json
Saved layout positions to data/processed/layout_positions.json
Saved metrics to data/processed/gae_metrics.json
```

**Validation**:
```bash
# Check all output files exist
ls -lh data/processed/gae_*.* data/processed/layout_positions.json

# Inspect metrics
python3 << 'PY'
import json
import pandas as pd
import numpy as np

# Check metrics
with open("data/processed/gae_metrics.json") as f:
    metrics = json.load(f)

print("✓ GAE Metrics:")
print(f"  AUC: {metrics['auc']:.4f}")
print(f"  AP:  {metrics['ap']:.4f}")
print(f"  Epochs: {metrics['epochs']}")
print(f"  Device: {metrics['device']}")

assert 0.0 <= metrics['auc'] <= 1.0, f"Invalid AUC: {metrics['auc']}"
assert 0.0 <= metrics['ap'] <= 1.0, f"Invalid AP: {metrics['ap']}"

# Check training logs
with open("data/processed/gae_training_logs.json") as f:
    logs = json.load(f)

print(f"\n✓ Training Logs:")
print(f"  Epochs logged: {len(logs)}")
print(f"  Initial loss: {logs[0]['loss']:.4f}")
print(f"  Final loss:   {logs[-1]['loss']:.4f}")

# Check embeddings
emb = np.load("data/processed/gae_embeddings.npy")
print(f"\n✓ Embeddings:")
print(f"  Shape: {emb.shape} (num_nodes x embedding_dim)")

# Check predictions
preds = pd.read_csv("data/processed/predicted_links_top50.csv")
print(f"\n✓ Predictions:")
print(f"  Edges predicted: {len(preds)}")
print(f"  Score range: [{preds['score'].min():.4f}, {preds['score'].max():.4f}]")
print(f"  Top 3:")
for idx, row in preds.head(3).iterrows():
    print(f"    {row['u']:20s} <-> {row['v']:20s} score={row['score']:.4f}")

print("\n✓ All GAE outputs valid!")
PY
```

**Success Criteria**:
- ✅ Training runs and prints loss per epoch
- ✅ Final AUC/AP are finite and > 0.0 (not NaN)
- ✅ All 6 output files exist:
  - `gae_model.pt`
  - `gae_embeddings.npy`
  - `gae_training_logs.json`
  - `gae_metrics.json`
  - `predicted_links_top50.csv`
  - `layout_positions.json`
- ✅ Loss decreases over 5 epochs (e.g., 0.1234 → 0.0987)
- ✅ Prediction scores are in reasonable range (typically 0.0-1.0)

**Troubleshooting**:
```
If loss is NaN or Inf:
  → Check edge_index shape: should be (2, 2*num_edges)
  → Verify data normalization (StandardScaler applied to features)
  → Try CPU instead of GPU: use --device cpu

If AUC/AP are very low (< 0.5):
  → Graph may be too small (try larger --commit_limit)
  → Train/test split may be unlucky (try --seed 100)
  → Model may need more epochs (try --epochs 50)

If files not saved:
  → Check data/processed/ directory exists
  → Verify disk space
  → Check file permissions: chmod 755 data/processed/
```

---

### Test 4: Full Pipeline Run (Optional)

**Purpose**: Validate with full 200-epoch training (takes 2-5 min on GPU)

**Command**:
```bash
python3 scripts/train_gae.py --data-root data --seed 42
```

**Expected Output**:
```
Training GAE: epochs=200, seed=42, data_root=data
...
Epoch 025 | loss = 0.0856
Epoch 050 | loss = 0.0743
...
Epoch 200 | loss = 0.0512

GAE results -> AUC: 0.8234, AP: 0.7543
...
```

**Validation**: Same as Test 3, but with higher epochs and potentially better AUC/AP.

---

## Compare Results

After all tests pass, compare baselines vs GAE:

```bash
python3 << 'PY'
import json
import pandas as pd

# Load baseline metrics
with open("data/processed/baseline_metrics.json") as f:
    baseline = json.load(f)

# Load GAE metrics
with open("data/processed/gae_metrics.json") as f:
    gae = json.load(f)

# Create comparison table
print("=" * 60)
print("BASELINE VS GAE COMPARISON")
print("=" * 60)
print(f"{'Method':<20} {'AUC':<10} {'AP':<10}")
print("-" * 60)

for b in baseline['baselines']:
    print(f"{b['method']:<20} {b['auc']:<10.4f} {b['ap']:<10.4f}")

print("-" * 60)
print(f"{'GAE':<20} {gae['auc']:<10.4f} {gae['ap']:<10.4f}")
print("=" * 60)

# Determine winner
baseline_auc = max([b['auc'] for b in baseline['baselines']])
gae_auc = gae['auc']

print(f"\n🏆 Winner (AUC):")
if gae_auc > baseline_auc:
    improvement = ((gae_auc - baseline_auc) / baseline_auc) * 100
    print(f"   GAE: {gae_auc:.4f} (+{improvement:.1f}% improvement)")
else:
    print(f"   Baseline: {baseline_auc:.4f}")

PY
```

---

## 🔍 What to Inspect If Something Looks Wrong

If tests fail or results look suspicious, use these inspection points to diagnose the issue:

### 1. Train/Test Imbalance

**Symptom**: Metrics don't make sense, or very low AUC/AP

**Inspection**:
```bash
python3 << 'PY'
# Check if test set is large enough
import pandas as pd
import networkx as nx
import pickle

# Load data
G = nx.read_gexf("data/processed/github_collab_graph_clean.gexf")
edges_df = pd.read_csv("data/processed/edges.csv")

train_edges = pd.read_csv("data/processed/train_edges.csv") if os.path.exists("data/processed/train_edges.csv") else None
test_edges = pd.read_csv("data/processed/test_edges.csv") if os.path.exists("data/processed/test_edges.csv") else None

print(f"Total edges: {len(edges_df)}")
if train_edges is not None:
    print(f"Train edges: {len(train_edges)}")
if test_edges is not None:
    print(f"Test edges: {len(test_edges)}")
    
# Diagnose
if test_edges is not None and len(test_edges) < 5:
    print("⚠️  WARNING: Test set is tiny (< 5 edges)!")
    print("    → Increase holdout fraction in baselines_link_pred.py")
    print("    → Or increase commit_limit in run_pipeline.py for more edges")
else:
    print("✓ Test set size looks reasonable")
PY
```

**What it means**:
- If `test_pos` is 0 or 1, predictions are meaningless
- Very small test sets make metrics unreliable
- **Fix**: Increase `holdout_fraction` in `baselines_link_pred.py` or `commit_limit` in `run_pipeline.py`

---

### 2. Edge Ordering Mismatches

**Symptom**: Predictions don't align with original graph, or duplicate edges appear

**Inspection**:
```bash
python3 << 'PY'
import pandas as pd
import networkx as nx

G = nx.read_gexf("data/processed/github_collab_graph_clean.gexf")
edges_df = pd.read_csv("data/processed/edges.csv")

# Check canonicalization (min, max)
print("First 10 edges from GEXF:")
for i, (u, v) in enumerate(list(G.edges())[:10]):
    canonical = tuple(sorted((u, v)))
    print(f"  {i}: ({u}, {v}) → canonical: {canonical}")

print("\nFirst 10 edges from CSV:")
for i, row in edges_df.head(10).iterrows():
    u, v = row[0], row[1]  # Depends on CSV structure
    canonical = tuple(sorted((u, v)))
    print(f"  {i}: ({u}, {v}) → canonical: {canonical}")

# Diagnose
if not all(u <= v for u, v in edges_df.iloc[:, :2].values):
    print("\n⚠️  WARNING: Not all edges are canonical (u < v or u == v)!")
    print("    → Apply canonicalization in edge loading")
else:
    print("\n✓ All edges appear canonicalized")
PY
```

**What it means**:
- If `(a, b)` and `(b, a)` are both listed, edges aren't canonical
- Mismatched ordering between GEXF and CSV causes alignment issues
- **Fix**: Ensure `canon(u, v) = tuple(sorted((u, v)))` applied everywhere

---

### 3. edge_index Shape Mismatch

**Symptom**: Model training fails, or "edge_index shape error"

**Inspection**:
```bash
python3 << 'PY'
import torch
import networkx as nx

# Load PyG data
d = torch.load("data/processed/graph_data.pt", weights_only=False)
data = d['data']

# Load original graph
G = nx.read_gexf("data/processed/github_collab_graph_clean.gexf")

num_nodes = len(d['nodes'])
num_original_edges = len(G.edges())

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges (original): {num_original_edges}")
print(f"edge_index shape: {data.edge_index.shape}")
print(f"Expected shape for undirected: (2, {2 * num_original_edges})")

# Diagnose
if data.edge_index.shape[1] != 2 * num_original_edges:
    print(f"\n⚠️  WARNING: edge_index has {data.edge_index.shape[1]} entries")
    print(f"          Expected: {2 * num_original_edges} (both directions)")
    print("    → Check prepare_pyg_data.py: missing bidirectional edge construction?")
    print("    → Verify: edges_idx.append((i, j)); edges_idx.append((j, i))")
else:
    print("\n✓ edge_index shape is correct (both directions present)")
PY
```

**What it means**:
- For undirected graphs, `edge_index.shape[1]` should be `2 * num_original_edges`
- If only ~half, the reverse direction edges are missing
- **Fix**: Ensure `edges_idx.append((j, i))` follows `edges_idx.append((i, j))`

---

### 4. Node Ordering Mismatch

**Symptom**: Baseline scores don't align with predictions, or node names don't match

**Inspection**:
```bash
python3 << 'PY'
import torch
import pandas as pd

# Load PyG data
d = torch.load("data/processed/graph_data.pt", weights_only=False)
pyg_nodes = d['nodes']

# Load CSV nodes
csv_nodes = pd.read_csv("data/processed/nodes.csv")['node'].tolist()

print(f"PyG data has {len(pyg_nodes)} nodes")
print(f"CSV file has {len(csv_nodes)} nodes")

print("\nFirst 10 nodes (PyG):")
for i, n in enumerate(pyg_nodes[:10]):
    print(f"  {i}: {n}")

print("\nFirst 10 nodes (CSV):")
for i, n in enumerate(csv_nodes[:10]):
    print(f"  {i}: {n}")

# Diagnose
if pyg_nodes != csv_nodes:
    print("\n⚠️  WARNING: Node order differs between PyG and CSV!")
    mismatches = sum(1 for i in range(min(len(pyg_nodes), len(csv_nodes))) 
                     if pyg_nodes[i] != csv_nodes[i])
    print(f"    → {mismatches} mismatches in first {min(len(pyg_nodes), len(csv_nodes))} nodes")
    print("    → Baseline and GNN predictions will be misaligned!")
    print("    → Fix: Ensure same node ordering in prepare_pyg_data.py and baselines_link_pred.py")
else:
    print("\n✓ Node ordering matches between PyG and CSV")
PY
```

**What it means**:
- If node order differs between `graph_data.pt` and `nodes.csv`, indices are misaligned
- Node 0 in PyG might be "user_a", but CSV thinks it's "user_b"
- Predictions and baseline scores will reference wrong nodes
- **Fix**: Keep consistent node list across all files

---

### 5. Quick Diagnostic Checklist

Run this if results seem wrong:

```bash
python3 << 'PY'
import torch
import pandas as pd
import json
import networkx as nx
import os

print("=" * 60)
print("DIAGNOSTIC CHECKLIST")
print("=" * 60)

# Check 1: Graph
G = nx.read_gexf("data/processed/github_collab_graph_clean.gexf")
print(f"\n1. Original Graph:")
print(f"   ✓ Nodes: {G.number_of_nodes()}")
print(f"   ✓ Edges: {G.number_of_edges()}")

# Check 2: PyG data
try:
    d = torch.load("data/processed/graph_data.pt", weights_only=False)
    print(f"\n2. PyG Data (graph_data.pt):")
    print(f"   ✓ Nodes in data: {d['data'].x.shape[0]}")
    print(f"   ✓ Edges in index: {d['data'].edge_index.shape[1]} (expect {2*G.number_of_edges()})")
    print(f"   ✓ Nodes in list: {len(d['nodes'])}")
    if d['data'].edge_index.shape[1] == 2 * G.number_of_edges():
        print("   ✓ Bidirectional edges OK")
    else:
        print("   ✗ EDGE COUNT MISMATCH!")
except Exception as e:
    print(f"\n2. PyG Data: ✗ ERROR - {e}")

# Check 3: Baseline metrics
try:
    with open("data/processed/baseline_metrics.json") as f:
        baseline = json.load(f)
    print(f"\n3. Baseline Metrics:")
    for b in baseline['baselines']:
        print(f"   ✓ {b['method']:20s}: AUC={b['auc']:.4f}, AP={b['ap']:.4f}")
    if baseline['num_test_pos'] > 0:
        print(f"   ✓ Test edges: {baseline['num_test_pos']}")
    else:
        print(f"   ✗ NO TEST EDGES - results invalid!")
except Exception as e:
    print(f"\n3. Baseline Metrics: ✗ ERROR - {e}")

# Check 4: GAE metrics
try:
    with open("data/processed/gae_metrics.json") as f:
        gae = json.load(f)
    print(f"\n4. GAE Metrics:")
    print(f"   ✓ AUC: {gae['auc']:.4f}")
    print(f"   ✓ AP:  {gae['ap']:.4f}")
    print(f"   ✓ Epochs: {gae['epochs']}")
    print(f"   ✓ Seed: {gae.get('seed', 'not set')}")
except Exception as e:
    print(f"\n4. GAE Metrics: ✗ ERROR - {e}")

# Check 5: Output files
print(f"\n5. Output Files:")
files = [
    "data/processed/graph_data.pt",
    "data/processed/baseline_metrics.json",
    "data/processed/gae_model.pt",
    "data/processed/gae_embeddings.npy",
    "data/processed/gae_metrics.json",
    "data/processed/gae_training_logs.json",
    "data/processed/predicted_links_top50.csv",
    "data/processed/layout_positions.json",
]
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"   ✓ {os.path.basename(f):30s} ({size:,} bytes)")
    else:
        print(f"   ✗ {os.path.basename(f):30s} MISSING")

print("\n" + "=" * 60)
PY
```

**What to look for**:
- ✓ All values in reasonable ranges (0-1 for AUC/AP)
- ✓ All output files present
- ✓ Node counts match across files
- ✓ Edge counts include bidirectional (2× original)
- ✓ Test set has at least 5 edges
- ✗ Any "ERROR" or "MISSING" → investigate that section

---

## Automated Testing

Run the entire test suite with one command:

```bash
bash test_smoke.sh
```

This runs Tests 1-3 automatically with proper error handling and colored output.

---

## Next Steps

✅ **All tests pass?**
1. Open the results notebook:
   ```bash
   jupyter notebook notebooks/gae_quick_demo.ipynb
   ```

2. Inspect visualizations of:
   - Top 50 predictions
   - Training loss curves
   - Metric comparisons

3. For production, run full training:
   ```bash
   python3 scripts/train_gae.py --data-root data --epochs 200
   ```

---

## Quick Checklist

```
[ ] Test 1: prepare_pyg_data.py runs and creates graph_data.pt
[ ] Test 1: graph_data.pt validates (correct shape, num_nodes)
[ ] Test 2: baselines_link_pred.py runs and prints AUC/AP
[ ] Test 2: baseline_metrics.json contains valid scores
[ ] Test 3: train_gae.py --sample runs in < 1 minute
[ ] Test 3: All 6 GAE output files created
[ ] Test 3: gae_metrics.json has valid AUC/AP > 0.0
[ ] Test 3: predicted_links_top50.csv has 50 rows with valid scores
[ ] Comparison: GAE AUC > baseline average (typical)
[ ] Results: gae_quick_demo.ipynb loads without errors
```

---

**All passing? You're ready to use the full pipeline! 🚀**
