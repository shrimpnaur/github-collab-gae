# Copy-Paste Test Commands

These are the exact commands to run for smoke testing. Copy and paste them directly.

## Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Verify dependencies
python3 -c "import torch; print(f'torch: {torch.__version__}')"
python3 -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

---

## Test 1: Prepare PyG Data

```bash
python3 scripts/prepare_pyg_data.py --data-root data
```

**Expected output:**
```
Loaded graph with X nodes, Y edges
Saved data/processed/graph_data.pt
Graph data: num_nodes=X, num_edges=2*Y
```

---

## Test 1b: Validate PyG Data Structure

```bash
python3 << 'EOF'
import torch

d = torch.load("data/processed/graph_data.pt", weights_only=False)
print("✓ Keys:", list(d.keys()))
print("✓ Nodes:", len(d['nodes']))
print("✓ Node features shape:", d['data'].x.shape)
print("✓ Edge index shape:", d['data'].edge_index.shape)

# Assertions
assert d['data'].edge_index.shape[0] == 2
assert d['data'].x.shape[0] == len(d['nodes'])
print("\n✓ Data structure valid!")
EOF
```

---

## Test 2: Run Baselines

```bash
python3 scripts/baselines_link_pred.py --data-root data
```

**Expected output:**
```
Loaded graph: n_nodes=X, n_edges=Y
Jaccard: AUC=0.XXXX, AP=0.XXXX
Adamic-Adar: AUC=0.XXXX, AP=0.XXXX
PreferentialAttachment: AUC=0.XXXX, AP=0.XXXX

Saved baseline metrics to data/processed/baseline_metrics.json
```

---

## Test 2b: Validate Baseline Metrics

```bash
python3 << 'EOF'
import json

with open("data/processed/baseline_metrics.json") as f:
    metrics = json.load(f)

print("✓ Baselines evaluated:", len(metrics['baselines']))
for b in metrics['baselines']:
    print(f"  {b['method']:20s} AUC={b['auc']:.4f} AP={b['ap']:.4f}")
    assert 0.0 <= b['auc'] <= 1.0
    assert 0.0 <= b['ap'] <= 1.0

print("\n✓ All baseline metrics valid!")
EOF
```

---

## Test 3: Quick GAE Training (5 epochs)

```bash
python3 scripts/train_gae.py --data-root data --sample --seed 42
```

**Expected output:**
```
Training GAE: epochs=5, seed=42, data_root=data
Loaded data: num_nodes=X, num_node_features=2
Training GAE on device: cpu (or cuda)
Epoch 001 | loss = 0.XXXX
Epoch 005 | loss = 0.XXXX

GAE results -> AUC: 0.XXXX, AP: 0.XXXX
Saved top 50 predicted links to data/processed/predicted_links_top50.csv
Saved model to data/processed/gae_model.pt
Saved embeddings to data/processed/gae_embeddings.npy
Saved training logs to data/processed/gae_training_logs.json
Saved layout positions to data/processed/layout_positions.json
Saved metrics to data/processed/gae_metrics.json
```

---

## Test 3b: Validate GAE Outputs

```bash
python3 << 'EOF'
import json
import pandas as pd
import numpy as np

# Metrics
with open("data/processed/gae_metrics.json") as f:
    m = json.load(f)
print("✓ GAE Metrics:")
print(f"  AUC: {m['auc']:.4f}")
print(f"  AP:  {m['ap']:.4f}")
print(f"  Epochs: {m['epochs']}")
assert 0.0 <= m['auc'] <= 1.0
assert 0.0 <= m['ap'] <= 1.0

# Logs
with open("data/processed/gae_training_logs.json") as f:
    logs = json.load(f)
print(f"\n✓ Training Logs:")
print(f"  Epochs: {len(logs)}")
print(f"  Initial loss: {logs[0]['loss']:.6f}")
print(f"  Final loss:   {logs[-1]['loss']:.6f}")

# Embeddings
emb = np.load("data/processed/gae_embeddings.npy")
print(f"\n✓ Embeddings: {emb.shape}")

# Predictions
preds = pd.read_csv("data/processed/predicted_links_top50.csv")
print(f"\n✓ Predictions: {len(preds)} edges")
print("  Top 3:")
for _, row in preds.head(3).iterrows():
    print(f"    {row['u']:20s} <-> {row['v']:20s} score={row['score']:.4f}")

print("\n✓ All GAE outputs valid!")
EOF
```

---

## Test 4: Run All Automated Tests

```bash
bash test_smoke.sh
```

This runs Tests 1-3 with colored output and error handling.

---

## Test 5: Validate All Outputs

```bash
python3 validate_outputs.py --data-root data
```

This comprehensively checks all files and metrics.

---

## Test 6: Compare Results

```bash
python3 << 'EOF'
import json

with open("data/processed/baseline_metrics.json") as f:
    b = json.load(f)
    
with open("data/processed/gae_metrics.json") as f:
    g = json.load(f)

print("=" * 60)
print("BASELINE VS GAE COMPARISON")
print("=" * 60)
print(f"{'Method':<20} {'AUC':<10} {'AP':<10}")
print("-" * 60)

for baseline in b['baselines']:
    print(f"{baseline['method']:<20} {baseline['auc']:<10.4f} {baseline['ap']:<10.4f}")

print("-" * 60)
print(f"{'GAE':<20} {g['auc']:<10.4f} {g['ap']:<10.4f}")
print("=" * 60)
EOF
```

---

## Test 7: View Results Interactively

```bash
jupyter notebook notebooks/gae_quick_demo.ipynb
```

Then open the notebook and run all cells to see:
- Metrics comparison (bar chart)
- Top 50 predictions (interactive table)
- Training loss curves
- Score distributions

---

## Test 8: Full Training (Optional)

If smoke tests pass, run full 200-epoch training:

```bash
python3 scripts/train_gae.py --data-root data --seed 42
```

Expected time: 2-5 minutes on GPU, 10-30 minutes on CPU.

---

## Troubleshooting Commands

**Check if files exist:**
```bash
ls -lh data/processed/graph_data.pt
ls -lh data/processed/gae_*.pt
ls -lh data/processed/gae_*.json
ls -lh data/processed/predicted_links_top50.csv
```

**Check file sizes (should be reasonable):**
```bash
du -h data/processed/* | sort -h
```

**Check for errors in latest run:**
```bash
python3 scripts/train_gae.py --data-root data --sample 2>&1 | tail -20
```

**Force CPU (no CUDA):**
```bash
CUDA_VISIBLE_DEVICES="" python3 scripts/train_gae.py --data-root data --sample
```

**Force specific GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_gae.py --data-root data --sample
```

**Check GPU status:**
```bash
nvidia-smi
```

**Test with different random seed:**
```bash
python3 scripts/train_gae.py --data-root data --sample --seed 100
```

**Test with custom epoch count:**
```bash
python3 scripts/train_gae.py --data-root data --epochs 10
```

---

## Quick Checklist

Copy this checklist and mark as you go:

```
[ ] Activate venv: source venv/bin/activate
[ ] Check versions: torch and PyG installed
[ ] Test 1: prepare_pyg_data.py runs
[ ] Test 1b: graph_data.pt validates
[ ] Test 2: baselines_link_pred.py runs
[ ] Test 2b: baseline_metrics.json validates
[ ] Test 3: train_gae.py --sample runs
[ ] Test 3b: gae_*.pt and gae_*.json validate
[ ] Test 4: bash test_smoke.sh passes
[ ] Test 5: python3 validate_outputs.py passes
[ ] Test 6: Comparison shows GAE > baselines (typical)
[ ] Test 7: gae_quick_demo.ipynb loads and runs
[ ] Optional: Full training with --epochs 200
[ ] Done! 🎉
```

---

## Expected File Sizes (Approximate)

After all tests:

```
graph_data.pt              50 KB - 1 MB    (PyG Data object)
gae_model.pt              100 KB - 500 KB (Model weights)
gae_embeddings.npy        50 KB - 200 KB  (Node embeddings)
gae_metrics.json          <1 KB            (Metrics)
gae_training_logs.json    <10 KB           (Loss per epoch)
baseline_metrics.json     <1 KB            (Baseline scores)
predicted_links_top50.csv <5 KB            (Top 50 predictions)
layout_positions.json     <50 KB           (Node coordinates)
```

If files are much larger or missing, check for errors in script output.

---

## Getting Help

If tests fail:

1. **Read the error message** - it usually tells you what's wrong
2. **Check prerequisites** - venv, torch, PyG, data files
3. **Try individual tests** - isolate which part is failing
4. **Check file permissions** - `chmod 755 data/processed/`
5. **Force CPU mode** - `CUDA_VISIBLE_DEVICES="" python3 ...`
6. **Use validation script** - `python3 validate_outputs.py --data-root data`

**All tests should pass in < 10 minutes on a typical machine!** ⏱️
