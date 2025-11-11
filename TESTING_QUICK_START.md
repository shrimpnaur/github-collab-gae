# 🚀 Quick Start Testing Guide

**TL;DR**: Run these 3 commands in order to test everything:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run all tests automatically
bash test_smoke.sh

# 3. View results
jupyter notebook notebooks/gae_quick_demo.ipynb
```

---

## What Gets Tested

The smoke test suite validates:

| Test | Command | Purpose | Expected Output |
|------|---------|---------|-----------------|
| **1** | `prepare_pyg_data.py` | Convert GEXF to PyG format | ✓ Creates `graph_data.pt` |
| **2** | `baselines_link_pred.py` | Run baseline methods | ✓ Prints AUC/AP for 3 methods |
| **3** | `train_gae.py --sample` | Train GAE for 5 epochs | ✓ Prints loss, saves metrics |
| **4** | `validate_outputs.py` | Check all files/metrics | ✓ All validations pass |

---

## Manual Testing (Step-by-Step)

If you prefer to run tests manually:

### Step 1: Prepare PyG Data (30 seconds)

```bash
python3 scripts/prepare_pyg_data.py --data-root data
```

✅ **Success**: File `data/processed/graph_data.pt` created

### Step 2: Run Baselines (30 seconds)

```bash
python3 scripts/baselines_link_pred.py --data-root data
```

✅ **Success**: Prints 3 baseline scores (Jaccard, Adamic-Adar, PA)

Example output:
```
Jaccard: AUC=0.7234, AP=0.6456
Adamic-Adar: AUC=0.7567, AP=0.6789
PreferentialAttachment: AUC=0.6890, AP=0.5923
```

### Step 3: Quick GAE Training (< 1 minute)

```bash
python3 scripts/train_gae.py --data-root data --sample --seed 42
```

✅ **Success**: 
- Prints loss every epoch
- Final AUC/AP (non-zero)
- Saves 6 files

Example output:
```
Epoch 001 | loss = 0.1234
Epoch 005 | loss = 0.0987

GAE results -> AUC: 0.7834, AP: 0.7123
```

### Step 4: Validate Everything (15 seconds)

```bash
python3 validate_outputs.py --data-root data
```

✅ **Success**: All 7 validations pass

Example output:
```
✓ PyG Data
✓ Baseline Metrics
✓ GAE Metrics
✓ Training Logs
✓ Embeddings
✓ Predictions
✓ Model
```

### Step 5: View Results

```bash
jupyter notebook notebooks/gae_quick_demo.ipynb
```

✅ **Success**: Notebook loads, visualizations show metrics

---

## Expected Results

After all tests, you should see:

**Baseline Metrics** (from `baseline_metrics.json`):
```json
{
  "baselines": [
    {"method": "Jaccard", "auc": 0.70-0.80, "ap": 0.60-0.75},
    {"method": "Adamic-Adar", "auc": 0.70-0.80, "ap": 0.60-0.75},
    {"method": "PreferentialAttachment", "auc": 0.65-0.75, "ap": 0.55-0.70}
  ]
}
```

**GAE Metrics** (from `gae_metrics.json`):
```json
{
  "auc": 0.75-0.85,
  "ap": 0.70-0.80,
  "epochs": 5,
  "device": "cpu" or "cuda"
}
```

**Typical**: GAE AUC > Baseline AUC ✓

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | Run: `pip install -r requirements.txt` |
| `FileNotFoundError: graph_data.pt` | Run: `prepare_pyg_data.py` first |
| `NaN` loss in GAE training | Ensure edge_index shape is (2, 2*num_edges) |
| Very low AUC/AP (~0.5) | Graph may be too small; use more commits |
| Out of memory | Use `--sample` flag or CPU: `CUDA_VISIBLE_DEVICES="" python3 ...` |

---

## File Structure After Tests

```
data/processed/
├── github_collab_graph_clean.gexf  ← Input (from run_pipeline.py)
├── edges.csv                        ← Input (from run_pipeline.py)
├── nodes.csv                        ← Input (from run_pipeline.py)
├── graph_data.pt                    ← Created by prepare_pyg_data.py
├── baseline_metrics.json            ← Created by baselines_link_pred.py
├── gae_model.pt                     ← Created by train_gae.py
├── gae_embeddings.npy               ← Created by train_gae.py
├── gae_metrics.json                 ← Created by train_gae.py
├── gae_training_logs.json           ← Created by train_gae.py
├── predicted_links_top50.csv        ← Created by train_gae.py
└── layout_positions.json            ← Created by train_gae.py
```

---

## Time Estimates

| Test | Time |
|------|------|
| Prepare PyG Data | ~30 sec |
| Baselines | ~30 sec |
| GAE --sample (5 epochs) | <1 min |
| Validation | ~15 sec |
| **Total** | **~3 minutes** |

Optional: Full GAE training (200 epochs) = 2-5 min on GPU, 10-30 min on CPU

---

## Next Steps

✅ **All tests pass?**

1. **View results**:
   ```bash
   jupyter notebook notebooks/gae_quick_demo.ipynb
   ```

2. **Full training** (optional):
   ```bash
   python3 scripts/train_gae.py --data-root data --epochs 200
   ```

3. **Try different seeds** for reproducibility:
   ```bash
   python3 scripts/train_gae.py --data-root data --sample --seed 100
   python3 scripts/train_gae.py --data-root data --sample --seed 200
   ```

4. **Analyze results**:
   - Compare baseline vs GAE in the notebook
   - Inspect top 50 predictions
   - Review training curves

---

## Common Questions

**Q: Do I need CUDA?**  
A: No, CPU works fine for testing. For full runs, GPU is ~10x faster.

**Q: What if I get "NaN" loss?**  
A: Check that edge_index shape is (2, 2*E). Run: `python3 validate_outputs.py`

**Q: Can I skip the baseline test?**  
A: No - it's quick and validates the canonicalization fix.

**Q: What are realistic AUC/AP values?**  
A: AUC 0.65-0.85, AP 0.60-0.80 (depending on graph size/structure)

**Q: How big should the graph be?**  
A: Minimum 10 nodes, 20 edges. Optimal: 50+ nodes, 100+ edges.

---

## Files You Should Read

1. **TEST_COMMANDS.md** - Copy-paste commands for each test
2. **SMOKE_TESTS.md** - Detailed explanation of each test
3. **README.md** - Full setup and usage guide
4. **QUICK_REFERENCE.md** - Command examples

---

**Ready to start? Run:**

```bash
source venv/bin/activate && bash test_smoke.sh
```

**Got questions? Check:**
```
SMOKE_TESTS.md    (detailed explanations)
TEST_COMMANDS.md  (copy-paste commands)
README.md         (full setup guide)
```

**Everything working? Celebrate! 🎉 You're ready to use the full pipeline.**
