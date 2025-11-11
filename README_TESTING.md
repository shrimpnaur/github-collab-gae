# 📚 All Testing Resources - Complete Index

## 🎯 Start Here

**New to testing?** Read in this order:

1. **TESTING_QUICK_START.md** ← Start with this (3 min read)
   - Quick overview
   - Single command to test everything
   - Expected results

2. **TEST_COMMANDS.md** ← Then use this (copy-paste)
   - All commands in one place
   - Copy-paste ready
   - Troubleshooting commands

3. **SMOKE_TESTS.md** ← Deep dive (20 min read)
   - Detailed explanations
   - What's being tested and why
   - Python validation snippets

---

## 📖 Documentation Files

### Core Testing Docs

| File | Purpose | Read Time | Best For |
|------|---------|-----------|----------|
| **TESTING_QUICK_START.md** | Get started fast | 3 min | New users |
| **TEST_COMMANDS.md** | Copy-paste commands | 5 min | Manual testing |
| **SMOKE_TESTS.md** | Detailed guide | 20 min | Understanding |
| **TESTING_GUIDE_SUMMARY.md** | Overview (this file) | 10 min | Navigation |

### General Documentation

| File | Purpose | Read Time |
|------|---------|-----------|
| **README.md** | Full setup & usage | 15 min |
| **QUICK_REFERENCE.md** | Command examples | 5 min |
| **IMPLEMENTATION_SUMMARY.md** | What was implemented | 10 min |
| **CHECKLIST_COMPLETE.md** | Implementation checklist | 10 min |

---

## 🛠️ Testing Tools

### Automated Testing

**test_smoke.sh** - Run everything at once
```bash
bash test_smoke.sh
```
- ✓ Runs all 3 main tests in sequence
- ✓ Colored output (green ✓, red ✗)
- ✓ Error handling
- ✓ ~3 minutes total
- ✓ Best for: Getting quick validation

**validate_outputs.py** - Comprehensive validation
```bash
python3 validate_outputs.py --data-root data
```
- ✓ Validates 7 categories of files
- ✓ Detailed error messages
- ✓ Checks metrics, shapes, ranges
- ✓ ~15 seconds
- ✓ Best for: Finding specific issues

### Manual Testing

Use **TEST_COMMANDS.md** for copy-paste commands:
- 8 different test scenarios
- Copy-paste ready
- Troubleshooting commands

---

## 📋 What Gets Tested

### Test 1: PyG Data Preparation
**File**: `prepare_pyg_data.py`
**Validates**:
- ✓ GEXF graph loads correctly
- ✓ Edge index shape is (2, 2*num_edges)
- ✓ Node features match num_nodes
- ✓ Output file created

### Test 2: Baseline Methods
**File**: `baselines_link_pred.py`
**Validates**:
- ✓ Jaccard coefficient scores
- ✓ Adamic-Adar scores
- ✓ Preferential Attachment scores
- ✓ Metrics saved to JSON
- ✓ AUC and AP are in [0.0, 1.0]

### Test 3: GAE Training
**File**: `train_gae.py --sample`
**Validates**:
- ✓ Model trains and prints loss
- ✓ Loss decreases over epochs
- ✓ Model saved
- ✓ Embeddings saved
- ✓ Metrics saved
- ✓ Predictions CSV created
- ✓ Final AUC/AP are valid

---

## 🚀 Quick Start Commands

### Fastest Path (All-in-One)
```bash
# Activate, test, and view results
source venv/bin/activate && \
bash test_smoke.sh && \
python3 validate_outputs.py && \
jupyter notebook notebooks/gae_quick_demo.ipynb
```

### Step-by-Step Manual
```bash
# 1. Prepare data (30 sec)
python3 scripts/prepare_pyg_data.py --data-root data

# 2. Run baselines (30 sec)
python3 scripts/baselines_link_pred.py --data-root data

# 3. Train GAE quick (< 1 min)
python3 scripts/train_gae.py --data-root data --sample --seed 42

# 4. Validate everything (15 sec)
python3 validate_outputs.py --data-root data

# 5. View results in Jupyter
jupyter notebook notebooks/gae_quick_demo.ipynb
```

---

## 📊 Expected Results

### Baseline Scores
```
Jaccard:               AUC=0.70-0.80, AP=0.60-0.75
Adamic-Adar:           AUC=0.70-0.80, AP=0.60-0.75
Preferential Attach:   AUC=0.65-0.75, AP=0.55-0.70
```

### GAE Performance
```
AUC: 0.75-0.85  (typically better than baselines)
AP:  0.70-0.80
```

### Output Files (8 total)
```
✓ graph_data.pt (50 KB - 1 MB)
✓ baseline_metrics.json (<1 KB)
✓ gae_model.pt (100 KB - 500 KB)
✓ gae_embeddings.npy (50 KB - 200 KB)
✓ gae_metrics.json (<1 KB)
✓ gae_training_logs.json (<10 KB)
✓ predicted_links_top50.csv (<5 KB)
✓ layout_positions.json (<50 KB)
```

---

## ❓ Troubleshooting

### Problem: Test fails, need help
**Solution**:
1. Run: `python3 validate_outputs.py` to identify problem
2. Look up specific test in `SMOKE_TESTS.md`
3. Try solution in "Troubleshooting" section
4. If still stuck, check `TEST_COMMANDS.md` for the command

### Problem: Don't know which test to run
**Solution**:
1. Read: `TESTING_QUICK_START.md` (3 min)
2. Run: `bash test_smoke.sh` (automatic)
3. Done! ✓

### Problem: Want to understand everything
**Solution**:
1. Read: `SMOKE_TESTS.md` (detailed, 20 min)
2. Run: `TEST_COMMANDS.md` manually (learn each step)
3. Master! ✓

### Problem: Out of memory / slow
**Solution**:
- GPU: `CUDA_VISIBLE_DEVICES=0 python3 scripts/train_gae.py ...`
- CPU: Already using CPU, can't speed up further
- Skip: Use `--sample` flag for 5-epoch quick test

---

## 📚 Reading Guide

### For Impatient Users (5 minutes)
1. TESTING_QUICK_START.md
2. `bash test_smoke.sh`
3. Done!

### For Careful Users (20 minutes)
1. TESTING_QUICK_START.md (3 min)
2. SMOKE_TESTS.md (20 min)
3. Run tests manually using TEST_COMMANDS.md (5 min)

### For Complete Understanding (45 minutes)
1. README.md (15 min)
2. TESTING_QUICK_START.md (3 min)
3. SMOKE_TESTS.md (20 min)
4. Run all tests and inspect outputs (5 min)

---

## ✅ Success Metrics

### You'll know it's working when:

```
✓ prepare_pyg_data.py creates graph_data.pt
✓ graph_data.pt validates (correct shape)
✓ baselines_link_pred.py prints 3 baseline scores
✓ baseline_metrics.json has valid AUC/AP
✓ train_gae.py --sample completes in < 1 min
✓ Loss decreases over 5 epochs
✓ All 6 GAE output files created
✓ gae_metrics.json has AUC > 0.70, AP > 0.60
✓ predict_links_top50.csv has 50 rows
✓ validate_outputs.py passes all checks ← This is the key check!
✓ jupyter notebook loads without errors
✓ Can see baseline vs GAE comparison
```

**If all ✓, you're done!** 🎉

---

## 🔗 Document Links & Navigation

### Quick Navigation
- **Just want to test?** → TESTING_QUICK_START.md
- **Copy-paste commands?** → TEST_COMMANDS.md
- **Understand deeply?** → SMOKE_TESTS.md
- **Full setup guide?** → README.md
- **Command examples?** → QUICK_REFERENCE.md

### By Use Case

**"I just deployed the code"**
```
1. Read: TESTING_QUICK_START.md
2. Run: bash test_smoke.sh
3. Done!
```

**"I want to validate manually"**
```
1. Read: TEST_COMMANDS.md
2. Copy-paste each command
3. Inspect outputs
```

**"I need to understand the pipeline"**
```
1. Read: README.md
2. Read: SMOKE_TESTS.md
3. Run: Test suite and inspect each file
```

**"Something is broken"**
```
1. Run: python3 validate_outputs.py
2. Find which test failed
3. Look up that test in SMOKE_TESTS.md
4. Follow troubleshooting steps
```

---

## 📝 File Checklist

After running all tests, you should have:

### Input Files (must exist)
```
✓ data/processed/github_collab_graph_clean.gexf
✓ data/processed/edges.csv
✓ data/processed/nodes.csv
```

### Output Files (created by tests)
```
✓ data/processed/graph_data.pt (prepare_pyg_data.py)
✓ data/processed/baseline_metrics.json (baselines_link_pred.py)
✓ data/processed/gae_model.pt (train_gae.py)
✓ data/processed/gae_embeddings.npy (train_gae.py)
✓ data/processed/gae_metrics.json (train_gae.py)
✓ data/processed/gae_training_logs.json (train_gae.py)
✓ data/processed/predicted_links_top50.csv (train_gae.py)
✓ data/processed/layout_positions.json (train_gae.py)
```

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Read TESTING_QUICK_START.md | 3 min |
| Read TEST_COMMANDS.md | 5 min |
| Read SMOKE_TESTS.md | 20 min |
| Run bash test_smoke.sh | 3 min |
| Run validate_outputs.py | 15 sec |
| **Total (quick path)** | **~10 min** |
| **Total (full path)** | **~45 min** |

---

## 🎯 Recommended Path for You

**I recommend this path:**

1. **Now** (3 min):
   - Read: TESTING_QUICK_START.md
   
2. **Then** (3 min):
   - Run: `bash test_smoke.sh`
   
3. **Next** (1 min):
   - Check: All tests passed?
   - YES → Go to step 4
   - NO → Run: `python3 validate_outputs.py` and fix

4. **Finally** (5 min):
   - View: `jupyter notebook notebooks/gae_quick_demo.ipynb`
   - See the results!

**Total time: ~15 minutes to go from nothing to working system!** ⚡

---

## 🤝 Need Help?

### Quick Questions
- Read: `TESTING_QUICK_START.md` (likely answers there)
- See: Troubleshooting section in that file

### Specific Test Issues
- Read: `SMOKE_TESTS.md` (most detailed)
- Run: `python3 validate_outputs.py` (identifies problem)
- Look up that test section in SMOKE_TESTS.md

### Want to Understand Everything
- Read: All docs in order (README → TESTING_QUICK_START → SMOKE_TESTS)
- Run: All tests manually using TEST_COMMANDS.md
- You'll be an expert! 🚀

---

## 📞 Support Resources

| Question | Resource |
|----------|----------|
| "How do I start?" | TESTING_QUICK_START.md |
| "What command do I run?" | TEST_COMMANDS.md |
| "Why did test X fail?" | SMOKE_TESTS.md (troubleshooting) |
| "How do I set up?" | README.md |
| "What is this project?" | IMPLEMENTATION_SUMMARY.md |
| "What was implemented?" | CHECKLIST_COMPLETE.md |

---

**Ready to test? Start here:**
```bash
cd /home/shria-nair/Documents/college/github-collab-gae
cat TESTING_QUICK_START.md
bash test_smoke.sh
```

**You got this! 🚀**
