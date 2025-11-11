# 📋 Complete Testing Documentation Summary

This directory now includes comprehensive testing resources:

## Testing Documents

### 1. **TESTING_QUICK_START.md** ⭐ START HERE
- **Time**: 3 minutes to read
- **For**: Anyone wanting to quickly test the pipeline
- TL;DR: Run `bash test_smoke.sh` and `jupyter notebook`
- Includes: Step-by-step guide, expected results, troubleshooting

### 2. **TEST_COMMANDS.md**
- **Time**: 5 minutes to copy-paste
- **For**: Manual testing, copy-paste friendly
- Includes: Every command you need, no guessing
- Format: Copy-paste blocks for each test

### 3. **SMOKE_TESTS.md**
- **Time**: 20 minutes to read thoroughly
- **For**: Deep understanding of what's being tested
- Includes: Detailed explanations, validation logic, troubleshooting
- Format: Step-by-step with Python validation snippets

### 4. **README.md** (updated)
- **Time**: 15 minutes to read
- **For**: Full setup and usage documentation
- Includes: Installation, 4 workflows, CLI reference, output files
- Format: Professional documentation with examples

---

## Testing Tools

### 1. **test_smoke.sh** (Automated)
```bash
bash test_smoke.sh
```
- Runs all tests in sequence
- Colored output (✓ pass, ✗ fail)
- Error handling and validation
- ~3 minutes total

**Tests**:
1. ✓ Virtual environment
2. ✓ Input files
3. ✓ prepare_pyg_data.py
4. ✓ graph_data.pt structure
5. ✓ baselines_link_pred.py
6. ✓ baseline_metrics.json
7. ✓ train_gae.py --sample
8. ✓ All GAE output files
9. ✓ Metrics validation
10. ✓ Predictions validation

### 2. **validate_outputs.py** (Comprehensive)
```bash
python3 validate_outputs.py --data-root data
```
- Validates every output file
- Detailed error messages
- 7 validation checks:
  1. PyG data structure
  2. Baseline metrics (AUC/AP)
  3. GAE metrics (AUC/AP)
  4. Training logs
  5. Embeddings
  6. Predictions CSV
  7. Model state dict

---

## Quick Reference

### All-in-One Test (3 minutes)
```bash
source venv/bin/activate
bash test_smoke.sh
python3 validate_outputs.py --data-root data
jupyter notebook notebooks/gae_quick_demo.ipynb
```

### Individual Tests (Manual)
```bash
# Test 1: Prepare PyG data
python3 scripts/prepare_pyg_data.py --data-root data

# Test 2: Run baselines
python3 scripts/baselines_link_pred.py --data-root data

# Test 3: Train GAE (quick)
python3 scripts/train_gae.py --data-root data --sample --seed 42

# Test 4: Validate outputs
python3 validate_outputs.py --data-root data

# Test 5: Full training (optional)
python3 scripts/train_gae.py --data-root data --epochs 200
```

---

## What Gets Tested

### ✅ Functionality Tests
- PyG data preparation (prepare_pyg_data.py)
- Edge canonicalization (baselines_link_pred.py)
- Baseline scoring (Jaccard, AA, PA)
- GAE training and inference
- Model artifact saving

### ✅ Output Validation
- File existence and size
- Data structure correctness
- Metric ranges (0.0-1.0 for AUC/AP)
- Array shapes and dtypes
- CSV format and columns

### ✅ Reproducibility Tests
- Random seed control
- Deterministic outputs
- Consistent metrics across runs

---

## Expected Results

### Baseline Metrics
```
Jaccard:               AUC=0.70-0.80, AP=0.60-0.75
Adamic-Adar:           AUC=0.70-0.80, AP=0.60-0.75
Preferential Attach:   AUC=0.65-0.75, AP=0.55-0.70
```

### GAE Metrics
```
AUC: 0.75-0.85  (typically better than baselines)
AP:  0.70-0.80
```

### Files Created
```
graph_data.pt              (50 KB - 1 MB)
baseline_metrics.json      (<1 KB)
gae_model.pt               (100 KB - 500 KB)
gae_embeddings.npy         (50 KB - 200 KB)
gae_metrics.json           (<1 KB)
gae_training_logs.json     (<10 KB)
predicted_links_top50.csv  (<5 KB)
layout_positions.json      (<50 KB)
```

---

## Choosing Your Path

### Path 1: "Just test it!" ⚡
1. Read: `TESTING_QUICK_START.md` (2 min)
2. Run: `bash test_smoke.sh` (3 min)
3. Done! ✓

### Path 2: "I want to understand it" 🔬
1. Read: `SMOKE_TESTS.md` (20 min)
2. Run: `TEST_COMMANDS.md` manually (10 min)
3. Run: `validate_outputs.py` (1 min)
4. Understand: Full knowledge of what's tested ✓

### Path 3: "Full documentation" 📚
1. Read: `README.md` (15 min)
2. Read: `SMOKE_TESTS.md` (20 min)
3. Read: `QUICK_REFERENCE.md` (5 min)
4. Run: All tools and commands
5. Complete mastery! ✓

---

## Troubleshooting Flow

```
Does bash test_smoke.sh pass?
├─ YES → You're done! 🎉
└─ NO → Run validate_outputs.py
         │
         ├─ PyG Data failed?
         │  └─ Run: prepare_pyg_data.py --data-root data
         │
         ├─ Baseline Metrics failed?
         │  └─ Run: baselines_link_pred.py --data-root data
         │
         ├─ GAE Metrics failed?
         │  └─ Run: train_gae.py --data-root data --sample
         │
         └─ Still failing?
            └─ Read: SMOKE_TESTS.md troubleshooting section
```

---

## Key Testing Metrics

| Metric | Meaning | Pass Criteria |
|--------|---------|---------------|
| AUC | Area under ROC curve | 0.0-1.0 (0.5 = random) |
| AP | Average Precision | 0.0-1.0 (0.0 = worst, 1.0 = best) |
| Loss | Reconstruction loss | Decreases over epochs |
| Edge count | Undirected edges | Should be 2 × original |
| Features | Node features | Should match num_nodes |

---

## Files in This Testing Suite

```
test_smoke.sh           (Executable script, runs all tests)
validate_outputs.py     (Python script, validates files/metrics)

TESTING_QUICK_START.md  (Start here - TL;DR)
TEST_COMMANDS.md        (Copy-paste commands)
SMOKE_TESTS.md          (Detailed explanations)
TESTING_GUIDE_SUMMARY.md (This file)

README.md               (Full documentation)
QUICK_REFERENCE.md     (Command examples)
SMOKE_TESTS.md          (Detailed testing guide)
```

---

## Success Checklist

- [ ] venv activated and dependencies installed
- [ ] Graph data files (GEXF, CSV) exist in data/processed/
- [ ] prepare_pyg_data.py runs and creates graph_data.pt
- [ ] graph_data.pt validates (correct shape, num_nodes)
- [ ] baselines_link_pred.py runs and prints AUC/AP
- [ ] baseline_metrics.json contains valid scores
- [ ] train_gae.py --sample runs in < 1 minute
- [ ] All 6 GAE output files created
- [ ] gae_metrics.json has valid AUC/AP > 0.0
- [ ] predicted_links_top50.csv has 50 rows
- [ ] validate_outputs.py passes all 7 checks
- [ ] jupyter notebook loads without errors
- [ ] Can compare baseline vs GAE results
- [ ] Ready for full training (optional)

**If all ✓, you're ready to deploy!** 🚀

---

## Getting Help

### Quick Questions
- See: `TESTING_QUICK_START.md`
- Run: `bash test_smoke.sh` to identify the issue

### Specific Tests
- See: `TEST_COMMANDS.md` for exact commands
- See: `SMOKE_TESTS.md` for detailed explanations

### Full Understanding
- Read: `README.md` for complete documentation
- Run: `validate_outputs.py` for comprehensive validation

### Still Stuck?
1. Run: `python3 validate_outputs.py` to identify problem area
2. Check: Error message and suggested fix
3. See: `SMOKE_TESTS.md` troubleshooting section

---

## Performance Notes

| Component | Time | Hardware |
|-----------|------|----------|
| prepare_pyg_data.py | ~30 sec | CPU |
| baselines_link_pred.py | ~30 sec | CPU |
| train_gae.py --sample | <1 min | CPU |
| train_gae.py (200 epochs) | 2-5 min | GPU |
| train_gae.py (200 epochs) | 10-30 min | CPU |
| validate_outputs.py | ~15 sec | CPU |

**Total smoke test time: ~3 minutes** ⏱️

---

## Document Versions

```
TESTING_QUICK_START.md      ← Quick start (easiest)
TEST_COMMANDS.md             ← Copy-paste (fastest)
SMOKE_TESTS.md               ← Detailed (most thorough)
README.md                    ← Full docs (most complete)
TESTING_GUIDE_SUMMARY.md     ← This file (overview)
```

**Recommended reading order:**
1. TESTING_QUICK_START.md (2 min)
2. SMOKE_TESTS.md (20 min) OR TEST_COMMANDS.md (5 min)
3. README.md (15 min)

---

## Commands You'll Use

```bash
# Activate environment
source venv/bin/activate

# Run all tests (RECOMMENDED)
bash test_smoke.sh

# Validate everything
python3 validate_outputs.py --data-root data

# View results
jupyter notebook notebooks/gae_quick_demo.ipynb

# Full training (optional)
python3 scripts/train_gae.py --data-root data
```

---

**Status**: ✅ Complete testing infrastructure  
**Quality**: Production-ready with comprehensive validation  
**Time to test**: ~3 minutes for smoke tests  
**Time to understand**: 20-30 minutes reading all docs  

**Now run:**
```bash
source venv/bin/activate && bash test_smoke.sh
```

**Questions? See TESTING_QUICK_START.md** 📖
