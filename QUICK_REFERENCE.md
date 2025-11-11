# Quick Command Reference

## Full Pipeline (Recommended Order)

```bash
# 1. Fetch commits and build graph
python3 notebooks/scripts/run_pipeline.py --repo owner/repo --commit_limit 100 --data-root data

# 2. Prepare PyG data format
python3 notebooks/scripts/prepare_pyg_data.py --data-root data

# 3a. Run baseline methods (Jaccard, Adamic-Adar, Preferential Attachment)
python3 notebooks/scripts/baselines_link_pred.py --data-root data

# 3b. Train GAE (QUICK: 5 epochs for testing)
python3 notebooks/scripts/train_gae.py --data-root data --sample

# 3c. Train GAE (FULL: 200 epochs for production)
python3 notebooks/scripts/train_gae.py --data-root data

# 4. View results interactively
jupyter notebook notebooks/gae_quick_demo.ipynb
```

## Advanced Options

### train_gae.py

```bash
# Custom epoch count
python3 notebooks/scripts/train_gae.py --data-root data --epochs 50

# Reproducible with specific seed
python3 notebooks/scripts/train_gae.py --data-root data --seed 123

# Combine options
python3 notebooks/scripts/train_gae.py --data-root data --sample --seed 42

# Force CPU (no CUDA)
CUDA_VISIBLE_DEVICES="" python3 notebooks/scripts/train_gae.py --data-root data
```

### run_pipeline.py

```bash
# Small sample for testing
python3 notebooks/scripts/run_pipeline.py --repo owner/repo --commit_limit 10

# Large run for production
python3 notebooks/scripts/run_pipeline.py --repo oppia/oppia --commit_limit 500
```

### prepare_pyg_data.py & baselines_link_pred.py

```bash
# Change data directory
python3 notebooks/scripts/prepare_pyg_data.py --data-root notebooks/data
python3 notebooks/scripts/baselines_link_pred.py --data-root notebooks/data
```

## Output Files

After running the full pipeline, check:

```bash
# Graph metrics and baseline results
cat data/processed/baseline_metrics.json

# GAE training metrics
cat data/processed/gae_metrics.json

# Top 50 predictions
cat data/processed/predicted_links_top50.csv | head -20

# Training loss curve
cat data/processed/gae_training_logs.json | head -10
```

## View Visualizations

```bash
# Interactive plot of predictions
jupyter notebook notebooks/gae_quick_demo.ipynb

# Network graph (requires Gephi)
open data/processed/github_collab_graph_clean.gexf

# Predicted edges overlay
open data/processed/predicted_overlay.png
```

## Troubleshooting

```bash
# No CUDA available? Use CPU:
CUDA_VISIBLE_DEVICES="" python3 notebooks/scripts/train_gae.py --data-root data

# Quick test everything works?
python3 notebooks/scripts/prepare_pyg_data.py && \
python3 notebooks/scripts/baselines_link_pred.py && \
python3 notebooks/scripts/train_gae.py --sample

# Check installed versions:
python3 -c "import torch; print(f'torch: {torch.__version__}')"
python3 -c "import torch_geometric; print(f'pyg: {torch_geometric.__version__}')"
```

## Reproducible Runs

```bash
# Same seed every time (seed 42 is default)
python3 notebooks/scripts/train_gae.py --data-root data --seed 42

# Different random initialization
python3 notebooks/scripts/train_gae.py --data-root data --seed 123

# Cross-validation: run 5 times with different seeds
for seed in 42 100 200 300 400; do
  python3 notebooks/scripts/train_gae.py --data-root data --seed $seed
done
```

## Save & Compare Results

```bash
# Save baseline results
cp data/processed/baseline_metrics.json data/processed/baseline_metrics_v1.json

# Save GAE results
cp data/processed/gae_metrics.json data/processed/gae_metrics_v1.json

# Compare in notebook
# (open gae_quick_demo.ipynb and modify paths to compare multiple runs)
```
