
# GitHub Collaboration Graph — SNA, Link Prediction & GAE

This repository contains a pipeline to fetch commits from a GitHub repository, build a weighted co-edit (collaboration) graph of contributors, compute basic metrics, and train a **Graph Autoencoder (GAE)** for link prediction.

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended): `python3 -m venv venv && source venv/bin/activate`
- GitHub token for API access: export `GITHUB_TOKEN="ghp_..."`

### Installation

#### 1. Basic installation (data pipeline only)

```bash
pip install -r requirements.txt
```

#### 2. Full installation with PyG/torch (for GAE model training)

**Local (CPU or CUDA 11.8)**:

```bash
# CPU version
pip install torch==2.2.0 torchvision==0.15.2 torchaudio==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 version
pip install torch==2.2.0 torchvision==0.15.2 torchaudio==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Then install torch-geometric
pip install torch-geometric==2.3.0 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

**Google Colab**:

```python
# In a cell:
!pip install torch==2.2.0 torchvision==0.15.2 torchaudio==0.15.2
!pip install torch-geometric==2.3.0
```

(Colab has CUDA pre-installed; PyG will auto-detect it.)

### Export GitHub Token

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

## Core Workflows

### Workflow 1: Build Graph from Commits

```bash
# From repo root, fetch commits and build graph
python3 notebooks/scripts/run_pipeline.py --repo owner/repo --commit_limit 100

# Outputs to data/processed/: github_collab_graph_clean.gexf, edges.csv, nodes.csv, summary.json
```

### Workflow 2: Run Baseline Link Predictors (Jaccard, Adamic-Adar, Preferential Attachment)

```bash
# Compute simple link prediction baselines
python3 notebooks/scripts/baselines_link_pred.py --data-root data

# Outputs: baseline_metrics.json, gexf + csv files
# Example output:
# Jaccard: AUC=0.7123, AP=0.6234
# Adamic-Adar: AUC=0.7456, AP=0.6789
# PreferentialAttachment: AUC=0.6890, AP=0.5912
```

### Workflow 3: Prepare PyG Data & Train GAE

```bash
# Step 1: Convert GEXF to PyG Data format
python3 notebooks/scripts/prepare_pyg_data.py --data-root data

# Outputs: data/processed/graph_data.pt (node features, edges, PyG Data object)

# Step 2: Train GAE (200 epochs, full run; ~2-5 min on GPU)
python3 notebooks/scripts/train_gae.py --data-root data

# Or quick 5-epoch debug run:
python3 notebooks/scripts/train_gae.py --data-root data --sample

# Outputs:
# - gae_model.pt (model weights)
# - gae_embeddings.npy (learned node embeddings)
# - gae_metrics.json (AUC, AP results)
# - gae_training_logs.json (loss per epoch)
# - layout_positions.json (node coordinates for visualization)
# - predicted_links_top50.csv (top 50 predicted future edges)
# - predicted_overlay.png (visualization)
```

### Workflow 4: Compare Results

```bash
# Compare baselines vs GAE
cat data/processed/baseline_metrics.json
cat data/processed/gae_metrics.json
```

Expected comparison (example):
```json
// baseline_metrics.json
{
  "baselines": [
    {"method": "Jaccard", "auc": 0.712, "ap": 0.623},
    {"method": "Adamic-Adar", "auc": 0.746, "ap": 0.679},
    ...
  ]
}

// gae_metrics.json
{
  "auc": 0.82,
  "ap": 0.74,
  ...
}
```

## Files & Outputs

### Core Scripts

- **`notebooks/scripts/run_pipeline.py`** — Main ETL: fetch commits, build graph, compute metrics
- **`notebooks/scripts/baselines_link_pred.py`** — Baseline link prediction (Jaccard, AA, PA)
- **`notebooks/scripts/prepare_pyg_data.py`** — Convert GEXF to PyG Data format
- **`notebooks/scripts/train_gae.py`** — Train graph autoencoder for link prediction
- **`notebooks/scripts/phase1_5_enhance.py`** — Compute centralities, community detection, visualizations

### Output Files (in `data/processed/`)

| File | Description |
|------|-------------|
| `github_collab_graph_clean.gexf` | Graph in GEXF format (open in Gephi) |
| `edges.csv` | Edge list: u, v, weight, co_edit_count |
| `nodes.csv` | Node metrics: degree, weighted_degree, centrality, community |
| `summary.json` | Graph overview: num_nodes, num_edges, density, etc. |
| `baseline_metrics.json` | Jaccard/AA/PA AUC & AP scores |
| `graph_data.pt` | PyG Data object (node features, edges) |
| `gae_model.pt` | Trained GAE model weights |
| `gae_embeddings.npy` | Learned node embeddings (64-dim) |
| `gae_metrics.json` | GAE test AUC & AP |
| `gae_training_logs.json` | Loss per epoch during training |
| `layout_positions.json` | Node coordinates for consistent visualization |
| `predicted_links_top50.csv` | Top 50 predicted edges: u, v, u_idx, v_idx, score |
| `predicted_overlay.png` | Graph with predicted edges in red dashed lines |

## Command-Line Options

### `run_pipeline.py`

```bash
python3 notebooks/scripts/run_pipeline.py \
  --repo owner/repo \
  --commit_limit 100 \
  --data-root data
```

- `--repo owner/repo`: GitHub repo to analyze (required)
- `--commit_limit N`: Sample N commits (default: 100)
- `--data-root PATH`: Directory for outputs (default: `data`)

### `baselines_link_pred.py` & `prepare_pyg_data.py`

```bash
python3 notebooks/scripts/baselines_link_pred.py --data-root data
python3 notebooks/scripts/prepare_pyg_data.py --data-root data
```

- `--data-root PATH`: Data directory (default: `data`)

### `train_gae.py`

```bash
python3 notebooks/scripts/train_gae.py \
  --data-root data \
  --sample \
  --epochs 50 \
  --seed 42
```

- `--data-root PATH`: Data directory (default: `data`)
- `--sample`: Quick 5-epoch debug run (default: 200 epochs)
- `--epochs N`: Override number of epochs
- `--seed N`: Random seed for reproducibility (default: 42)

## Reproducibility

All scripts use deterministic random seeds:
- `random.seed(42)`
- `np.random.seed(42)`
- `torch.manual_seed(42)` (in GAE training)

To ensure full reproducibility across runs:

```bash
python3 notebooks/scripts/train_gae.py --data-root data --seed 42
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch_geometric'`

Install PyG (see installation section above).

### `RuntimeError: CUDA out of memory`

Try reducing batch size or use CPU:

```bash
CUDA_VISIBLE_DEVICES="" python3 notebooks/scripts/train_gae.py --data-root data
```

### Slow GitHub API (rate limiting)

Reduce `--commit_limit` or reuse cached commits at `data/raw/commits.json`.

## Next Steps

- [ ] Implement **temporal train/test split** using commit timestamps
- [ ] Add **node features** from commit metadata (author activity, file types)
- [ ] Run link prediction evaluation on **multiple random seeds** (cross-validation)
- [ ] Compare to more sophisticated baselines (e.g., Node2Vec, GraphSAGE)
- [ ] Visualize embeddings with t-SNE / UMAP

## Resources

- **NetworkX**: Graph construction & metrics — https://networkx.org
- **PyG (Torch Geometric)**: Graph neural networks — https://pytorch-geometric.readthedocs.io
- **PyGithub**: GitHub API access — https://pygithub.readthedocs.io
- Add a `Makefile` or short scripts for common commands
- Produce filtered visualizations (e.g., edges with weight >= 3) or color nodes by detected community

License
This project is intended for educational and research use. Reuse is permitted with attribution.

