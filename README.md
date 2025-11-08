#![brief]
# GitHub Collaboration Graph — SNA & Analysis

This repository contains a pipeline to fetch commits from a GitHub repository, build a weighted co-edit (collaboration) graph of contributors, compute basic metrics, and export files for visualization and further analysis.

Core scripts
- `notebooks/01_fetch_build_graph.py` — quick script to sample commits and build a co-edit graph (GEXF + JSON summary)
- `notebooks/run_pipeline.py` — full pipeline: fetch (cached), build graph, normalize, optional Louvain community detection, export GEXF/CSV/JSON, and optional interactive HTML (PyVis)
- `notebooks/scripts/visualize_graph.py` and `notebooks/scripts/visualize_latest.py` — create PNG/SVG visualizations from GEXF

Typical outputs (when running from `notebooks/`, these will be in `notebooks/data/processed/`)
- `github_collab_graph_clean.gexf` — graph file (open with Gephi)
- `edges.csv` — edge list with weights
- `nodes.csv` — node-level metrics (degree, community)
- `summary.json` — quick overview
- `github_collab_graph.png` / `.svg` or `github_collab_latest.png` / `.svg` — visualizations

Quick start

1) Create & activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate   # Windows
```

2) Install required packages (example):

```bash
pip install PyGithub networkx pandas matplotlib python-louvain
# optional interactive HTML
pip install pyvis jinja2
```

3) Export your GitHub token (required for API access):

```bash
export GITHUB_TOKEN="ghp_..."
```

4) Run the pipeline (from repository root):

```bash
python3 notebooks/run_pipeline.py --repo owner/repo --commit_limit 100
```

Notes
- If you run `run_pipeline.py` while located inside `notebooks/`, outputs will be created under `notebooks/data/processed/`. Paths used by scripts are relative to the working directory.
- To skip the interactive HTML (PyVis) step, omit `--make_html`.

Troubleshooting
- `ModuleNotFoundError: No module named 'github'`: install `PyGithub` in the active venv (`pip install PyGithub`).
- `AttributeError: 'NoneType' object has no attribute 'render'` when creating interactive HTML — usually a Jinja2/template issue inside PyVis. Fix by installing/upgrading Jinja2 in the venv:

```bash
pip install --upgrade jinja2 pyvis
```

- If GitHub API requests are slow or you hit rate limits, reduce `--commit_limit` or reuse cached commits saved at `notebooks/data/raw/commits.json`.

Inspecting outputs
- `notebooks/data/processed/summary.json` — quick summary
- `notebooks/data/processed/edges.csv` — edge list (open in spreadsheet software)
- `notebooks/data/processed/nodes.csv` — node metrics
- `notebooks/data/processed/*.gexf` — open with Gephi for interactive layout and exploration

Suggested next changes I can make
- Add a `requirements.txt` generated from your venv
- Add a `Makefile` or short scripts for common commands
- Produce filtered visualizations (e.g., edges with weight >= 3) or color nodes by detected community

License
This project is intended for educational and research use. Reuse is permitted with attribution.

Contributors
- Shria Nair — pipeline implementation, visualization, documentation

---

If you want, I can add a `requirements.txt` or an `examples/` section with the exact commands you used (I can auto-fill the `--repo` value based on recent runs). Tell me which you prefer.