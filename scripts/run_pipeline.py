#!/usr/bin/env python3
"""
run_pipeline.py
Single script to:
- fetch commits from a GitHub repo (cached)
- build a weighted co-edit graph (stoplist filtered)
- normalize weights for visualization
- run Louvain community detection
- export: GEXF, edges.csv, nodes.csv, summary.json
- optionally produce an interactive HTML via pyvis (if pyvis installed)
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime

try:
    from github import Github
except Exception as e:
    print("PyGithub not installed. Please run: pip install PyGithub", file=sys.stderr)
    raise

import networkx as nx
import pandas as pd
import subprocess
import sys

# Optional libs
HAS_PYVIS = False
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

# Community detection (python-louvain)
HAS_LOUVAIN = False
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False

# -----------------------------
# Configurable defaults
# -----------------------------
DEFAULT_COMMIT_LIMIT = 100
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
RAW_COMMITS_FILE = os.path.join(RAW_DIR, "commits.json")
GEXF_OUT = os.path.join(PROC_DIR, "github_collab_graph_clean.gexf")
EDGES_CSV = os.path.join(PROC_DIR, "edges.csv")
NODES_CSV = os.path.join(PROC_DIR, "nodes.csv")
SUMMARY_JSON = os.path.join(PROC_DIR, "summary.json")
PYVIS_HTML = os.path.join(PROC_DIR, "github_collab_graph.html")

# default stoplist of file names that cause noisy edges
DEFAULT_STOPLIST = {
    "readme.md", "readme", "license", "license.md",
    "package-lock.json", ".gitignore", "composer.lock"
}

# -----------------------------
# Helpers
# -----------------------------
def load_cached_commits(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_cached_commits(path, commits_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(commits_data, f, indent=2)

def fetch_commits_from_github(token, repo_name, commit_limit=DEFAULT_COMMIT_LIMIT, force_refresh=False):
    if not force_refresh:
        cached = load_cached_commits(RAW_COMMITS_FILE)
        if cached:
            print(f"Loaded {len(cached)} commits from cache: {RAW_COMMITS_FILE}")
            return cached

    if not token:
        raise RuntimeError("GITHUB token is empty. Export GITHUB_TOKEN in your shell before running.")

    print("Authenticating to GitHub...")
    g = Github(token)
    repo = g.get_repo(repo_name)
    print(f"Fetching up to {commit_limit} commits from {repo_name}...")
    commits_iter = repo.get_commits()
    commits_data = []
    count = 0
    for c in commits_iter:
        if count >= commit_limit:
            break
        # try to safely extract author, date, files (some commits may not have files attribute readily)
        try:
            author = c.author.login if c.author else None
        except Exception:
            author = None
        try:
            date = None
            if c.commit and c.commit.author and c.commit.author.date:
                date = c.commit.author.date.isoformat()
        except Exception:
            date = None
        # files: try to get filenames; if not present skip file list
        try:
            files = [f.filename for f in getattr(c, "files", [])] if getattr(c, "files", None) else []
        except Exception:
            files = []
        commits_data.append({
            "sha": c.sha,
            "author": author,
            "date": date,
            "files": files
        })
        count += 1
        if count % 25 == 0:
            print(f"  fetched {count} commits...")
    print(f"Fetched {len(commits_data)} commits.")
    save_cached_commits(RAW_COMMITS_FILE, commits_data)
    return commits_data

def build_file_to_authors(commits_data, stoplist=None):
    stoplist = set(x.lower() for x in (stoplist or []))
    file_to_authors = {}
    for c in commits_data:
        author = c.get("author")
        if not author:
            continue
        for filename in c.get("files", []):
            if not filename:
                continue
            fname = filename.strip().lower()
            if fname in stoplist:
                continue
            file_to_authors.setdefault(filename, set()).add(author)
    return file_to_authors

def build_weighted_graph(file_to_authors):
    G = nx.Graph()
    # gather all authors
    all_authors = set()
    for s in file_to_authors.values():
        all_authors.update(s)
    G.add_nodes_from(all_authors)
    # for each file, connect every pair of authors who edited it
    for authors in file_to_authors.values():
        authors = list(authors)
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                a, b = authors[i], authors[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
    return G

def normalize_weights_and_add_viz(G):
    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1.0
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1))
        d["weight_norm"] = w / max_w
        d["viz_width"] = 1 + d["weight_norm"] * 4  # width between 1 and 5

def run_louvain(G):
    if not HAS_LOUVAIN:
        print("python-louvain (community) not installed. Skip Louvain. Install with: pip install python-louvain")
        return {}
    part = community_louvain.best_partition(G, weight="weight")
    nx.set_node_attributes(G, part, "community")
    return part

def export_graph_files(G):
    os.makedirs(PROC_DIR, exist_ok=True)
    nx.write_gexf(G, GEXF_OUT)
    # edges CSV
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"u": u, "v": v, "weight": d.get("weight", 1), "weight_norm": d.get("weight_norm", 0)})
    pd.DataFrame(edges).to_csv(EDGES_CSV, index=False)
    # nodes CSV
    deg = dict(G.degree(weight="weight"))
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append({"node": n, "degree": deg.get(n, 0), "community": d.get("community")})
    pd.DataFrame(nodes).to_csv(NODES_CSV, index=False)

def save_summary(G, repo_name, commits_sampled):
    deg = dict(G.degree(weight="weight"))
    top5 = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:5]
    summary = {
        "repo": repo_name,
        "commits_sampled": commits_sampled,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "top5_degree": top5
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", SUMMARY_JSON)

def make_pyvis_html(G, out_html=PYVIS_HTML, notebook=False):
    if not HAS_PYVIS:
        print("PyVis not installed; skipping interactive HTML. (pip install pyvis)")
        return
    net = Network(height="750px", width="100%", notebook=notebook)
    # copy node attributes to pyvis nodes
    for n, d in G.nodes(data=True):
        title = f"{n}<br>degree={d.get('degree', None)}<br>community={d.get('community', None)}"
        net.add_node(n, label=n, title=title, value=d.get("degree", 1), group=d.get("community", 0))
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1), title=f"w={d.get('weight',1)}")
    net.show(out_html)
    print("Saved interactive HTML to", out_html)

# -----------------------------
# Main pipeline / CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="GitHub collaboration pipeline: fetch, build, analyze, export.")
    parser.add_argument("--repo", type=str, default="tensorflow/tensorflow", help="GitHub repo (owner/name)")
    parser.add_argument("--commit_limit", type=int, default=DEFAULT_COMMIT_LIMIT, help="Number of recent commits to sample")
    parser.add_argument("--force_refresh", action="store_true", help="Force re-fetch from GitHub (ignore cached commits)")
    parser.add_argument("--stoplist", nargs="*", default=None, help="Additional filenames to ignore (space separated)")
    parser.add_argument("--make_html", action="store_true", help="Make interactive HTML via pyvis (if installed)")
    parser.add_argument("--run_analysis", action="store_true", help="Run analysis script after pipeline (analyze_graph.py)")
    parser.add_argument("--run_visualize", action="store_true", help="Run visualization script after pipeline (visualize.py)")
    parser.add_argument("--viz-out", type=str, default=None, help="Custom output prefix for visualization (overrides default)")
    parser.add_argument("--parallel", action="store_true", help="If set, run analysis and visualization in parallel")
    parser.add_argument("--data-root", type=str, default="data", help="Data root directory (contains raw/ and processed/ subfolders)")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN env var not set. Export it and re-run.", file=sys.stderr)
        sys.exit(1)

    stoplist = set(DEFAULT_STOPLIST)
    if args.stoplist:
        stoplist.update(args.stoplist)

    # allow overriding the data directories via CLI (keeps backward compatibility)
    global RAW_DIR, PROC_DIR, RAW_COMMITS_FILE, GEXF_OUT, EDGES_CSV, NODES_CSV, SUMMARY_JSON, PYVIS_HTML
    RAW_DIR = os.path.join(args.data_root, "raw")
    PROC_DIR = os.path.join(args.data_root, "processed")
    RAW_COMMITS_FILE = os.path.join(RAW_DIR, "commits.json")
    GEXF_OUT = os.path.join(PROC_DIR, "github_collab_graph_clean.gexf")
    EDGES_CSV = os.path.join(PROC_DIR, "edges.csv")
    NODES_CSV = os.path.join(PROC_DIR, "nodes.csv")
    SUMMARY_JSON = os.path.join(PROC_DIR, "summary.json")
    PYVIS_HTML = os.path.join(PROC_DIR, "github_collab_graph.html")

    commits_data = fetch_commits_from_github(token, args.repo, commit_limit=args.commit_limit, force_refresh=args.force_refresh)
    file_to_authors = build_file_to_authors(commits_data, stoplist=stoplist)
    print(f"Found {len(file_to_authors)} unique files after stoplist filtering.")
    G = build_weighted_graph(file_to_authors)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (weighted).")

    normalize_weights_and_add_viz(G)
    # add degree attribute for convenience
    deg = dict(G.degree(weight="weight"))
    for n, d in G.nodes(data=True):
        d['degree'] = deg.get(n, 0)

    # community detection
    part = {}
    if HAS_LOUVAIN:
        part = run_louvain(G)
        print(f"Detected {len(set(part.values()))} communities (Louvain).")
    else:
        print("Louvain not available; skipping community detection. Install python-louvain to enable it.")

    export_graph_files(G)
    save_summary(G, args.repo, len(commits_data))

    if args.make_html:
        make_pyvis_html(G, out_html=PYVIS_HTML)

    print("Pipeline finished. Outputs:")
    print("  - GEXF:", GEXF_OUT)
    print("  - edges CSV:", EDGES_CSV)
    print("  - nodes CSV:", NODES_CSV)
    print("  - summary JSON:", SUMMARY_JSON)
    if args.make_html and HAS_PYVIS:
        print("  - HTML:", PYVIS_HTML)

    # Optionally run analysis and visualization scripts
    def _run_cmd(cmd):
        try:
            print("Running:", " ".join(cmd))
            res = subprocess.run(cmd, check=True)
            print("Command finished:", " ".join(cmd))
            return res.returncode
        except subprocess.CalledProcessError as e:
            print(f"Command failed ({' '.join(cmd)}): {e}", file=sys.stderr)
            return e.returncode

    procs = []
    if args.run_analysis:
        analysis_cmd = [sys.executable, os.path.join("scripts", "analyze_graph.py")]
        if args.parallel:
            p = subprocess.Popen(analysis_cmd)
            procs.append(("analysis", p))
        else:
            rc = _run_cmd(analysis_cmd)
            if rc != 0:
                print("Analysis step failed; exiting with error.", file=sys.stderr)
                sys.exit(rc)

    if args.run_visualize:
        out_prefix = args.viz_out if args.viz_out else os.path.join(PROC_DIR, "github_collab_latest")
        visualize_cmd = [sys.executable, os.path.join("scripts", "visualize.py"), "--gexf", GEXF_OUT, "--out", out_prefix, "--title", f"GitHub Collaboration Network ({args.repo})"]
        if args.parallel:
            p = subprocess.Popen(visualize_cmd)
            procs.append(("visualize", p))
        else:
            rc = _run_cmd(visualize_cmd)
            if rc != 0:
                print("Visualization step failed; exiting with error.", file=sys.stderr)
                sys.exit(rc)

    # If parallel, wait for subprocesses to finish and report
    if procs:
        print("Waiting for parallel tasks to finish...")
        failed = False
        for name, p in procs:
            ret = p.wait()
            if ret == 0:
                print(f"{name} finished successfully")
            else:
                print(f"{name} exited with code {ret}")
                failed = True
        if failed:
            print("One or more parallel tasks failed. Exiting with non-zero status.", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
