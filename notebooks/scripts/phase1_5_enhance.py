#!/usr/bin/env python3
"""
scripts/phase1_5_enhance.py

Phase 1.5 enhancement script:
- Recompute Louvain communities and save as node attributes
- Compute modularity
- Compute closeness, eigenvector, betweenness, pagerank, degree
- Save updated nodes.csv
- Compute similarity baselines (Jaccard, Adamic-Adar) and save CSV
- Export adjacency, transition, distance matrices
- Run a small Girvan-Newman comparison and report modularity of splits
- Produce improved community-colored visualization (PNG & SVG)
- Update edges.csv with normalized weight
- Save summary.json

Run from project root:
    python3 scripts/phase1_5_enhance.py
"""

import os
import json
import math
from itertools import combinations
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Optional: networkx link prediction functions
from networkx.algorithms.link_prediction import jaccard_coefficient, adamic_adar_index

# python-louvain
try:
    import community as community_louvain
except Exception as e:
    raise SystemExit("python-louvain not installed. Install via: pip install python-louvain") from e

# Paths (allow overriding data root via CLI)
import argparse

parser = argparse.ArgumentParser(description='Phase1.5 enhancement: compute extra metrics and visualizations')
parser.add_argument('--data-root', type=str, default='data', help='Data root directory (contains raw/ and processed/ subfolders)')
args = parser.parse_args()

DATA_PROCESSED = os.path.join(args.data_root, 'processed')
GEXF_PATH = os.path.join(DATA_PROCESSED, 'github_collab_graph_clean.gexf')
OUT_GEXF_COMM = os.path.join(DATA_PROCESSED, 'github_collab_graph_with_communities.gexf')
NODES_CSV = os.path.join(DATA_PROCESSED, 'nodes.csv')
EDGES_CSV = os.path.join(DATA_PROCESSED, 'edges.csv')
SIM_CSV = os.path.join(DATA_PROCESSED, 'similarity_baselines.csv')
ADJ_PATH = os.path.join(DATA_PROCESSED, 'adjacency_matrix.csv')
TRANS_PATH = os.path.join(DATA_PROCESSED, 'transition_matrix.csv')
DIST_PATH = os.path.join(DATA_PROCESSED, 'distance_matrix.csv')
COMM_PLOT_PNG = os.path.join(DATA_PROCESSED, 'github_collab_community_colored.png')
COMM_PLOT_SVG = os.path.join(DATA_PROCESSED, 'github_collab_community_colored.svg')
SUMMARY_JSON = os.path.join(DATA_PROCESSED, 'summary.json')

os.makedirs(DATA_PROCESSED, exist_ok=True)

# ---------- 1. Load graph ----------
if not os.path.exists(GEXF_PATH):
    raise SystemExit(f"GEXF not found: {GEXF_PATH}\nRun the pipeline to generate it first.")

G = nx.read_gexf(GEXF_PATH)
print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Ensure weights exist as numeric
for u, v, d in G.edges(data=True):
    try:
        d['weight'] = float(d.get('weight', 1.0))
    except Exception:
        d['weight'] = 1.0

# ---------- 2. Louvain communities ----------
print("Computing Louvain partition...")
partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'community')

modularity = community_louvain.modularity(partition, G, weight='weight')
print(f"Louvain modularity Q = {modularity:.4f}")

# Save gexf with communities
nx.write_gexf(G, OUT_GEXF_COMM)
print("Saved GEXF with community attribute:", OUT_GEXF_COMM)

# ---------- 3. Centralities ----------
print("Computing centralities...")
deg = dict(G.degree(weight='weight'))
bet = nx.betweenness_centrality(G, weight='weight')
closeness = nx.closeness_centrality(G)
pagerank = nx.pagerank(G, weight='weight')

# Eigenvector: try on full graph; fallback to largest component if it fails
try:
    eigen = nx.eigenvector_centrality_numpy(G, weight='weight')
except Exception as e:
    print("Eigenvector failed on full graph, computing on largest connected component. Error:", e)
    largest_cc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(largest_cc)
    eigen_sub = nx.eigenvector_centrality_numpy(sub, weight='weight')
    eigen = {n: (eigen_sub[n] if n in eigen_sub else 0.0) for n in G.nodes()}

# Build nodes dataframe and save
nodes_df = pd.DataFrame({
    'node': list(G.nodes()),
    'degree': [deg.get(n, 0) for n in G.nodes()],
    'betweenness': [bet.get(n, 0.0) for n in G.nodes()],
    'closeness': [closeness.get(n, 0.0) for n in G.nodes()],
    'eigenvector': [eigen.get(n, 0.0) for n in G.nodes()],
    'pagerank': [pagerank.get(n, 0.0) for n in G.nodes()],
    'community': [partition.get(n, -1) for n in G.nodes()]
})
nodes_df = nodes_df.sort_values(by='degree', ascending=False)
nodes_df.to_csv(NODES_CSV, index=False)
print("Saved nodes.csv with extra centralities:", NODES_CSV)

# ---------- 4. Similarity baselines ----------
print("Computing similarity baselines (Jaccard, Adamic-Adar)...")
nodes = list(G.nodes())
# For small graphs (n ~ <200) compute all non-edge pairs; otherwise sample or use neighbor-of-neighbor heuristic
MAX_FULL_PAIRS = 250000  # safety threshold
candidate_pairs = []

# prepare all non-edge pairs
for u, v in combinations(nodes, 2):
    if not G.has_edge(u, v):
        candidate_pairs.append((u, v))

if len(candidate_pairs) > MAX_FULL_PAIRS:
    # If too many, reduce by sampling neighbors-of-neighbors
    print("Too many candidate pairs; sampling neighbor-of-neighbor candidates.")
    candidate_pairs = []
    for n in nodes:
        neighbors = set(G.neighbors(n))
        for nb in neighbors:
            for nb2 in G.neighbors(nb):
                if nb2 != n and not G.has_edge(n, nb2):
                    candidate_pairs.append(tuple(sorted((n, nb2))))
    candidate_pairs = list(dict.fromkeys(candidate_pairs))  # unique

print("Candidate pairs count:", len(candidate_pairs))

# Compute jaccard and adamic-adar for candidate pairs
jac_scores = {}
for u, v, p in jaccard_coefficient(G, ebunch=candidate_pairs):
    jac_scores[(u, v)] = p

aa_scores = {}
for u, v, p in adamic_adar_index(G, ebunch=candidate_pairs):
    aa_scores[(u, v)] = p

sim_rows = []
for (u, v) in candidate_pairs:
    key = (u, v)
    rev = (v, u)
    j = jac_scores.get(key, jac_scores.get(rev, 0.0))
    aa = aa_scores.get(key, aa_scores.get(rev, 0.0))
    sim_rows.append({'u': u, 'v': v, 'jaccard': j, 'adamic_adar': aa})

sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv(SIM_CSV, index=False)
print("Saved similarity baselines:", SIM_CSV)

# ---------- 5. Export adjacency / transition / distance matrices ----------
print("Exporting adjacency, transition, distance matrices...")
A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')  # adjacency (rows follow nodes order)
np.savetxt(ADJ_PATH, A, delimiter=',')
# Transition matrix: row-normalize adjacency
row_sums = A.sum(axis=1)
P = np.divide(A, row_sums[:, None], out=np.zeros_like(A), where=row_sums[:, None] != 0)
np.savetxt(TRANS_PATH, P, delimiter=',')

# Distance matrix (all-pairs shortest path lengths, weights as cost)
dist_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
D = np.full((len(nodes), len(nodes)), np.inf)
node_index = {n: i for i, n in enumerate(nodes)}
for u, dists in dist_dict.items():
    i = node_index[u]
    for v, dist in dists.items():
        j = node_index[v]
        D[i, j] = dist
np.savetxt(DIST_PATH, D, delimiter=',')
print("Saved matrices:", ADJ_PATH, TRANS_PATH, DIST_PATH)

# ---------- 6. Girvan-Newman comparison (first few splits) ----------
print("Running Girvan-Newman (limited levels)...")
from networkx.algorithms.community import girvan_newman
gn = girvan_newman(G)
gn_levels = []
max_levels = 3
for i, communities in enumerate(gn):
    comps = [list(c) for c in communities]
    gn_levels.append(comps)
    print(f"GN level {i+1} sizes:", [len(c) for c in comps])
    # compute modularity for this partition
    part = {}
    for cid, com in enumerate(comps):
        for node in com:
            part[node] = cid
    try:
        q_gn = community_louvain.modularity(part, G, weight='weight')
        print(f"GN level {i+1} modularity: {q_gn:.4f}")
    except Exception as e:
        print("Could not compute modularity for GN level:", e)
    if i + 1 >= max_levels:
        break

# ---------- 7. Improved visualization (clean, clear, presentation-quality) ----------
print("Creating enhanced community-colored visualization...")

import matplotlib
matplotlib.use('Agg')  # Safe for headless servers
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

# Recompute layout with better spacing
pos = nx.spring_layout(G, seed=42, k=1.5, iterations=200)

# Node and edge properties
deg_vals = dict(G.degree(weight='weight'))
communities_list = [partition.get(n, -1) for n in G.nodes()]
unique_comms = sorted(set(communities_list))
cmap = cm.get_cmap('tab20', max(1, len(unique_comms)))
comm_to_idx = {c: i for i, c in enumerate(unique_comms)}

node_colors = [cmap(comm_to_idx.get(partition.get(n, -1), 0)) for n in G.nodes()]
max_edge_w = max((d.get('weight', 1.0) for _, _, d in G.edges(data=True)), default=1.0)
edge_widths = [0.5 + (d.get('weight', 1.0) / max_edge_w) * 3 for _, _, d in G.edges(data=True)]
node_sizes = [200 + 50 * math.sqrt(deg_vals.get(n, 0)) for n in G.nodes()]

plt.figure(figsize=(14, 10))
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10})

# Draw edges (fainter for readability)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths, edge_color="gray")

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.95, linewidths=0.8, edgecolors='white')

# Label only top contributors
topk = sorted(deg_vals.items(), key=lambda x: x[1], reverse=True)[:10]
labels_to_draw = {n: n for n, _ in topk}
nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=10, font_color='black', font_weight='bold')

# Legend for communities
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Community {c}',
           markerfacecolor=cmap(i), markersize=10)
    for i, c in enumerate(unique_comms)
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True, title='Communities', fontsize=8)

# Descriptive title
plt.title('GitHub Collaboration Graph — Louvain Communities Highlighted\n'
          '(Node size = weighted degree, Edge width = collaboration strength)', fontsize=12, pad=20)

plt.axis('off')
plt.tight_layout()
plt.savefig(COMM_PLOT_PNG, dpi=300, bbox_inches='tight')
plt.savefig(COMM_PLOT_SVG, bbox_inches='tight')
plt.close()
print("Saved enhanced community-colored visualizations:", COMM_PLOT_PNG, COMM_PLOT_SVG)

# ---------- 8. Update edges.csv with normalized weight ----------
print("Updating edges.csv with normalized weight...")
edges_list = []
max_w = max((d.get('weight', 1.0) for _, _, d in G.edges(data=True)), default=1.0)
for u, v, d in G.edges(data=True):
    w = d.get('weight', 1.0)
    edges_list.append({'u': u, 'v': v, 'weight': w, 'weight_norm': (w / max_w)})
edges_df = pd.DataFrame(edges_list)
edges_df.to_csv(EDGES_CSV, index=False)
print("Saved edges.csv:", EDGES_CSV)

# ---------- 9. Save summary.json ----------
print("Writing summary.json ...")
summary = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "louvain_modularity": float(modularity),
    "top5_by_degree": nodes_df[['node', 'degree']].head(5).values.tolist()
}
with open(SUMMARY_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved summary:", SUMMARY_JSON)

print("Phase 1.5 enhancements complete.")
