import os
import sys
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEXF_PATH = os.path.join('data', 'processed', 'github_collab_graph.gexf')
OUT_PNG = os.path.join('data', 'processed', 'github_collab_graph.png')
OUT_SVG = os.path.join('data', 'processed', 'github_collab_graph.svg')

if not os.path.exists(GEXF_PATH):
    print(f"GEXF file not found: {GEXF_PATH}")
    sys.exit(1)

G = nx.read_gexf(GEXF_PATH)

# Convert node ids to strings (they already are) and ensure weight attribute exists
for u, v, d in G.edges(data=True):
    # networkx stores attributes as strings when writing/reading gexf in some cases
    w = d.get('weight', 1)
    try:
        d['weight'] = float(w)
    except Exception:
        d['weight'] = 1.0

# Layout
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 8))

# Node sizes scaled by degree
deg = dict(G.degree(weight='weight'))
min_deg = min(deg.values()) if deg else 1
node_sizes = [300 + (deg.get(n, 0) - min_deg) * 200 for n in G.nodes()]

# Edge widths scaled by weight
edge_weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
max_w = max(edge_weights) if edge_weights else 1.0
edge_widths = [1 + (w / max_w) * 4 for w in edge_weights]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='tab:blue', alpha=0.9)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=9)

plt.axis('off')
plt.title('GitHub Collaboration Graph', fontsize=14)
plt.tight_layout()

# ensure output dir exists
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=150)
plt.savefig(OUT_SVG)
print('Saved visualization to', OUT_PNG, 'and', OUT_SVG)
