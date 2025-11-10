import os
import networkx as nx
import matplotlib.pyplot as plt

# Read the latest GEXF file
GEXF_PATH = os.path.join('data', 'processed', 'github_collab_graph_clean.gexf')
OUT_PNG = os.path.join('data', 'processed', 'github_collab_latest.png')
OUT_SVG = os.path.join('data', 'processed', 'github_collab_latest.svg')

print(f"Loading graph from {GEXF_PATH}")
G = nx.read_gexf(GEXF_PATH)

# Set up the figure
plt.figure(figsize=(15, 10))

# Calculate node sizes based on degree
deg = dict(G.degree(weight='weight'))
node_sizes = [100 + (deg[n] * 20) for n in G.nodes()]

# Calculate edge widths based on weights
edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
max_weight = max(edge_weights)
edge_widths = [1 + (w / max_weight) * 5 for w in edge_weights]

# Spring layout with more space between nodes
pos = nx.spring_layout(G, k=1, iterations=50)

# Draw the network
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)

# Add labels with smaller font for readability
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("GitHub Collaboration Network (Node size = weighted degree, Edge width = collaboration strength)")
plt.axis('off')

# Save both PNG and SVG
print(f"Saving to {OUT_PNG} and {OUT_SVG}")
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.savefig(OUT_SVG, bbox_inches='tight')
print("Done!")