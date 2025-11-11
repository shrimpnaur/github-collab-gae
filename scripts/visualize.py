import os
import sys
import argparse
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_graph(gexf_path, out_prefix, layout_seed=42, layout_k=None, layout_iter=None, node_size_scale=20, node_size_min=100, edge_width_scale=5, title=None):
    if not os.path.exists(gexf_path):
        print(f"GEXF file not found: {gexf_path}")
        sys.exit(1)

    G = nx.read_gexf(gexf_path)

    # Ensure edge weights are floats
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1)
        try:
            d['weight'] = float(w)
        except Exception:
            d['weight'] = 1.0

    # Layout
    layout_kwargs = {'seed': layout_seed}
    if layout_k is not None:
        layout_kwargs['k'] = layout_k
    if layout_iter is not None:
        layout_kwargs['iterations'] = layout_iter
    pos = nx.spring_layout(G, **layout_kwargs)

    plt.figure(figsize=(12, 9))

    # Node sizes scaled by weighted degree
    deg = dict(G.degree(weight='weight'))
    min_deg = min(deg.values()) if deg else 1
    node_sizes = [node_size_min + (deg.get(n, 0) - min_deg) * node_size_scale for n in G.nodes()]

    # Edge widths scaled by weight
    edge_weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [1 + (w / max_w) * edge_width_scale for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='tab:blue', alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.axis('off')
    if title:
        plt.title(title, fontsize=15)
    plt.tight_layout()

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    plt.savefig(f"{out_prefix}.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{out_prefix}.svg", bbox_inches='tight')
    print(f'Saved visualization to {out_prefix}.png and {out_prefix}.svg')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize a GEXF graph with degree/weight scaling.")
    parser.add_argument('--gexf', type=str, required=True, help='Path to input GEXF file')
    parser.add_argument('--out', type=str, required=True, help='Output file prefix (no extension)')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for layout')
    parser.add_argument('--k', type=float, default=None, help='Spring layout k parameter (optional)')
    parser.add_argument('--iterations', type=int, default=None, help='Spring layout iterations (optional)')
    parser.add_argument('--node-size-scale', type=float, default=20, help='Node size scaling factor')
    parser.add_argument('--node-size-min', type=float, default=100, help='Minimum node size')
    parser.add_argument('--edge-width-scale', type=float, default=5, help='Edge width scaling factor')
    args = parser.parse_args()

    visualize_graph(
        gexf_path=args.gexf,
        out_prefix=args.out,
        layout_seed=args.seed,
        layout_k=args.k,
        layout_iter=args.iterations,
        node_size_scale=args.node_size_scale,
        node_size_min=args.node_size_min,
        edge_width_scale=args.edge_width_scale,
        title=args.title
    )

if __name__ == '__main__':
    main()
