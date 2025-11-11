import torch
import networkx as nx
import numpy as np
import argparse
import os
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data', help='Root directory for data files')
    args = parser.parse_args()
    
    # 1️⃣ Load Phase 1 graph
    gexf_path = os.path.join(args.data_root, 'processed', 'github_collab_graph_clean.gexf')
    G = nx.read_gexf(gexf_path)
    print(f"Loaded graph with {len(G.nodes())} nodes, {len(G.edges())} edges")

    # 2️⃣ Create simple node features (degree + weighted degree)
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    X = np.vstack([[deg[n], wdeg[n]] for n in nodes]).astype(float)

    X = StandardScaler().fit_transform(X)
    x = torch.tensor(X, dtype=torch.float)

    # 3️⃣ Build edge_index tensor with both directions (undirected graph)
    edges_idx = []
    for u, v in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        edges_idx.append((i, j))
        edges_idx.append((j, i))  # Add both directions for undirected graph
    
    edge_index = torch.tensor(edges_idx, dtype=torch.long).t().contiguous()
    # Alternative: edge_index = to_undirected(edge_index)

    # 4️⃣ Save for GAE
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = len(nodes)
    
    output_dir = os.path.join(args.data_root, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'graph_data.pt')
    torch.save({"data": data, "nodes": nodes}, output_path)
    print(f"Saved {output_path}")
    print(f"Graph data: num_nodes={data.num_nodes}, num_edges={data.edge_index.shape[1]}")

if __name__ == '__main__':
    main()
