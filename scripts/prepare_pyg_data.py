import torch, networkx as nx, numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load Phase 1 graph
G = nx.read_gexf("data/processed/github_collab_graph_clean.gexf")
print(f"Loaded graph with {len(G.nodes())} nodes, {len(G.edges())} edges")

# 2️⃣ Create simple node features (degree + weighted degree)
nodes = list(G.nodes())
deg = dict(G.degree())
wdeg = dict(G.degree(weight="weight"))
X = np.vstack([[deg[n], wdeg[n]] for n in nodes]).astype(float)

X = StandardScaler().fit_transform(X)
x = torch.tensor(X, dtype=torch.float)

# 3️⃣ Build edge_index tensor
edge_index = torch.tensor([[nodes.index(u), nodes.index(v)] for u,v in G.edges()], dtype=torch.long).t().contiguous()

# 4️⃣ Save for GAE
data = Data(x=x, edge_index=edge_index)
torch.save({"data": data, "nodes": nodes}, "data/processed/graph_data.pt")
print("Saved data/processed/graph_data.pt")
