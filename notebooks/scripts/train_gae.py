# scripts/train_gae.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import train_test_split_edges, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations

# ---- Load prepared PyG data ----
saved = torch.load("data/processed/graph_data.pt", weights_only=False)
data = saved["data"]
nodes = saved["nodes"]  # list of node names in index order
print(f"Loaded data: num_nodes={len(nodes)}, num_node_features={data.x.shape[1]}")

# ---- Split edges for link prediction (non-temporal) ----
# If you have temporal info, replace with time-based split.
data = train_test_split_edges(data)

# ---- GAE encoder ----
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

out_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAE(Encoder(data.num_features, out_dim)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# ---- Train loop ----
epochs = 200
print("Training GAE on device:", device)
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    # optional: add regularization KL for VGAE
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | loss = {loss.item():.4f}")

# ---- Evaluate ----
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)

# Use model.test if available; else compute scores manually
try:
    auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
except Exception:
    # fallback manual scoring using dot product
    def score_edge_list(z, edge_index):
        z = z.cpu().numpy()
        return np.array([np.dot(z[i], z[j]) for i, j in edge_index.t().cpu().numpy()])
    pos_scores = score_edge_list(z, data.test_pos_edge_index)
    neg_scores = score_edge_list(z, data.test_neg_edge_index)
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(y, y_score)
    ap = average_precision_score(y, y_score)

print(f"\nGAE results -> AUC: {auc:.4f}, AP: {ap:.4f}")

# ---- Produce top predicted new links (non-train pairs only) ----
# build set of train edges for filtering
train_edges = set()
train_e = data.train_pos_edge_index.cpu().t().numpy()
for u,v in train_e:
    train_edges.add((int(u), int(v)))
    train_edges.add((int(v), int(u)))

# score all non-train pairs (might be heavy if large graph)
n = len(nodes)
z_np = z.cpu().numpy()
pairs = []
scores = []
for i in range(n):
    for j in range(i+1, n):
        if (i,j) in train_edges or (j,i) in train_edges:
            continue
        s = float(np.dot(z_np[i], z_np[j]))
        pairs.append((i,j))
        scores.append(s)

# top K predictions
K = 50
order = np.argsort(-np.array(scores))[:K]
top_pairs = [pairs[i] for i in order]
top_scores = [scores[i] for i in order]

pred_df = pd.DataFrame({
    "u_idx": [p[0] for p in top_pairs],
    "v_idx": [p[1] for p in top_pairs],
    "u": [nodes[p[0]] for p in top_pairs],
    "v": [nodes[p[1]] for p in top_pairs],
    "score": top_scores
})
pred_df.to_csv("data/processed/predicted_links_top50.csv", index=False)
print(f"Saved top {K} predicted links to data/processed/predicted_links_top50.csv")

# ---- Visualize predictions overlayed on existing graph ----
# load original graph (NetworkX) to preserve layout and labels
G = nx.read_gexf("data/processed/github_collab_graph_clean.gexf")
pos = nx.spring_layout(G, seed=42)  # deterministic layout

plt.figure(figsize=(10,10))
# base graph
nx.draw_networkx_nodes(G, pos, node_size=120)
nx.draw_networkx_edges(G, pos, alpha=0.25)
# overlay predicted edges (top 50) in red dashed
pred_edges = [(nodes[i], nodes[j]) for i,j in top_pairs]
nx.draw_networkx_edges(G, pos, edgelist=pred_edges, edge_color='r', style='dashed', width=2, alpha=0.8)
# optionally label top nodes
# nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Existing collaboration graph (predicted future links in red dashed)")
plt.axis('off')
plt.tight_layout()
plt.savefig("data/processed/predicted_overlay.png", dpi=300)
print("Saved overlay image to data/processed/predicted_overlay.png")

print("\nTop 10 predicted pairs (example):")
print(pred_df.head(10).to_string(index=False))
