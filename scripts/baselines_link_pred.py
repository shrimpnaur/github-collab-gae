# scripts/baselines_link_pred.py
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from itertools import combinations
import random

G = nx.read_gexf("data/processed/github_collab_graph_clean.gexf")
nodes = list(G.nodes())
n = len(nodes)
print(f"Loaded graph: n_nodes={n}, n_edges={G.number_of_edges()}")

# Build positive edges (we'll treat a fraction as "future" edges by random holdout for evaluation)
edges = list(G.edges())
random.seed(42)
holdout_ratio = 0.2
num_holdout = max(1, int(len(edges)*holdout_ratio))
holdout_edges = set(random.sample(edges, num_holdout))
train_edges = set(edges) - holdout_edges

# Create a training-only graph
G_train = nx.Graph()
G_train.add_nodes_from(G.nodes(data=True))
G_train.add_edges_from(train_edges)

# Create test positives and negatives
test_pos = [(u,v) for (u,v) in holdout_edges]
all_pairs = list(combinations(nodes,2))
train_pairs = set(tuple(sorted(e)) for e in G_train.edges())
non_edges = [p for p in all_pairs if tuple(sorted(p)) not in train_pairs]
random.shuffle(non_edges)
test_neg = non_edges[:len(test_pos)]

def eval_scores(y_true, y_scores, name):
    print(f"{name}: AUC={roc_auc_score(y_true,y_scores):.4f}, AP={average_precision_score(y_true,y_scores):.4f}")

# Jaccard
jacc_scores = []
y = []
for u,v in test_pos + test_neg:
    score = next(nx.jaccard_coefficient(G_train, [(u,v)]))[2]
    jacc_scores.append(score)
    y.append(1 if (u,v) in test_pos else 0)
eval_scores(y, jacc_scores, "Jaccard")

# Adamic-Adar
aa_scores = []
for u,v in test_pos + test_neg:
    score = next(nx.adamic_adar_index(G_train, [(u,v)]))[2]
    aa_scores.append(score)
eval_scores(y, aa_scores, "Adamic-Adar")

# Preferential attachment
pa_scores = []
for u,v in test_pos + test_neg:
    score = next(nx.preferential_attachment(G_train, [(u,v)]))[2]
    pa_scores.append(score)
eval_scores(y, pa_scores, "PreferentialAttachment")
