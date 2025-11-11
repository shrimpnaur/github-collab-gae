# scripts/baselines_link_pred.py
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from itertools import combinations
import random
import argparse
import os
import json
from datetime import datetime

def canon(u, v):
    """Canonicalize edge tuple for undirected graph."""
    return tuple(sorted((u, v)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data', help='Root directory for data files')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    gexf_path = os.path.join(args.data_root, 'processed', 'github_collab_graph_clean.gexf')
    G = nx.read_gexf(gexf_path)
    nodes = list(G.nodes())
    n = len(nodes)
    print(f"Loaded graph: n_nodes={n}, n_edges={G.number_of_edges()}")

    # Build positive edges (we'll treat a fraction as "future" edges by random holdout for evaluation)
    # Canonicalize all edges to sorted tuples for undirected graph consistency
    edges = [canon(u, v) for u, v in G.edges()]
    holdout_ratio = 0.2
    num_holdout = max(1, int(len(edges)*holdout_ratio))
    holdout_edges = set(random.sample(edges, num_holdout))
    train_edges = set(edges) - holdout_edges

    # Create a training-only graph
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(train_edges)

    # Create test positives and negatives
    # Use canonical form throughout for consistency
    test_pos = list(holdout_edges)
    all_pairs = [canon(u, v) for u, v in combinations(nodes, 2)]
    train_pairs = set(all_pairs) & train_edges  # Intersection of all pairs and train edges
    non_edges = [p for p in all_pairs if p not in train_pairs]
    random.shuffle(non_edges)
    test_neg = non_edges[:len(test_pos)]

    def eval_scores(y_true, y_scores, name):
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        print(f"{name}: AUC={auc:.4f}, AP={ap:.4f}")
        return {"method": name, "auc": float(auc), "ap": float(ap)}

    # Vectorized baseline scoring
    test_pairs = test_pos + test_neg
    
    # Compute all scores in batch using generators
    jacc = {canon(u, v): p for u, v, p in nx.jaccard_coefficient(G_train, ebunch=test_pairs)}
    aa = {canon(u, v): p for u, v, p in nx.adamic_adar_index(G_train, ebunch=test_pairs)}
    pa = {canon(u, v): p for u, v, p in nx.preferential_attachment(G_train, ebunch=test_pairs)}
    
    # Build score lists and labels
    jacc_scores = [jacc.get(p, 0.0) for p in test_pairs]
    aa_scores = [aa.get(p, 0.0) for p in test_pairs]
    pa_scores = [pa.get(p, 0.0) for p in test_pairs]
    
    y = [1 if p in holdout_edges else 0 for p in test_pairs]
    
    # Evaluate and collect metrics
    metrics_list = []
    metrics_list.append(eval_scores(y, jacc_scores, "Jaccard"))
    metrics_list.append(eval_scores(y, aa_scores, "Adamic-Adar"))
    metrics_list.append(eval_scores(y, pa_scores, "PreferentialAttachment"))
    
    # ---- Save baseline metrics and train/test split ----
    output_dir = os.path.join(args.data_root, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_metrics = {
        "timestamp": datetime.now().isoformat(),
        "num_nodes": n,
        "num_edges": G.number_of_edges(),
        "train_edges": len(train_edges),
        "test_pos_edges": len(test_pos),
        "test_neg_edges": len(test_neg),
        "holdout_ratio": holdout_ratio,
        "baselines": metrics_list
    }
    
    metrics_path = os.path.join(output_dir, 'baseline_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"\nSaved baseline metrics to {metrics_path}")
    
    # Save train/test edges to CSV for reproducibility
    train_edges_df = pd.DataFrame([list(e) for e in sorted(train_edges)], columns=['source', 'target'])
    train_edges_path = os.path.join(output_dir, 'train_edges.csv')
    train_edges_df.to_csv(train_edges_path, index=False)
    print(f"Saved {len(train_edges_df)} train edges to {train_edges_path}")
    
    test_edges_df = pd.DataFrame(
        [(e[0], e[1], 'positive') for e in test_pos] + 
        [(e[0], e[1], 'negative') for e in test_neg],
        columns=['source', 'target', 'label']
    )
    test_edges_path = os.path.join(output_dir, 'test_edges.csv')
    test_edges_df.to_csv(test_edges_path, index=False)
    print(f"Saved {len(test_edges_df)} test edges ({len(test_pos)} positive, {len(test_neg)} negative) to {test_edges_path}")

if __name__ == '__main__':
    main()
