# scripts/prepare_pyg_data.py
"""
Prepare PyG Data and deterministic train/val/test splits for link prediction.

Usage:
  # If you already downloaded/unzipped a Kaggle dataset to data/raw/<dataset>:
  python scripts/prepare_pyg_data.py --raw data/raw/rozemberczki_musae-github-social-network --out data/processed --seed 42

  # Or point to a folder that contains edges.csv / nodes.csv:
  python scripts/prepare_pyg_data.py --raw path/to/folder --out data/processed

Output:
  - data/processed/graph_data_full.pt  (original full graph + nodes)
  - data/processed/graph_data.pt       (train graph + split edge indices + negatives)
"""

import os
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.utils import remove_self_loops, add_self_loops

def read_edges_and_nodes(raw_dir: str) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor]]:
    """
    Try to find edges and nodes in raw_dir.
    Returns (nodes_list, edge_index_tensor (2, E), node_features_tensor or None)
    """
    files = os.listdir(raw_dir)
    # find edges
    edges_candidates = [f for f in files if f.lower().startswith("edge") or f.lower().endswith(".edges") or f.lower().endswith("_edges.csv") or f.lower()=="edges.csv"]
    nodes_candidates = [f for f in files if f.lower().startswith("node") or f.lower().endswith("_nodes.csv") or f.lower()=="nodes.csv"]
    # also allow common file names
    if not edges_candidates:
        # sometimes file named 'musae_github_edgelist.txt' etc
        edges_candidates = [f for f in files if 'edge' in f.lower() or 'edgelist' in f.lower()]
    if not edges_candidates:
        raise FileNotFoundError(f"No edges file found in {raw_dir}. Files: {files}")

    edges_path = os.path.join(raw_dir, edges_candidates[0])
    # read edges with pandas; be permissive about delimiter
    try:
        edges_df = pd.read_csv(edges_path)
    except Exception:
        edges_df = pd.read_csv(edges_path, sep=None, engine='python')

    # take first two columns as source, target
    src_col = edges_df.columns[0]
    dst_col = edges_df.columns[1]
    src = edges_df[src_col].astype(str).tolist()
    dst = edges_df[dst_col].astype(str).tolist()

    if nodes_candidates:
        nodes_path = os.path.join(raw_dir, nodes_candidates[0])
        try:
            nodes_df = pd.read_csv(nodes_path)
        except Exception:
            nodes_df = pd.read_csv(nodes_path, sep=None, engine='python')
        # try to find id/name column
        if 'id' in nodes_df.columns:
            node_ids = nodes_df['id'].astype(str).tolist()
        elif 'node' in nodes_df.columns:
            node_ids = nodes_df['node'].astype(str).tolist()
        elif 'name' in nodes_df.columns:
            node_ids = nodes_df['name'].astype(str).tolist()
        else:
            node_ids = nodes_df.iloc[:, 0].astype(str).tolist()
        # try numeric features
        numeric_cols = nodes_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            x = torch.tensor(nodes_df[numeric_cols].fillna(0).values, dtype=torch.float)
        else:
            x = None
    else:
        # infer nodes from edges
        unique_nodes = pd.Index(src + dst).unique()
        node_ids = list(unique_nodes.astype(str))
        x = None

    # build mapping and edge_index
    node_id_map = {n: i for i, n in enumerate(node_ids)}
    edge_list = []
    for s, d in zip(src, dst):
        if s not in node_id_map or d not in node_id_map:
            # skip unknowns (possible if node list uses different ids)
            continue
        u = node_id_map[s]
        v = node_id_map[d]
        # skip self-loops at this stage
        if u == v:
            continue
        edge_list.append([u, v])

    if len(edge_list) == 0:
        raise RuntimeError("No edges mapped after reading CSVs. Check formats and node ids.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)  # ensure undirected
    # remove possible self-loops and duplicates handled by to_undirected
    edge_index, _ = remove_self_loops(edge_index)

    return node_ids, edge_index, x

def build_pyg_data(node_ids: List[str], edge_index: torch.Tensor, x: Optional[torch.Tensor]) -> Data:
    """
    Build a PyG Data object; if x is None, create dummy 1-d features of ones.
    """
    if x is None:
        x = torch.ones((len(node_ids), 1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

def undirected_unique_edge_list(edge_index: torch.Tensor) -> List[Tuple[int,int]]:
    e = edge_index.cpu().numpy().T
    s = set()
    for u,v in e:
        a,b = int(u), int(v)
        if a == b:
            continue
        if a <= b:
            s.add((a,b))
        else:
            s.add((b,a))
    return list(s)

def make_splits(edge_index: torch.Tensor, num_nodes: int, val_frac: float, test_frac: float, seed: int=42, neg_sample_ratio: int=1) -> Dict[str, torch.Tensor]:
    """
    Deterministically split edges into train/val/test positive edges, produce negative samples
    for each of val/test and train (optional). Returns dictionary of tensors (each is 2 x M).
    - neg_sample_ratio: how many negative samples per positive sample (usually 1)
    """
    rng = np.random.RandomState(seed)
    unique_edges = undirected_unique_edge_list(edge_index)
    m = len(unique_edges)
    if m == 0:
        raise RuntimeError("Graph has 0 edges!")

    # shuffle deterministic
    perm = rng.permutation(m)
    unique_edges = [unique_edges[i] for i in perm]

    n_test = max(1, int(round(m * test_frac)))
    n_val = max(1, int(round(m * val_frac)))
    # ensure not overlapping and some left for train
    if n_test + n_val >= m:
        raise ValueError("Graph too small for given val/test fractions.")

    test_edges = unique_edges[:n_test]
    val_edges = unique_edges[n_test:n_test + n_val]
    train_edges = unique_edges[n_test + n_val:]

    # build torch edge_index tensors (train graph must contain only train_edges duplicated for undirected)
    def to_edge_index(edges_list):
        if len(edges_list) == 0:
            return torch.empty((2,0), dtype=torch.long)
        u = [e[0] for e in edges_list]
        v = [e[1] for e in edges_list]
        # make both directions
        uu = u + v
        vv = v + u
        return torch.tensor([uu, vv], dtype=torch.long)

    train_pos = to_edge_index(train_edges)
    val_pos = to_edge_index(val_edges)
    test_pos = to_edge_index(test_edges)

    # Now produce negative samples using torch_geometric.utils.negative_sampling.
    # negative_sampling requires edge_index of training graph to avoid sampling positives.
    # We'll use negative_sampling on full set of nodes and exclude all positive edges.
    # For stable behavior, set generator with torch.manual_seed
    torch.manual_seed(seed)
    full_pos_edge_index = to_edge_index(train_edges + val_edges + test_edges)  # all positive edges
    # number of negatives: for each positive in split, produce neg_sample_ratio negatives
    def gen_neg(pos_count):
        if pos_count == 0:
            return torch.empty((2,0), dtype=torch.long)
        num_neg = pos_count * neg_sample_ratio
        neg = negative_sampling(
            edge_index=full_pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg,
            method="sparse"
        )
        return neg

    train_neg = gen_neg(len(train_edges))
    val_neg = gen_neg(len(val_edges))
    test_neg = gen_neg(len(test_edges))

    return {
        "train_pos_edge_index": train_pos,
        "val_pos_edge_index": val_pos,
        "test_pos_edge_index": test_pos,
        "train_neg_edge_index": train_neg,
        "val_neg_edge_index": val_neg,
        "test_neg_edge_index": test_neg,
    }

def save_graphs(out_dir: str, full_data: Data, nodes: List[str], split_dict: Dict[str, torch.Tensor], seed: int):
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, "graph_data_full.pt")
    proc_path = os.path.join(out_dir, "graph_data.pt")
    torch.save({"data": full_data, "nodes": nodes}, full_path)
    # build training Data object (train graph only) for convenience
    train_data = Data(x=full_data.x, edge_index=split_dict["train_pos_edge_index"])
    # Save train-related splits together (so loader doesn't have to re-split)
    torch.save({
        "data": train_data,
        "nodes": nodes,
        "train_pos_edge_index": split_dict["train_pos_edge_index"],
        "val_pos_edge_index": split_dict["val_pos_edge_index"],
        "test_pos_edge_index": split_dict["test_pos_edge_index"],
        "train_neg_edge_index": split_dict["train_neg_edge_index"],
        "val_neg_edge_index": split_dict["val_neg_edge_index"],
        "test_neg_edge_index": split_dict["test_neg_edge_index"],
        "split_seed": seed
    }, proc_path)
    print("Saved full graph to:", full_path)
    print("Saved processed train graph + splits to:", proc_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Folder containing edges.csv and optional nodes.csv (or GEXF).")
    parser.add_argument("--out", default="data/processed", help="Output folder to save graph_data.pt")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction for validation positive edges")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Fraction for test positive edges")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    parser.add_argument("--neg-ratio", type=int, default=1, help="Negative samples per positive")
    args = parser.parse_args()

    raw_dir = args.raw
    out_dir = args.out
    seed = args.seed

    print("Reading raw data from:", raw_dir)
    node_ids, edge_index, x = read_edges_and_nodes(raw_dir)
    print(f"Found {len(node_ids)} nodes and {edge_index.size(1)} directed edges (after to_undirected).")

    data_full = build_pyg_data(node_ids, edge_index, x)
    split_dict = make_splits(edge_index, num_nodes=len(node_ids), val_frac=args.val_frac, test_frac=args.test_frac, seed=seed, neg_sample_ratio=args.neg_ratio)

    save_graphs(out_dir, data_full, node_ids, split_dict, seed)
    print("Done.")

if __name__ == "__main__":
    main()
