# scripts/train_gae.py
"""
Train a GAE on a collaboration graph. Safe, leak-free, and reproducible.
If --kaggle-dataset is provided the script will download the dataset (requires kaggle configured),
attempt to convert CSV edge/node files to a PyG Data object, then create a manual train/val/test
split (no leakage), train a GAE, and evaluate.

Usage examples:
  # Use local processed/graph_data.pt if present:
  python scripts/train_gae.py --data-root data

  # Download a Kaggle dataset (owner/dataset) that contains edges CSVs:
  python scripts/train_gae.py --data-root data --kaggle-dataset rozemberczki/musae-github-social-network

  # Force rebuild from raw CSVs even if data/processed/graph_data.pt exists:
  python scripts/train_gae.py --data-root data --kaggle-dataset ... --no-cache
"""

import os
import sys
import argparse
import json
import zipfile
import shutil
from datetime import datetime
from typing import Optional, Tuple, List, Set

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------
# Safe torch.load helper
# ---------------------------
def safe_load(path: str):
    import torch
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print("Normal torch.load failed:", repr(e))
        print("Attempting to allowlist PyG common classes (only for trusted files)...")
        try:
            # try to register some known PyG internals
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr  # may fail on some versions
            from torch_geometric.data.storage import GlobalStorage
            torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
            print("Added some PyG classes to safe globals; retrying load...")
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            # final fallback (only for trusted checkpoint)
            return torch.load(path, map_location="cpu", weights_only=False)

# ---------------------------
# Kaggle download helpers
# ---------------------------
def ensure_kaggle_configured():
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return False

def download_from_kaggle(dataset: str, dest_dir: str, file: Optional[str] = None) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading Kaggle dataset {dataset} to {dest_dir} (via KaggleApi)...")
        api.dataset_download_files(dataset, path=dest_dir, unzip=False, quiet=False)
        # find zip
        zips = [p for p in os.listdir(dest_dir) if p.endswith(".zip")]
        if not zips:
            return dest_dir
        zip_path = os.path.join(dest_dir, zips[0])
        with zipfile.ZipFile(zip_path, "r") as zf:
            if file:
                matches = [n for n in zf.namelist() if os.path.basename(n) == file]
                if not matches:
                    raise FileNotFoundError(f"{file} not found in dataset zip")
                for name in matches:
                    zf.extract(name, dest_dir)
            else:
                zf.extractall(dest_dir)
        try:
            os.remove(zip_path)
        except Exception:
            pass
        return dest_dir
    except Exception as e:
        print("kaggle python package approach failed:", repr(e))
        kaggle_cli = shutil.which("kaggle")
        if kaggle_cli is None:
            raise RuntimeError("Kaggle API not available. Please install kaggle and configure credentials.")
        print("Falling back to kaggle CLI.")
        cmd = f'kaggle datasets download -d {dataset} -p "{dest_dir}" --unzip -q'
        if file:
            cmd = f'kaggle datasets download -d {dataset} -p "{dest_dir}" -f {file} --unzip -q'
        rc = os.system(cmd)
        if rc != 0:
            raise RuntimeError("kaggle CLI download failed")
        return dest_dir

# ---------------------------
# CSV -> PyG conversion
# ---------------------------
def convert_csvs_to_graph_data(raw_dir: str, out_dir: str) -> str:
    files = os.listdir(raw_dir)
    edges_candidates = [f for f in files if f.lower().startswith("edge") or f.lower().endswith(".edges") or f.lower().endswith("_edges.csv") or f.lower()=="edges.csv"]
    nodes_candidates = [f for f in files if f.lower().startswith("node") or f.lower().endswith("_nodes.csv") or f.lower()=="nodes.csv"]
    if not edges_candidates:
        raise RuntimeError(f"No edges CSV found in {raw_dir}. Files: {files}")
    edges_path = os.path.join(raw_dir, edges_candidates[0])
    nodes_path = os.path.join(raw_dir, nodes_candidates[0]) if nodes_candidates else None

    edges_df = pd.read_csv(edges_path, header=0)
    src = edges_df.iloc[:,0].astype(str).tolist()
    dst = edges_df.iloc[:,1].astype(str).tolist()

    if nodes_path:
        nodes_df = pd.read_csv(nodes_path)
        # try common id columns
        if 'id' in nodes_df.columns:
            nodes = nodes_df['id'].astype(str).tolist()
        elif 'name' in nodes_df.columns:
            nodes = nodes_df['name'].astype(str).tolist()
        else:
            nodes = nodes_df.iloc[:,0].astype(str).tolist()
        node_id = {n:i for i,n in enumerate(nodes)}
        numeric_cols = nodes_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            x = torch.tensor(nodes_df[numeric_cols].fillna(0).values, dtype=torch.float)
        else:
            x = torch.ones((len(nodes), 1), dtype=torch.float)
    else:
        unique_nodes = pd.Index(src + dst).unique()
        nodes = list(unique_nodes)
        node_id = {n:i for i,n in enumerate(nodes)}
        x = torch.ones((len(nodes), 1), dtype=torch.float)

    edge_list = []
    for s,d in zip(src,dst):
        if s not in node_id or d not in node_id:
            continue
        edge_list.append([node_id[s], node_id[d]])

    if not edge_list:
        raise RuntimeError("No edges mapped. Check CSV formatting.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "graph_data.pt")
    torch.save({"data": data, "nodes": nodes}, save_path)
    print("Saved graph_data.pt to", save_path)
    return save_path

# ---------------------------
# Manual split (no leakage)
# ---------------------------
def manual_edge_split(edge_index: torch.Tensor, num_nodes: int, val_frac: float, test_frac: float, seed: int = 42):
    """
    Given undirected edge_index (2, E) return:
      - train_edge_index (2, E_train) (undirected, both directions)
      - test_pos_edge_index (2, E_test) (single direction pairs)
      - test_neg_edge_index (2, E_test) negatives sampled (single direction pairs)
      - val_pos_edge_index, val_neg_edge_index (or None if val_frac==0)
    Ensures disjoint sets, reproducible by seed.
    """
    np.random.seed(seed)
    # get unique undirected pairs where u < v
    e = edge_index.cpu().numpy().T
    uniq = set()
    for u,v in e:
        if u == v: continue
        a,b = (int(u), int(v))
        if a > b: a,b = b,a
        uniq.add((a,b))
    uniq = sorted(list(uniq))
    m = len(uniq)
    if m == 0:
        raise RuntimeError("Graph has no edges.")
    # shuffle indices
    idx = np.arange(m)
    np.random.shuffle(idx)
    n_test = max(1, int(np.round(test_frac * m)))
    n_val = max(0, int(np.round(val_frac * m)))
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test:n_test+n_val])
    train_idx = set(idx[n_test+n_val:])

    train_pairs = [uniq[i] for i in sorted(train_idx)]
    test_pairs = [uniq[i] for i in sorted(test_idx)]
    val_pairs = [uniq[i] for i in sorted(val_idx)] if n_val>0 else []

    # build train_edge_index with both directions
    train_edges_bidir = []
    for (u,v) in train_pairs:
        train_edges_bidir.append([u,v])
        train_edges_bidir.append([v,u])
    train_edge_index = torch.tensor(train_edges_bidir, dtype=torch.long).t().contiguous()

    # helper to sample negatives equal in count to positives
    all_pairs = set(uniq)
    def sample_negatives(k):
        negs = set()
        tries = 0
        while len(negs) < k and tries < k * 20:
            a = np.random.randint(0, num_nodes)
            b = np.random.randint(0, num_nodes)
            if a == b:
                tries += 1
                continue
            x,y = (a,b) if a < b else (b,a)
            if (x,y) in all_pairs or (x,y) in negs:
                tries += 1
                continue
            negs.add((x,y))
        if len(negs) < k:
            # fallback: iterate all possible pairs to fill
            for a in range(num_nodes):
                for b in range(a+1, num_nodes):
                    if (a,b) not in all_pairs and (a,b) not in negs:
                        negs.add((a,b))
                        if len(negs) >= k: break
                if len(negs) >= k: break
        negs = sorted(list(negs))[:k]
        return negs

    test_negs = sample_negatives(len(test_pairs))
    val_negs = sample_negatives(len(val_pairs)) if n_val>0 else []

    # convert lists to tensors (for pos edges we use single direction u->v as stored)
    def pairs_to_edge_index(pairs):
        if not pairs:
            return torch.empty((2,0), dtype=torch.long)
        u = [p[0] for p in pairs]
        v = [p[1] for p in pairs]
        return torch.tensor([u,v], dtype=torch.long)

    test_pos = pairs_to_edge_index(test_pairs)
    test_neg = pairs_to_edge_index(test_negs)
    val_pos = pairs_to_edge_index(val_pairs) if val_pairs else torch.empty((2,0), dtype=torch.long)
    val_neg = pairs_to_edge_index(val_negs) if val_negs else torch.empty((2,0), dtype=torch.long)

    return train_edge_index, test_pos, test_neg, val_pos, val_neg

# ---------------------------
# Model utilities
# ---------------------------
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)

def score_edges_sigmoid(z: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    if edge_index.numel() == 0:
        return np.array([])
    z = z.cpu()
    u = edge_index[0]
    v = edge_index[1]
    scr = (z[u] * z[v]).sum(dim=1)
    return torch.sigmoid(scr).cpu().numpy()

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--kaggle-dataset", default=None, help="owner/dataset (optional)")
    parser.add_argument("--kaggle-file", default=None, help="specific file inside dataset (optional)")
    parser.add_argument("--no-cache", action="store_true", help="Force rebuild from raw CSVs (ignore processed cache)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = args.data_root
    processed_dir = os.path.join(data_root, "processed")
    raw_base = os.path.join(data_root, "raw")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(raw_base, exist_ok=True)

    # If kaggle requested, download into raw/<dataset>
    raw_dataset_dir = None
    if args.kaggle_dataset:
        if not ensure_kaggle_configured():
            print("Kaggle not configured; please put kaggle.json in ~/.kaggle or set env vars.")
            sys.exit(1)
        raw_dataset_dir = os.path.join(raw_base, args.kaggle_dataset.replace("/", "_"))
        os.makedirs(raw_dataset_dir, exist_ok=True)
        print("Downloading dataset:", args.kaggle_dataset)
        download_from_kaggle(args.kaggle_dataset, raw_dataset_dir, file=args.kaggle_file)
        print("Downloaded to:", raw_dataset_dir)

    # Decide whether to use existing processed graph_data.pt or rebuild
    graph_pt = os.path.join(processed_dir, "graph_data.pt")
    if args.no_cache and raw_dataset_dir:
        # force convert CSV -> PT
        print("Rebuilding graph_data.pt from CSVs (no-cache enabled).")
        graph_pt = convert_csvs_to_graph_data(raw_dataset_dir, processed_dir)
    else:
        if os.path.exists(graph_pt):
            print("Found processed graph_data.pt; using it:", graph_pt)
        else:
            # try to find .pt inside raw download, else convert CSVs
            found = None
            if raw_dataset_dir:
                for root, _, files in os.walk(raw_dataset_dir):
                    for fn in files:
                        if fn.endswith(".pt") or fn.endswith(".pth"):
                            found = os.path.join(root, fn)
                            break
                    if found: break
            if found:
                graph_pt = found
                print("Using .pt found in raw dataset:", graph_pt)
            elif raw_dataset_dir:
                # convert CSVs
                print("No .pt found; attempting CSV -> PT conversion from raw dataset.")
                graph_pt = convert_csvs_to_graph_data(raw_dataset_dir, processed_dir)
            else:
                print("No processed graph_data.pt found and no raw dataset provided.")
                print("Place a processed/graph_data.pt in data/processed or supply --kaggle-dataset.")
                sys.exit(1)

    # Load the graph data (safe loader)
    saved = safe_load(graph_pt)
    if isinstance(saved, dict) and "data" in saved and "nodes" in saved:
        data = saved["data"]
        nodes = saved["nodes"]
    else:
        if hasattr(saved, "x") and hasattr(saved, "edge_index"):
            data = saved
            nodes = [str(i) for i in range(data.num_nodes)]
        else:
            raise RuntimeError("Loaded file is not in expected format (dict with 'data'/'nodes' or PyG Data)")

    print(f"Loaded graph: num_nodes={len(nodes)}, num_features={data.x.shape[1]}")

    # Ensure undirected
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)

    num_nodes = data.num_nodes

    # Create manual train/val/test split (no leakage)
    train_ei, test_pos_ei, test_neg_ei, val_pos_ei, val_neg_ei = manual_edge_split(
        data.edge_index, num_nodes, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    print("Split sizes:")
    print(" train edges (directed count):", train_ei.size(1))
    print(" test pos pairs:", test_pos_ei.size(1))
    print(" test neg pairs:", test_neg_ei.size(1))
    if val_pos_ei.numel() > 0:
        print(" val pos pairs:", val_pos_ei.size(1), " val neg pairs:", val_neg_ei.size(1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAE(Encoder(data.x.shape[1], out_channels=64)).to(device)

    # Move data to device for encoding
    x = data.x.to(device)
    train_edge_index = train_ei.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    epochs = args.epochs
    logs = []
    print("Training GAE on", device, "| epochs =", epochs)
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_edge_index)   # encode on training graph only
        loss = model.recon_loss(z, train_edge_index)  # recon over train positives
        loss.backward()
        optimizer.step()
        logs.append({"epoch": epoch, "loss": float(loss.detach().cpu().item())})
        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | loss = {float(loss):.6f}")

    # Evaluate on test set (use z from final model)
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_edge_index)

    pos_probs = score_edges_sigmoid(z, test_pos_ei)
    neg_probs = score_edges_sigmoid(z, test_neg_ei)

    y_true = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    y_score = np.concatenate([pos_probs, neg_probs])

    # If constant scores, warn
    if y_score.size > 0 and np.allclose(y_score, y_score[0]):
        print("WARNING: All predicted scores are identical — check model or data (can cause ROC/AP=nan or 0.5).")

    try:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    except Exception as e:
        print("Failed to compute ROC/AP:", repr(e))
        auc = float("nan")
        ap = float("nan")

    print(f"\nFinal results -> AUC: {auc:.6f}, AP: {ap:.6f}")

    # Top-K predictions among non-train pairs
    # Build train set to avoid proposing existing edges
    train_set = set()
    te = train_ei.cpu().numpy().T
    for u,v in te:
        a,b = int(u), int(v)
        if a <= b:
            train_set.add((a,b))
        else:
            train_set.add((b,a))

    # score all candidate upper-triangular pairs not in train_set
    cand_pairs = []
    cand_scores = []
    z_np = z.cpu().numpy()
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if (i,j) in train_set:
                continue
            s = float(1.0/(1.0 + np.exp(-np.dot(z_np[i], z_np[j]))))
            cand_pairs.append((i,j))
            cand_scores.append(s)

    if len(cand_scores) == 0:
        print("No candidate non-train pairs (graph may be complete).")
        top_pairs = []
        top_scores = []
    else:
        order = np.argsort(-np.array(cand_scores))[:args.topk]
        top_pairs = [cand_pairs[i] for i in order]
        top_scores = [cand_scores[i] for i in order]

    pred_df = pd.DataFrame({
        "u_idx": [p[0] for p in top_pairs],
        "v_idx": [p[1] for p in top_pairs],
        "u": [nodes[p[0]] for p in top_pairs],
        "v": [nodes[p[1]] for p in top_pairs],
        "score": top_scores
    })
    out_pred = os.path.join(processed_dir, f"predicted_links_top{args.topk}.csv")
    pred_df.to_csv(out_pred, index=False)
    print("Saved predictions to:", out_pred)

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(processed_dir, "gae_model.pt"))
    np.save(os.path.join(processed_dir, "gae_embeddings.npy"), z.cpu().numpy())
    with open(os.path.join(processed_dir, "gae_training_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
