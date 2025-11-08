# 01_fetch_build_graph.py
import os
import sys
from github import Github
import networkx as nx
import json

# --- config ---
# Put your GitHub token in an environment variable named GITHUB_TOKEN:
# export GITHUB_TOKEN="ghp_xxx"
TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    print("Error: set GITHUB_TOKEN environment variable (export GITHUB_TOKEN='ghp_...')", file=sys.stderr)
    sys.exit(1)

REPO_NAME = "oppia/oppia"   # change to your chosen repo
COMMIT_LIMIT = 30           # reduced from 500 to 50 for faster testing

# --- authenticate ---
g = Github(TOKEN)
repo = g.get_repo(REPO_NAME)

# --- gather commits (simple, may hit rate limits if large) ---
print("Fetching commits (this may take a bit)...")
commits = list(repo.get_commits()[:COMMIT_LIMIT])  # PyGithub supports slicing for small counts

# Build a map: file -> set(developers who touched it in sampled commits)
file_to_authors = {}
for i, c in enumerate(commits):
    author = None
    try:
        author = c.author.login if c.author else None
    except Exception:
        # some commits may have no author or data access; skip safely
        author = None
    if not author:
        continue
    try:
        # commit.files is available on commit object only if GitHub API returned it;
        # fallback: skip if not present.
        files = getattr(c, "files", None)
        if not files:
            continue
        for f in files:
            filename = f.filename
            file_to_authors.setdefault(filename, set()).add(author)
    except Exception as e:
        # skip errors for specific commits
        continue

# --- build weighted co-edit graph ---
G = nx.Graph()
# add nodes: unique authors
all_authors = set()
for s in file_to_authors.values():
    all_authors.update(s)
G.add_nodes_from(all_authors)

# for every file, connect every pair of authors who edited it
for authors in file_to_authors.values():
    authors = list(authors)
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            a, b = authors[i], authors[j]
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (weighted).")

# --- simple metrics ---
deg = dict(G.degree(weight="weight"))
top5 = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 by weighted degree:", top5)

# --- save outputs ---
out_dir = "data/processed"
os.makedirs(out_dir, exist_ok=True)
gexf_path = os.path.join(out_dir, "github_collab_graph.gexf")
nx.write_gexf(G, gexf_path)
print("Saved graph to", gexf_path)

# optional: save a small JSON summary
summary = {
    "repo": REPO_NAME,
    "commits_sampled": min(COMMIT_LIMIT, len(commits)),
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "top5_degree": top5
}
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary.")
