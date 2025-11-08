import os
import networkx as nx

GEXF_PATH = os.path.join('data', 'processed', 'github_collab_graph.gexf')
if not os.path.exists(GEXF_PATH):
    print('GEXF not found:', GEXF_PATH)
    raise SystemExit(1)

G = nx.read_gexf(GEXF_PATH)

print('Nodes:', G.number_of_nodes())
print('Edges:', G.number_of_edges())

# weighted degree
deg = dict(G.degree(weight='weight'))
top5 = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]
print('\nTop nodes by weighted degree:')
for n, d in top5:
    print(f'  {n}: {d}')

# connected components
comps = list(nx.connected_components(G))
comps_sorted = sorted(comps, key=lambda c: -len(c))
print('\nConnected components (size, members):')
for c in comps_sorted:
    print(' ', len(c), sorted(list(c)))

# edges with high weight
heavy = [(u, v, d.get('weight', 1)) for u, v, d in G.edges(data=True) if float(d.get('weight',1)) >= 3]
heavy_sorted = sorted(heavy, key=lambda x: -x[2])
print('\nHeavy edges (weight >= 3):')
for u, v, w in heavy_sorted:
    print(' ', u, '<->', v, 'weight=', w)

# clustering & centrality
avg_clust = nx.average_clustering(G, weight='weight')
print('\nAverage clustering coefficient (weighted):', round(avg_clust, 4))

bet = nx.betweenness_centrality(G, weight='weight')
top_bet = sorted(bet.items(), key=lambda x: -x[1])[:5]
print('\nTop nodes by betweenness centrality:')
for n, b in top_bet:
    print(f'  {n}: {round(b,4)}')

print('\nEdge list (with weight):')
for u, v, d in G.edges(data=True):
    print(' ', u, '--', v, 'w=', d.get('weight', 1))
