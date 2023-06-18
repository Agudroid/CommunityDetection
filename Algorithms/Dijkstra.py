import networkx as nx

def dijkstra_optimized(graph):
    return nx.all_pairs_dijkstra_path(graph, weight='weight')
