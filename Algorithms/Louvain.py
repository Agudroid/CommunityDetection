import networkx as nx
from networkx.algorithms import community
import random

def louvain_algorithm(graph):
    
    #Inicializamos la particion y establecemos que se puede mejorar
    partition = {node: node for node in graph.nodes}
    improved = True

    while improved:
        improved = False
        nodes = list(graph.nodes)
        random.shuffle(nodes)

        for node in nodes: 
            current_community = partition[node]
            best_community = current_community
            max_delta_q = 0.0

            for neighbor in graph.neighbors(node):
                if partition[neighbor] != current_community:
                    delta_q = __compute_delta_modularity__(graph, partition, node, neighbor)

                    if delta_q > max_delta_q:
                        max_delta_q = delta_q
                        best_community = partition[neighbor]

                    if best_community != current_community:
                        partition[node] = best_community
                        improved = True

    return partition

def __compute_delta_modularity__(graph, partition, node, community):
    intra_community_degree = sum(graph.degree[neighbor] for neighbor in graph.neighbors(node) if partition[neighbor] == community)
    total_degree = graph.degree[node]
    community_size = sum(1 for _, value in partition.items() if value == community)

    delta_q = (intra_community_degree / (2 * graph.number_of_edges())) - ((total_degree * community_size) / (2 * graph.number_of_edges()))**2

    return delta_q

def lovain_algorithm_optimized(graph):
    partition = community.greedy_modularity_communities(graph)
    community_list = []
    for node in range(graph.number_of_nodes()):
        for i, comm in enumerate(partition):
            if node in comm:
                community_list.append(i)
                break
    return community_list
