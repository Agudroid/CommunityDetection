import networkx as nx
import dgl
import numpy as np
import torch
import random

def __ant_colony_optimization__(graph, num_ants, num_iterations, alpha, beta, evaporation_rate):
    pheromones = torch.ones(graph.number_of_edges())
    best_partition = None
    best_modularity = float('-inf')

    for _ in range(num_iterations):
        partitions = []
        modularity_scores = []

        for _ in range(num_ants):
            partition = []
            current_node = random.randint(0, graph.number_of_nodes()-1)
            while len(partition) < graph.number_of_nodes():
                partition.append(current_node)  

                probabilities = __calculate_probabilities__(graph, pheromones, partition, alpha, beta)

                current_node = random.choices(range(graph.number_of_nodes()), weights=probabilities)[0]

            partitions.append(partition)
            modularity = __calculate_modularity__(graph, partitions)
            modularity_scores.append(modularity)

            if modularity > best_modularity:
                best_partition = partition
                best_modularity = modularity

        __update_pheromones__(pheromones, partitions, modularity_scores, evaporation_rate, graph)

    return best_partition


def __calculate_probabilities__(graph, pheromones, partition, alpha, beta):
    remaining_nodes = [node for node in range(graph.number_of_nodes()) if node not in partition]
    probabilities = [0.0] * graph.number_of_nodes()
    
    if not remaining_nodes:
        return [1 / graph.number_of_nodes()] * graph.number_of_nodes()

    
    for node in remaining_nodes:
        if graph.has_edges_between(partition[-1], node):
            edge_id = graph.edge_ids(partition[-1], node)
            probability = pheromones[edge_id]
            probability = probability ** alpha
            probability = probability * graph.in_degrees(node) ** beta
            probabilities[node] = probability
        else:
            probability = 0.0
        probabilities[node]=probability

    total_probability = sum(probabilities)

    if total_probability == 0:
        probabilities = [1 / graph.number_of_nodes()] * graph.number_of_nodes()
    else:
        probabilities = [p / total_probability for p in probabilities]

    return probabilities


def __calculate_modularity__(graph, partitions):
    num_edges = graph.number_of_edges()
    modularity = 0.0

    for community in partitions:
        for u in community:
            for v in community:
                adj = graph.has_edges_between(u, v)
                ki = graph.in_degrees(u)
                kj = graph.in_degrees(v)
                modularity += adj - (ki * kj) / (2 * num_edges)

    modularity /= 2 * num_edges

    return modularity


def __update_pheromones__(pheromones, partitions, modularity_scores, evaporation_rate, graph):
    pheromones *= (1 - evaporation_rate)

    for partition, modularity in zip(partitions, modularity_scores):
        for i in range(len(partition) - 1):
            if graph.has_edges_between(partition[i], partition[i+1]):
                edge_id = graph.edge_ids(partition[i], partition[i+1])
                pheromones[edge_id] += modularity

    pheromones /= torch.sum(pheromones)

def ants_algorithm(graph):
    num_ants = 2
    num_iterations = 20
    alpha = 1.0
    beta = 1.0
    evaporation_rate = 10
    communities = __ant_colony_optimization__(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
    print(communities)
    return communities

graph = nx.barbell_graph(2,2)
ants_algorithm(dgl.from_networkx(graph) )