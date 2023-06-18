import networkx as nx
from networkx.generators.community import LFR_benchmark_graph as lfr

def __graph_generator__(num_nodes, tau1, tau2, mu):
    graph = lfr(num_nodes, tau1, tau2, mu, average_degree=5, min_community=20, seed=0)
    return graph


num_nodes = 250
tau1 = 3
tau2 = 1.5
mu = 0.1

for i in range(1,100):
    graph = __graph_generator__(num_nodes,tau1,tau2,mu)
    instance_name = 'graph_lfr'+str(i)+'.gml'
    for node, data in graph.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = ", ".join(map(str, value))
    path = instance_name
    nx.write_gml(graph,path)