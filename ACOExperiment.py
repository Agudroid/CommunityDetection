import pandas as pd
import networkx as nx
from Algorithms.AntColonyOptimization import __ant_colony_optimization__ as aco
import dgl


num_ants = 2
num_iterations = 10
alpha = 1.0
beta = 1.0
evaporation_rate = 0.25
experiments = []
for i in range(10):
    dataDicc = {}
    graph = nx.barbell_graph(3,1)
    instance_name = 'graph_barbell'+str(i)+'.gml'
    for node, data in graph.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = ", ".join(map(str, value))
    path = instance_name
    nx.write_gml(graph,path)
    dataDicc['Instancia'] = path
    graph = dgl.from_networkx(graph)
    communities,modularity = aco(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
    dataDicc['Modularidad1'] = modularity
    evaporation_rate = 0.5
    communities,modularity = aco(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
    dataDicc['Modularidad2'] = modularity
    evaporation_rate = 0.75
    communities,modularity = aco(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
    dataDicc['Modularidad3'] = modularity
    experiments.append(dataDicc)

df = pd.DataFrame(experiments)

# Guardar el DataFrame en un archivo CSV
columns = list(experiments[0].keys())

print(experiments[0].keys())
df.to_csv('datos.csv', index=True, header=columns)