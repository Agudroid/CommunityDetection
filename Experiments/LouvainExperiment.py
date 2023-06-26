import networkx as nx
import community
import random 
import time
import pandas as pd
num_graphs = 100
modularities = []

experiments = []
for i in range(num_graphs):
    dataDicc = {}
    # Crear un grafo Barbell
    graph = nx.barbell_graph(random.randint(50,500), random.randint(50,500))
    instance_name = 'graph'+str(i)+'.gml'
    for node, data in graph.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = ", ".join(map(str, value))
                
    dataDicc['Instance'] = instance_name
    start = time.time()
    # Aplicar el algoritmo Louvain para clasificar las comunidades
    partition = community.best_partition(graph)
    end = time.time()
    dataDicc['Time'] = end -start
    nx.write_gml(graph,instance_name)
    # Calcular la modularidad
    modularity_value = community.modularity(partition, graph)
    dataDicc['Modularity'] = modularity_value
    print(modularity_value)
    modularities.append(modularity_value)
    experiments.append(dataDicc)

df = pd.DataFrame(experiments)
df.to_csv('datos2.csv', index=True, sep=',')