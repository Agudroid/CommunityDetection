import pandas as pd
import networkx as nx
from Algorithms.AntColonyOptimization import __ant_colony_optimization__ as aco
import dgl


num_ants = 2
num_iterations = 10
alpha = 1.0
beta = 1.0
evaporation_rate = 0.5
graphList = []
modularity1 = []
modularity2 = []
modularity3 = []
for i in range(1):
    graph = nx.barbell_graph(3,1)
    instance_name = 'graph_barbell'+str(i)+'.gml'
    for node, data in graph.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = ", ".join(map(str, value))
    path = instance_name
    nx.write_gml(graph,path)
    graphList.append(instance_name)
    graph = dgl.from_networkx(graph)
    communities,modularity = aco(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
    modularity1.append(modularity)
    evaporation_rate = 0.5
    communities,modularity = aco(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
    modularity2.append(modularity)
    evaporation_rate = 0.75
    communities,modularity = aco(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
    modularity3.append(modularity)

dict1 = {'Instancia': graphList}
print(dict1)
dict2 = {'modularidad': modularity1}
print(dict2)
dict3 = {'modularidad': modularity2}
print(dict3)
dict4 = {'modularidad': modularity3}
print(dict4)
df1 = pd.DataFrame.from_dict(dict1, orient='index', columns=['Column1'])
df2 = pd.DataFrame.from_dict(dict2, orient='index', columns=['Column2'])
df3 = pd.DataFrame.from_dict(dict3, orient='index', columns=['Column3'])
df4 = pd.DataFrame.from_dict(dict4, orient='index', columns=['Column4'])

# Concatenar los DataFrames verticalmente
df_concat = pd.concat([df1, df2, df3,df4], axis=0)

# Guardar el DataFrame en un archivo CSV
df_concat.to_csv('datos.csv', index=True)