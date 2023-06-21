import pandas as pd
import networkx as nx
import random
import time
from Algorithms.AntColonyOptimization import __ant_colony_optimization__ as aco
import dgl


num_ants = 2
num_iterations = 10
alpha = 1.0
beta = 1.0
evaporation_rate = 0.25
experiments = []
for i in range(10):
    print('graph number:' + str(i))
    dataDicc = {}
    graph = nx.barbell_graph(3,3)
    instance_name = 'graph'+str(i)+'.gml'
    for node, data in graph.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, set):
                data[key] = ", ".join(map(str, value))
    path = instance_name
    nx.write_gml(graph,path)
    modularityList = []
    orderColumn = ['Instancia']
    evaporationRateList = [0.25,0.5,0.75,random.random(),0.25,0.25,0.25]
    alphaBetaList = [(1,1),(1,1),(1,1),(1,1),(1,1),(2,1),(1,2)]
    dataDicc['Instancia'] = path
    graph = dgl.from_networkx(graph)
    
    for i in range(len(evaporationRateList)):
        j = i + 1
        evaporation_rate = evaporationRateList[i]
        alpha, beta = alphaBetaList[i]
        start = time.time()
        communities,modularity = aco(graph, num_ants, num_iterations, alpha, beta, evaporation_rate)
        end = time.time()
        modularityKey = 'Modularidad' + str(j)
        dataDicc[modularityKey] = modularity
        timeKey = 'Time' + str(j)
        modularityList.append(modularity)
        dataDicc[timeKey] = end - start

     
    experiments.append(dataDicc)
    maxIndexPheromone = modularityList.index(max(modularityList[:4]))
    maxIndexAlphaBeta = modularityList.index(max(modularityList[4:]))
    for i in range(len(modularityList)):
        maxIndex = maxIndexPheromone if i < 4 else maxIndexAlphaBeta
        j = i+1
        modularityKey = 'Modularidad'+str(j)
        orderColumn.append(modularityKey)
        timeKey = 'Time' + str(j)
        orderColumn.append(timeKey)
        BestKey = 'Best'+ str(j)
        orderColumn.append(BestKey)
        dataDicc[BestKey] = 1 if i == maxIndex else 0
        DevKey = 'Dev' + str(j)
        orderColumn.append(DevKey)
        bestModularity = modularityList[maxIndex]
        
        dataDicc[DevKey] = (bestModularity - dataDicc[modularityKey]) / bestModularity
        print('Desviation: ' + str(dataDicc[DevKey]))  

    
df = pd.DataFrame(experiments)

# Guardar el DataFrame en un archivo CSV
columns = list(experiments[0].keys())
df = df.reindex(columns=orderColumn)
df.to_csv('datos1.csv', index=True, sep=',')