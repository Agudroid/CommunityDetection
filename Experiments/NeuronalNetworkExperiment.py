from Algorithms.Louvain import lovain_algorithm_optimized as lv
import Neuronal_Network.Network as nt
import random
import networkx as nx
import time
import pandas as pd
import warnings
import torch
warnings.filterwarnings("ignore")

experiments = []
for i in range(100):
    print('graph number:' + str(i))
    dataDicc = {}
    order_column = []
    graph = nx.barbell_graph(random.randint(10,100),random.randint(10,100))
    instance_name = 'graph'+ str(i) + '.gml'
    dataDicc['Instancia'] = instance_name
    learning_rate_list =  [0.025, 0.050, 0.075, 0.1]
    communities = lv(graph)
    nmi_list = []
    for i in range(len(learning_rate_list)):
        j = i + 1
        lr = learning_rate_list[i]
        start = time.time()
        nmi = nt.train(graph,communities,lr)
        end = time.time()
        dataDicc['Time' + str(j)] = end - start
        nmi_list.append(nmi)
    
    maxIndex = nmi_list.index(max(nmi_list))
    bestNMI = nmi_list[maxIndex] 
    for i in range(len(nmi_list)):
        j = i + 1
        dataDicc['NMI-'+str(j)] = nmi_list[i]
        dataDicc['Best'+str(j)] = 1 if i == maxIndex else 0
        dataDicc['Desv-'+str(j)] = (bestNMI - nmi_list[i])/ bestNMI
        order = ['NMI-'+str(j), 'Time-'+ str(j), 'Best'+str(j), 'Desv-'+ str(j)]
        order_column.append(order)
    
    experiments.append(dataDicc)    

df = pd.DataFrame(experiments)
df.to_csv('datos3.csv', index=True, sep=',')

        
        
        
        
        
        
        
        