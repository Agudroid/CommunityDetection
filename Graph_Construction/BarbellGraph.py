import networkx as nx
from random import randint


def Barbell_Graph():
    n1 = randint(2,10)
    n2 = randint(2,10)
    return nx.barbell_graph(n1,n2)