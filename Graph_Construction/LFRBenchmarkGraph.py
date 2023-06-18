from networkx import LFR_benchmark_graph as lfr    
    
def LFR_Graph():    
    n = 250
    tau1 = 1.5
    tau2 = 1.5
    mu = 0.1
    return lfr(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=0)