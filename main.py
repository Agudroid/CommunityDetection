import argparse
import logging
import Graph_Construction.BarbellGraph as bg
import Graph_Construction.LFRBenchmarkGraph as lfr
import Algorithms.AntColonyOptimization as aco



logging.basicConfig(format='[%(levelname)s]:%(message)s',level=logging.INFO)
logging.basicConfig(format='[%(levelname)s]:%(message)s',level=logging.WARNING)

logging.info("Community Detection started..")
logging.info("Parsing parameters..")
parser = argparse.ArgumentParser(description='Community detection program')
parser.add_argument('-g', type=str, help='type of graph')
parser.add_argument('-a', type=str, help='type of algorithm')
args = parser.parse_args()

graph_type = args.g
algorithm_selected = args.a

logging.info("Generating Graph..")
if(graph_type == 'barbell'):
    logging.info("Generating a Barbell Graph..")
    graph = bg.Barbell_Graph()
    logging.info("Barbell Graph created")
elif(graph_type == 'lfr'):
    logging.info("Generating LFR_Benchmark graph")
    graph = lfr.LFR_Graph()
    logging.info("Barbell Graph created")    
else:
    logging.warn("No graph selected..")
    logging.warn("Using a lfr graph as default..")
    graph = lfr.LFR_Graph()

logging.info("Selecting the algorithm..")
if(algorithm_selected == 'ACO'):
    logging.info("Using Ant Colony Optimization to find the communities..")
    communities = aco.ants_algorithm(graph=graph)
    logging.info("Communities found: " + communities)
elif(algorithm_selected == 'Louvain'):
    logging.info("Generating LFR_Benchmark graph")
    graph = lfr.LFR_Graph()
    logging.info("Barbell Graph created")    
else:
    logging.info("Using Ant Colony Optimization to find the communities..")
    communities = aco.ants_algorithm(graph=graph)
    logging.info("Communities found: " + communities)
