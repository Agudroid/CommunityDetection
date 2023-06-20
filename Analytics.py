from sklearn.metrics.cluster import normalized_mutual_info_score
import networkx as nx
import matplotlib.pyplot as plt


def nmi(labels, pred):
    return normalized_mutual_info_score(labels_true=labels, labels_pred=pred)


def estadistics(graph, lossList, nmiList, communities):
    color_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            color_map[node] = i
            
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color=list(color_map.values()), cmap='tab10', node_size=200)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.axis('off')
    plt.title(f'Grafo con Comunidades (NMI: {nmi:.3f})')
    plt.show()
    