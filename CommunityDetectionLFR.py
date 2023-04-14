import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics.cluster import normalized_mutual_info_score

# 0. Lo manejamos como script
parser = argparse.ArgumentParser(description='Selecciona el modo del script')
parser.add_argument('-r', '--random', action='store_true', help='random')
parser.add_argument('-d', '--dijkstra', action='store_true', help='Los datos de los nodos son un dijkstra')
parser.add_argument('-p', '--dephpath', action='store_true', help='Los datos de los nodos son un recorrido en anchura')

args = parser.parse_args()






def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracy = correct.item() * 1.0 / len(labels)
    return accuracy

# 1. Generar un grafo LFR_benchmark con la biblioteca NetworkX.
n = 250
tau1 = 3
tau2 = 1.5
mu = 0.1
G = nx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)

# 2. Convertir el grafo de NetworkX a un grafo de DGL.
dgl_G = dgl.from_networkx(G)




features = torch.randn(n, n)
print(features)
shortest_paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
distances = []
for node, paths in shortest_paths.items():
    list_path = list(paths.values())
    node_path = []
    for path in list_path:
        node_path.append(len(path))
    distances.append(node_path)
features = torch.Tensor(distances)
print(features.shape)





# 4. Definir las etiquetas de los nodos, que representan la clase a la que pertenece cada nodo.
labels = []
for node in G.nodes:
    node_community = G.nodes[node]['community']
    labels.append(list(node_community)[0])
num_communities = max(labels) + 1
labels = torch.LongTensor(labels)

# 5. Dividir las etiquetas de los nodos en conjuntos de entrenamiento, validación y prueba.
n_train = int(n * 0.6)
n_val = int(n * 0.2)
n_test = n - n_train - n_val

train_mask = torch.zeros(n, dtype=torch.bool)
train_mask[:n_train] = True
val_mask = torch.zeros(n, dtype=torch.bool)
val_mask[n_train:n_train+n_val] = True
test_mask = torch.zeros(n, dtype=torch.bool)
test_mask[n_train+n_val:] = True

# 6. Definir una red neuronal de DGL para la clasificación de nodos.
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 7. Entrenar la red neuronal con el conjunto de entrenamiento y evaluarla con el conjunto de validación.
model = GCN(n, int(n/2), num_communities)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
f = nn.CrossEntropyLoss()
for epoch in range(1000):
    logits = model(dgl_G, features)
    pred = logits.argmax(axis=1)
    train_logits = logits[train_mask]
    train_labels = labels[train_mask]
    loss = F.cross_entropy(train_logits, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        acc = evaluate(model, dgl_G, features, labels, val_mask)
        print("Epoch {:2d}, Loss {:.4f}, Accuracy {:.4f}".format(epoch, loss.item(), acc))
        nmi = normalized_mutual_info_score(labels_true=labels, labels_pred=pred)
        print("NMI: {}".format(nmi))



