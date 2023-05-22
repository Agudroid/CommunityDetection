import networkx as nx
import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import Heuristics
from sklearn.metrics import normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")
torch.device("cpu")

# Generar grafo de barbell con comunidades
G = nx.barbell_graph(5, 1)


distances = dict(nx.shortest_path_length(G))

# Convertir a grafo DGL y agregar características
dgl_G = dgl.from_networkx(G)

communities = Heuristics.ants_heuristic(dgl_G)
communities = torch.LongTensor(communities)

ndata_distance = []
for node in range(dgl_G.number_of_nodes()):
    ndata_distance.append(torch.FloatTensor(list(distances[node].values())))
dgl_G.ndata['distance'] = torch.stack(ndata_distance, dim=0)

dgl_G.ndata['feat'] = torch.rand((dgl_G.num_nodes(), 10))
dgl_G.ndata['feat'] = torch.cat([dgl_G.ndata['feat'], dgl_G.ndata['distance']], dim=1)  # Concatenar distancias a las características existentes

num_nodes = dgl_G.num_nodes()


# Calcular la matriz de distancia
distances = dict(nx.all_pairs_shortest_path_length(G))


# Definir modelo de detección de comunidades
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_feats, hidden_size))
        self.layers.append(torch.nn.Linear(hidden_size, num_classes))

    def forward(self, g):
        h = g.ndata['feat']
        for layer in self.layers:
            h = F.relu(layer(h))
        return h

def test_net(graph):
    new_node_feat = torch.randn(1, 10)  # Características del nuevo nodo
    graph.add_nodes(1)  # Agregar un nuevo nodo al grafo
    graph.ndata['feat'] = torch.cat([dgl_G.ndata['feat'], new_node_feat], dim=0)
    model.eval()
    with torch.no_grad():
        predictions = model(graph)
        predicted_community = torch.argmax(predictions[-1])
        print("El nuevo nodo pertenece a la comunidad:", predicted_community.item())


communities_set = set(communities)
# Entrenar modelo
print(len(dgl_G.ndata['feat']))
model = GCN(dgl_G.ndata['feat'].shape[1],dgl_G.ndata['feat'].shape[0], len(communities_set))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    logits = model(dgl_G)
    loss = criterion(logits, communities) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predicted_labels = logits.argmax(dim=1)
    nmi = normalized_mutual_info_score(communities, predicted_labels.numpy())
    print('Epoch {}, loss {:.4f}, NMI {:.4f}'.format(epoch, loss.item(), nmi))


