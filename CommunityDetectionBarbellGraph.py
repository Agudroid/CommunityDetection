import networkx as nx
import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

# Generar grafo de barbell con comunidades
G = nx.barbell_graph(5, 0)
communities = np.zeros(G.number_of_nodes())
communities[0:5] = 1
communities[10:15] = 2

distances = dict(nx.shortest_path_length(G))

# Convertir a grafo DGL y agregar características
dgl_G = dgl.from_networkx(G)

ndata_distance = []
for node in range(dgl_G.number_of_nodes()):
    ndata_distance.append(torch.FloatTensor(list(distances[node].values())))
dgl_G.ndata['distance'] = torch.stack(ndata_distance, dim=0)

#dgl_G.ndata['feat'] = torch.rand((dgl_G.num_nodes(), 4))
num_nodes = dgl_G.num_nodes()


# Calcular la matriz de distancia
distances = dict(nx.all_pairs_shortest_path_length(G))


# Definir modelo de detección de comunidades
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, 16)
        self.conv2 = dgl.nn.GraphConv(16, out_feats)
        self.fc = nn.Linear(out_feats, 3)
        self.conv1.reset_parameters()
        
    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        x = self.fc(x)
        return x

# Entrenar modelo
model = GCN(len(ndata_distance), 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    logits = model(dgl_G, dgl_G.ndata['distance'])
    loss = criterion(logits, torch.LongTensor(communities))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    nmi = normalized_mutual_info_score(communities, logits.argmax(1).detach().numpy())
    print('Epoch {}, loss {:.4f}, NMI {:.4f}'.format(epoch, loss.item(), nmi))