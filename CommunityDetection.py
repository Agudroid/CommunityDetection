import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from networkx.generators.community import LFR_benchmark_graph


def __graph_generator__(n):

    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    G = LFR_benchmark_graph(n,tau1,tau2,mu,average_degree=5,min_community=20,seed=0)
    return G

def __dataset_generator__():
    # Definir los parámetros del grafo de tipo LFR
    n = 1000   # Número de nodos
    tau1 = 3   # Exponente de la distribución de grado interno
    tau2 = 1.5 # Exponente de la distribución de grado externo
    mu = 0.1   # Fracción de enlaces entre comunidades

    # Generar el grafo utilizando la función LFR_benchmark_graph de NetworkX
    G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=0)

    # Convertir el grafo de NetworkX a un grafo de DGL
    dgl_G = dgl.from_networkx(G)

    # Agregar características a los nodos y aristas (opcional)
    dgl_G.ndata['feat'] = torch.randn(n, 10)
    dgl_G.edata['feat'] = torch.randn(dgl_G.num_edges(), 5)
    return dgl_G,G

class CommunityDetection(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(CommunityDetection,self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dgl.nn.GraphConv(hidden_feats, out_feats)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = f.relu(h)
        h = self.conv2(g,h)
        return h

def train(G,dgl_G):
    optimizer = torch.optim.Adam(dgl_G.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = dgl_G.ndata["feat"]
    labels = []
    for node in G.nodes:
        node_community = G.nodes[node]['community']
        labels.append(list(node_community)[0])
    train_mask = dgl_G.ndata["train_mask"]
    val_mask = dgl_G.ndata["val_mask"]
    test_mask = dgl_G.ndata["test_mask"]
    for e in range(100):
        #Forward
        logits = dgl_G(dgl_G, features)

        #Compute
    """train_mask = np.zeros(1000, dtype=bool)
        train_mask[:int(1000*0.8)] = True
        test_mask = np.logical_not(train_mask)
        train_mask = torch.BoolTensor(train_mask)
        test_mask = torch.BoolTensor(test_mask)

        # Crear el modelo y optimizador
        model = CommunityDetection(10, 16, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Entrenar el modelo
        features = dgl_G.ndata['feat']
        labels = []
        for node in G.nodes:
            node_community = G.nodes[node]['community']
            labels.append(list(node_community)[0])
        labels = torch.LongTensor(labels)
        train_labels = labels[train_mask]
        n_train = train_labels.shape[0]
        for epoch in range(50):
            logits = model(dgl_G, features)

            #Compute prediction
            pred = logits.argmax(1)

            loss = f.cross_entropy(logits[train_mask], labels[train_mask])

            train_acc = (pred[train_mask] == labels[train_mask].float().mean())
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 5 == 0:
                print(
                    "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                        e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                    )
                )      """   
dgl_G,G = __dataset_generator__()
train(G,dgl_G)