import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
import Algorithms.Louvain as al
import Algorithms.Dijkstra as dj
import Graph_Construction as gc
import Graph_Construction.BarbellGraph as barbell
import Graph_Construction.LFRBenchmarkGraph as lfr
import logging

class GCN(nn.Module):
    
    def __init__(self, in_feats, hidden_size,num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, hidden_size)
        self.conv3 = dgl.nn.GraphConv(hidden_size, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h
    
def train(graph, communities):
    
    dgl_G = dgl.from_networkx(graph)
    shortest_paths = dict(dj.dijkstra_optimized(graph))
    distances = []
    for node, paths in shortest_paths.items():
        list_path = list(paths.values())
        node_path = []
        for path in list_path:
            node_path.append(len(path))
        distances.append(node_path)
    features = torch.Tensor(distances)

    num_communities = max(communities) + 1
    communities = torch.LongTensor(communities)

    num_nodes = dgl_G.num_nodes()
    n_train = int(num_nodes * 0.6)
    n_val = int(num_nodes * 0.2)
    n_test = num_nodes - n_train - n_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[n_train:n_train+n_val] = True

    model = GCN(features.shape[1], features.shape[0], num_communities)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(50):
        logits = model(dgl_G, features)
        pred = logits.argmax(axis=1)
        train_logits = logits[train_mask]
        train_labels = communities[train_mask]
        loss = F.cross_entropy(train_logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print("Epoch {:2d}, Loss {:.4f}".format(epoch, loss.item()))

def main():
    logging.info("prueba")
    graph = lfr.LFR_Graph()
    communities = al.lovain_algorithm_optimized(graph)
    logging.info("prueba 2")
    train(graph=graph, communities=communities)

main()