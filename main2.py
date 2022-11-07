import dgl
import torch
from dgl.data import DGLDataset
import networkx as nx
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from dgl.nn.pytorch import GraphConv
from torch import nn
from torch.nn import functional as f
import matplotlib as plt


def __nodes_data__(g):
    data = {'feat': [], 'label': []}

    for i in range(g.number_of_nodes()):
        data['feat'].append(random.randint(1, 10))
        data['label'].append(random.randint(0, 1))

    return data


class BarbellGraph(DGLDataset):

    def __init__(self, train_val_test_split=None):
        self.train_val_test_split = train_val_test_split
        self.graph = None
        super().__init__(name='karateClub')

    def process(self):
        nx_graph = nx.barbell_graph(300, 500)
        nodes_data = __nodes_data__(nx_graph)
        node_features = torch.from_numpy(np.asarray(nodes_data['feat']))
        node_labels = torch.from_numpy(np.asarray(nodes_data['label'])).type(torch.LongTensor)

        n_nodes = len(nodes_data['feat'])
        n_train = int(n_nodes * self.train_val_test_split[0])
        n_val = int(n_nodes * self.train_val_test_split[1])

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train: n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        scaler = StandardScaler()
        scaler.fit(node_features[train_mask].reshape(-1, 1))
        node_features_standarized = scaler.transform(torch.reshape(node_features, (-1, 1)))
        node_features_standarized = torch.from_numpy(
            node_features_standarized).to(torch.float32)

        self.graph = dgl.from_networkx(nx_graph)
        self.graph.ndata['feat'] = node_features_standarized
        self.graph.ndata['label'] = node_labels

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()

        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = f.relu(h)
        h = self.conv2(g,h)
        h = f.relu(h)
        h = self.conv3(g, h)
        return h


def train(graph, model, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    best_val_acc = 0
    best_test_acc = 0

    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    for epoch in range(1, epochs+1):

        logits = model(graph, features)
        pred = logits.argmax(axis=1)

        loss = f.cross_entropy(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(f'In {epoch}, loss: {loss:.3f}, val acc: {best_val_acc:.3f}, test acc: {best_test_acc:.3f}')


dataset = BarbellGraph(train_val_test_split=[0.6, 0.2, 0.2])
graph = dataset[0]


model = GCN(in_feats=graph.ndata['feat'].shape[1], h_feats=100, num_classes=2)
train(graph,model,epochs=100)
print(model)

