import dgl
import torch
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

#Define the customized DGLDataset
class CustomDataset(DGLDataset):
    def __init__(self, train_val_test_split = None):
        self.train_val_test_split = train_val_test_split

        super().__init__(name = 'custom_dataset')
    
    def process(self):

        nodes_data = pd.read_csv('./csv/members.csv')

        edges_data = pd.read_csv('./csv/interactions.csv')

        node_features = torch.from_numpy(nodes_data['Age'].to_numpy()).type(torch.LongTensor)

        node_labels = torch.from_numpy(
            nodes_data["Club"].astype("category").cat.codes.to_numpy()
        )

        edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())

        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())

        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())

        n_nodes = nodes_data.shape[0]

        n_train = int(n_nodes * self.train_val_test_split[0])

        n_val = int(n_nodes * self.train_val_test_split[1])

        train_mask = torch.zeros(n_nodes, dtype = torch.bool)
        val_mask = torch.zeros(n_nodes, dtype = torch.bool)
        test_mask = torch.zeros(n_nodes, dtype = torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train: n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        scaler = StandardScaler()
        scaler.fit(node_features[train_mask].reshape(-1,1))
        node_features_standarized = scaler.transform(node_features.reshape(-1,1))
        node_features_standarized = torch.from_numpy(node_features_standarized).to(torch.float32)

        self.graph = dgl.graph(
            (edges_src, edges_dst),
            num_nodes= nodes_data.shape[0]
        )


