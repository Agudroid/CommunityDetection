import dgl
from dgl.data import DGLDataset
import torch
import pandas as pd

class Dataset(DGLDataset):
    def __init__(self, path):
        self.path = path
        super().__init__(name="csv_dataset", raw_dir=path)
        self.process()

    def process(self):
        
        csv = pd.read_csv(self.path, encoding='iso-8859-1', error_bad_lines=False)
        
        edges_src = []
        edges_dst = []
        nodes_data = []

        for index, row in csv.iterrows():
            row_data = []
            for i in range(len(row)):
                if i == 1:
                    edges_src.append(row[i])
                elif i == 2:
                    edges_dst.append(row[i])
                else:
                    row_data.append[row[i]]
            nodes_data.append[row_data]
    
        tensor_data = torch.tensor(nodes_data)
        tensor_src = torch.tensor(edges_src)
        tensor_dst = torch.tensor(edges_dst)
        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes = tensor_data.shape[0]
        )


    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

        

def get_csv_graph(path):
    dataset = Dataset(path)
    graph = dataset[0]
    print(graph)

get_csv_graph(r"C:\Users\Antonio\VisualStudioProjects\VisualStudioProjects\CommunityDetection\csv\example.csv")