from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.utils import to_networkx
import numpy as np

def load_dataset():
    # 加载Cora数据集
    # dataset = Planetoid(root='./data/Cora', name='Cora')
    # data = dataset[0]

    # 加载Amazon Co-purchase数据集
    # nodes: 13752
    # edges: 245861
    dataset = Amazon(
        root='../data/Amazon',  # 指定数据集存储路径
        name='Computers'  # 指定类别，例如'Computers'
    )
    data = dataset[0]


    # 创建图
    G = to_networkx(data, to_undirected=True)
    return G, data



# 加载数据集
G, data = load_dataset()

print(len(G.nodes()), len(G.edges()))
# print(data.x)
# print(data.y)
# print(data.edge_index)