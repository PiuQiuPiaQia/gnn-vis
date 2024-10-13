from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

def load_dataset():
    # 加载Cora数据集
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]

    # 创建图
    G = to_networkx(data, to_undirected=True)
    return G, data