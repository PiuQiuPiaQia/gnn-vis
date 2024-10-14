from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.utils import to_networkx

def load_dataset():
    # 加载Cora数据集
    dataset = Planetoid(root='./data/Cora', name='Cora')

    # 加载Amazon Co-purchase数据集
    # nodes: 13,752
    # edges: 491,722
    dataset = Amazon(
        root='/data/Cora',  # 指定数据集存储路径
        name='Computers'  # 指定类别，例如'Computers'
    )
    data = dataset[0]

    # 创建图
    G = to_networkx(data, to_undirected=True)
    return G, data