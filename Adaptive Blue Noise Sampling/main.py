import networkx as nx
from torch_geometric.datasets import Planetoid
from node2vec import Node2Vec
import numpy as np

# 加载Cora数据集
def load_cora():
    # 加载Cora数据集
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]

    # 创建图
    G = data.to_networkx()
    return G

# 创建Cora图
G = load_cora()

# 应用node2vec模型
node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 获取节点向量表示
embeddings = {node: model.wv[node] for node in G.nodes()}

# 打印某个节点的向量表示
print(embeddings[1])  # 假设节点1的向量表示

# 保存嵌入向量到文件
np.save("cora_embeddings.npy", np.array(list(embeddings.values())))