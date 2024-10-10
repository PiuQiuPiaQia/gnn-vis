import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

# 加载Cora数据集
def load_cora():
    # 加载Cora数据集
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]

    # 创建图
    G = to_networkx(data, to_undirected=True)
    return G

# 创建Cora图
G = load_cora()

# 加载之前保存的节点嵌入向量
embeddings = np.load("./cora_embeddings.npy")

# 初始化t-SNE对象
tsne = TSNE(n_components=2, random_state=0)
tsne_result = tsne.fit_transform(embeddings)

# 可视化t-SNE结果
plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='lightblue', edgecolor='none', s=5)

# 绘制边
edge_color = 'gray'
for edge in G.edges():
    source_idx = list(G.nodes()).index(edge[0])
    target_idx = list(G.nodes()).index(edge[1])
    plt.plot([tsne_result[source_idx, 0], tsne_result[target_idx, 0]], [tsne_result[source_idx, 1], tsne_result[target_idx, 1]], color=edge_color, alpha=0.1)

plt.title('t-SNE visualization of node embeddings with edges')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')

# 保存图像到文件
plt.savefig('t-sne-cora-embeddings.png', dpi=300)  # dpi参数控制图像质量
