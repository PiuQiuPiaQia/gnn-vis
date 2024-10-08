from sklearn.manifold import TSNE
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
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
embeddings = np.load("cora_embeddings.npy")

# 初始化t-SNE对象
tsne = TSNE(n_components=2, random_state=0)

# 进行降维处理
tsne_result = tsne.fit_transform(embeddings)

# 可视化降维结果
plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])# 绘制边
edge_color = 'gray'
for edge in G.edges():
    source, target = edge
    source_idx = list(embeddings.keys()).index(source)
    target_idx = list(embeddings.keys()).index(target)
    plt.plot([tsne_result[source_idx, 0], tsne_result[target_idx, 0]], [tsne_result[source_idx, 1], tsne_result[target_idx, 1]], color=edge_color)

plt.title('t-SNE visualization of node embeddings with edges')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()

# ------------ 采样 -------------

def adaptive_blue_noise_sampling(points, k=4, radius_scale=1.0):
    """
    自适应蓝噪声采样
    :param points: t-SNE降维后的点集
    :param k: 每个点的最近邻数，用于计算密度
    :param radius_scale: 半径的比例因子
    :return: 采样点的索引
    """
    # 计算k最近邻距离，用于估计密度
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)
    distances, indices = neigh.kneighbors(points)

    # 计算密度，即1/k最近邻距离
    densities = 1.0 / distances[:, -1].squeeze()

    # 计算每个点的Poisson圆盘半径
    radius = np.mean(distances[:, -1]) * radius_scale

    # 创建网格
    grid_x, grid_y = np.mgrid[0:1:128j, 0:1:128j]
    points_grid = np.array([grid_x.ravel(), grid_y.ravel()]).T

    # 计算采样概率
    sampling_prob = np.minimum(1.0, densities / np.max(densities))
    sampling_mask = generic_filter(sampling_prob, lambda arr: np.sum(arr <= radius * radius_scale), size=(128, 128),
                                   mode='constant')

    # 选择采样点
    sampled_indices = np.where(sampling_mask.ravel())
    sampled_points_indices = np.random.choice(sampled_indices[0], size=int(len(sampled_indices[0]) * 0.05),
                                              replace=False)

    return sampled_points_indices


# # 使用自适应蓝噪声采样
# sampled_indices = adaptive_blue_noise_sampling(tsne_result)
#
# # 可视化采样结果
# plt.figure(figsize=(10, 8))
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='lightgray', edgecolor='none', s=1)
# plt.scatter(tsne_result[sampled_indices, 0], tsne_result[sampled_indices, 1], c='red', edgecolor='none', s=10)
# plt.title('Adaptive Blue Noise Sampling')
# plt.xlabel('t-SNE feature 1')
# plt.ylabel('t-SNE feature 2')
# plt.show()