import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

# 加载Cora数据集
def load_cora():
    dataset = Planetoid(root='../Adaptive Blue Noise Sampling/data/Cora', name='Cora')
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    return G, data

# 绘制矩形区域
def draw_box(ax, region, color):
    rect = plt.Rectangle((region['lng_min'], region['lat_max']),
                         region['lng_max'] - region['lng_min'],
                         region['lat_min'] - region['lat_max'], fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)

if __name__ == '__main__':
    # 设置日志记录
    logging.basicConfig(level=logging.INFO)

    # 加载Cora图和数据
    G, data = load_cora()

    # 读取 t-SNE 结果
    tsne_result = np.load("../Adaptive Blue Noise Sampling/tsne_results.npy")
    points = [{'id': i, 'lat': float(point[0]), 'lng': float(point[1])} for i, point in enumerate(tsne_result)]

    # TIES采样算法

    def ties_sampling(G, num_samples):
        nodes = list(G.nodes())  # 获取所有节点的列表
        sampled_nodes = set()  # 初始化一个空集合来存储采样的节点

        # 循环直到采样的节点数量达到num_samples
        while len(sampled_nodes) < num_samples:
            node = random.choice(nodes)  # 从所有节点中随机选择一个节点
            if node not in sampled_nodes:  # 如果该节点尚未被采样
                sampled_nodes.add(node)  # 将其添加到采样节点集合中

        # 创建采样子图，包含所有采样的节点及其相关的边
        subgraph = G.subgraph(sampled_nodes)
        return subgraph

    # 计算原图中点数量的10%
    num_sampling_nodes = int(0.34 * len(G.nodes()))
    G_sampling = ties_sampling(G, num_sampling_nodes)

    # 定义左上角区域
    top_left_region = {'lat_max': 32, 'lng_min': -59, 'lat_min': 28, 'lng_max': -54}

    # 创建绘图
    plt.figure(figsize=(12, 8))

    # 绘制采样图
    pos = {node: (point['lng'], point['lat']) for node, point in zip(G_sampling.nodes(), [points[node] for node in G_sampling.nodes()])}
    nx.draw_networkx(G_sampling, pos, node_color='black', node_size=10, edge_color='lightgray', with_labels=False)
    plt.title('TIES Sampling Graph with Top-Left Region')

    # 在采样图上标记左上角区域
    ax = plt.gca()
    draw_box(ax, top_left_region, 'r')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Top-Left Region'])
    plt.tight_layout()
    plt.savefig(f'./ties-top-left-region-sampling.png')
    plt.show()
    plt.close()

    logging.info('Plot saved for TIES top-left region sampling')

    logging.info(f"采样率：{len(G_sampling.nodes()) / len(points) * 100:.2f}%")