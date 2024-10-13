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
        edges = list(G.edges())
        sampled_edges = set()
        while len(sampled_edges) < num_samples:
            edge = random.choice(edges)
            if edge not in sampled_edges and (edge[1], edge[0]) not in sampled_edges:
                sampled_edges.add(edge)
        sampled_nodes = set()
        for u, v in sampled_edges:
            sampled_nodes.add(u)
            sampled_nodes.add(v)
        return G.subgraph(sampled_nodes)

    num_samples = 1000  # 你可以根据需要调整采样的边数
    G_sampling = ties_sampling(G, num_samples)

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