import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from blue_noise import blueNoise

# 加载Cora数据集
def load_cora():
    dataset = Planetoid(root='./data/Cora', name='Cora')
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
    tsne_result = np.load("tsne_results.npy")
    points = [{'id': i, 'lat': float(point[0]), 'lng': float(point[1])} for i, point in enumerate(tsne_result)]

    # 进行蓝噪声采样
    samplePoints = blueNoise(points, 10)

    # 保存采样结果
    recentBlueNoiseFilePath = f'./samplePoints-10.json'
    with open(recentBlueNoiseFilePath, 'w', encoding='utf-8') as f:
        logging.info(f'10 sampling over')
        logging.info('-------------------')
        f.write(json.dumps(samplePoints, indent=4))

    # 定义左上角区域
    top_left_region = {'lat_max': 32, 'lng_min': -59, 'lat_min': 28, 'lng_max': -54}

    # 创建绘图
    plt.figure(figsize=(12, 8))

    # 绘制原图
    pos = {node: (point['lng'], point['lat']) for node, point in zip(range(len(points)), points)}
    nx.draw_networkx(G, pos, node_color='black', node_size=10, edge_color='lightgray', with_labels=False)
    plt.title('Original Graph with Top-Left Region')

    # 在原图上标记左上角区域
    draw_box(plt.gca(), top_left_region, 'r')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Top-Left Region'])
    plt.tight_layout()
    plt.savefig(f'./top-left-region-sampling.png')
    plt.show()
    plt.close()

    logging.info('Plot saved for top-left region sampling')

    logging.info(top_left_region)