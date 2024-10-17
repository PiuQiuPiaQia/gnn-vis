import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils.blue_noise import blueNoise
from load_dataset import load_dataset

# 创建Cora图
G, data = load_dataset()

# 绘制矩形区域
def draw_box(ax, region, color):
    rect = plt.Rectangle((region['lng_min'], region['lat_max']),
                         region['lng_max'] - region['lng_min'],
                         region['lat_min'] - region['lat_max'], fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)

if __name__ == '__main__':
    # 设置日志记录
    logging.basicConfig(level=logging.INFO)

    # 读取 t-SNE 结果
    tsne_result = np.load("amazon-tsne_results.npy")
    points = [{'id': i, 'lat': float(point[0]), 'lng': float(point[1])} for i, point in enumerate(tsne_result)]

    # 进行蓝噪声采样
    samplePoints = blueNoise(points, 8)
    # 获取采样点的ID
    sampled_node_ids = [point['id'] for point in samplePoints]
    # 创建采样子图
    sampled_subgraph = G.subgraph(sampled_node_ids)

    # 保存采样结果
    recentBlueNoiseFilePath = f'./samplePoints-35.json'
    with open(recentBlueNoiseFilePath, 'w', encoding='utf-8') as f:
        logging.info(f'35 sampling over')
        logging.info('-------------------')
        f.write(json.dumps(samplePoints, indent=4))

    # 定义左上角区域
    top_left_region = {'lat_max': 32, 'lng_min': -96, 'lat_min': 24, 'lng_max': -88}

    # 创建绘图
    plt.figure(figsize=(12, 8))

    # 绘制原图
    pos = {node: (point['lng'], point['lat']) for node, point in zip(range(len(points)), points)}
    print(len(points))
    nx.draw_networkx(G, pos, node_color='black', node_size=10, edge_color='lightgray', with_labels=False)
    plt.title('Original Graph')

    # 在原图上标记左上角区域
    ax = plt.gca()
    draw_box(ax, top_left_region, 'r')

    # 创建另一个绘图以展示采样后的图
    plt.figure(figsize=(12, 8))
    # 绘制采样点
    sampled_nodes = set([point['id'] for point in samplePoints])
    nx.draw_networkx_nodes(sampled_subgraph, pos, nodelist=sampled_nodes, node_color='black', node_size=10)
    nx.draw_networkx_edges(sampled_subgraph, pos, edge_color='lightgray', width=0.5, alpha=0.5)
    plt.title('Sampled Graph')
    # 在原图上标记左上角区域
    ax = plt.gca()
    draw_box(ax, top_left_region, 'r')


    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Top-Left Region', 'Sampled Points'])
    plt.tight_layout()
    plt.savefig(f'./sampled-graph.png')
    plt.show()
    plt.close()

    logging.info('Plot saved for sampled graph')
    logging.info(f"Sampling rate: {len(samplePoints) / len(points) * 100:.2f}%")