import time
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

if __name__ == '__main__':
    # 设置日志记录
    logging.basicConfig(level=logging.INFO)

    # 加载Cora图和数据
    G, data = load_cora()

    for r in [10]:
        t1 = time.time()

        # 读取 t-SNE 结果
        tsne_result = np.load("tsne_results.npy")
        points = [{'id': i, 'lat': float(point[0]), 'lng': float(point[1])} for i, point in enumerate(tsne_result)]

        # 进行蓝噪声采样
        samplePoints = blueNoise(points, r)

        # 保存采样结果
        recentBlueNoiseFilePath = f'./samplePoints-{r}.json'
        with open(recentBlueNoiseFilePath, 'w', encoding='utf-8') as f:
            logging.info(f'{r} sampling over, {(time.time() - t1) / 60:.2f} minutes')
            logging.info('-------------------')
            f.write(json.dumps(samplePoints, indent=4))

        # 绘制采样点和连接关系
        plt.figure(figsize=(10, 8))
        sample_point_ids = [point['id'] for point in samplePoints]

        # 创建采样点的子图
        sample_graph = G.subgraph(sample_point_ids)
        pos = {node: (point['lng'], point['lat']) for node, point in zip(sample_point_ids, samplePoints)}

        # 绘制节点
        nx.draw_networkx_nodes(sample_graph, pos, node_color='lightblue', node_size=50)
        # 绘制边
        nx.draw_networkx_edges(sample_graph, pos, edge_color='gray', width=0.5, alpha=0.5)
        # 绘制节点标签
        nx.draw_networkx_labels(sample_graph, pos, font_size=8, font_color='black')

        plt.title(f'Blue Noise Sampling with radius {r}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'./samplePoints-{r}-Graph.png')
        plt.show()
        plt.close()

        logging.info(f'Plot saved for radius {r}')

        # 随机选择一个采样点
        pick = np.random.choice(sample_point_ids)
        logging.info(f'Selected point: {pick}')

        # 展示pick及连接的点和边的信息
        neighbors_in_sample = list(sample_graph.neighbors(pick))
        logging.info(f'Neighbors of {pick} in sampled graph: {neighbors_in_sample}')
        logging.info(f'Edges of {pick} in sampled graph: {list((pick, neighbor) for neighbor in neighbors_in_sample)}')

        # 展示pick点在t-SNE图中连接的点和边的信息
        idx = int(pick)
        neighbors_in_tsne = [int(point['id']) for point in points if (idx, point['id']) in G.edges or (point['id'], idx) in G.edges]
        logging.info(f'Neighbors of {pick} in t-SNE graph: {neighbors_in_tsne}')
        logging.info(f'Edges of {pick} in t-SNE graph: {list((pick, neighbor) for neighbor in neighbors_in_tsne)}')