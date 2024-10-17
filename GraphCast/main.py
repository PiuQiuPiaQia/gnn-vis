import logging

import networkx as nx
from matplotlib import pyplot as plt

from transformDataset import tansformEra5
from utils.blue_noise import blueNoise

G, data = tansformEra5()

print(len(G.nodes()), len(data.pos))

# 将data.pos转换为字典列表，以符合blueNoise函数的输入要求
points = []
for i, pos in enumerate(data.pos):
    points.append({'id': i, 'lat': pos[0], 'lng': pos[1]})

# 使用blueNoise函数进行采样
sampled_points = blueNoise(points, 30)

# 创建一个新的图对象，只包含采样的节点
sampled_G = G.subgraph([p['id'] for p in sampled_points])

# 添加采样点对应的边
for point in sampled_points:
    for neighbor in G.neighbors(point['id']):
        if neighbor in sampled_G.nodes():  # 确保邻居节点也在采样图中
            sampled_G.add_edge(point['id'], neighbor)

# 获取采样节点的位置信息
sampled_pos = {p['id']: (p['lat'], p['lng']) for p in sampled_points}

# 绘制采样后的图
plt.figure(figsize=(8, 8))
nx.draw(sampled_G, sampled_pos, with_labels=True, node_color='skyblue', node_size=50, edge_color='k', linewidths=1, font_size=8)
plt.title('GraphCast Visualization With Blue Noise Sampling')
plt.savefig('GraphCast-BlueNoiseSampling.png')
plt.show()
plt.close()

logging.info('Plot saved for sampled graph')
logging.info(f"Sampling rate: {len(sampled_points) / len(points) * 100:.2f}%")
