import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

mpl.use('Agg')

# 指定数据集存储路径
dataset_path = './data'

# 加载 Cora 数据集
dataset = Planetoid(root=dataset_path, name='Cora')

# 获取 Cora 数据集的一个数据对象
data = dataset[0]

num_nodes = data.num_nodes
edge_index = data.edge_index

print(num_nodes)
print(edge_index[0].shape)

# 将 PyG Data 对象转换为 NetworkX 图
G = to_networkx(data, to_undirected=True)

print("G")
print(len(G.nodes()))
print(len(G.edges()))



############################# 采样 #############################

def ties_sampling(G, num_samples):
    # 从图中随机选择边
    edges = list(G.edges())
    sampled_edges = set()
    while len(sampled_edges) < num_samples:
        edge = random.choice(edges)
        u, v = edge
        # 确保不会重复选择相同的边
        if edge not in sampled_edges and (u, v) not in sampled_edges:
            sampled_edges.add(edge)
    
    # 基于选择的边，选择对应的节点
    sampled_nodes = set()
    for edge in sampled_edges:
        sampled_nodes.add(edge[0])
        sampled_nodes.add(edge[1])
    
    # 创建采样子图
    subgraph_nodes = list(sampled_nodes)
    subgraph = G.subgraph(subgraph_nodes)
    return subgraph

# 使用TIES采样算法对Cora数据集的图进行采样
num_samples = 100  # 你可以根据需要调整采样的边数
G_sampling = ties_sampling(G, num_samples)


print("G_sampling")
print(len(G_sampling.nodes()))
print(len(G_sampling.edges()))


#############################  可视化  #############################

def visualize_graph(G, pos, title):
    nx.draw(G, pos, with_labels=False, node_color='skyblue', edge_color='red', node_size=20)
    plt.title(title)

# 创建一个图形窗口，并设置大小
plt.figure(figsize=(18, 6))

# 第一张子图：原始图
plt.subplot(1, 3, 1)  # (行数，列数，当前子图的索引)
pos_original = nx.spring_layout(G, seed=42, k=0.5)  # 为原始图计算布局
visualize_graph(G, pos_original, 'Original Graph')

# 第二张子图：样本边图
plt.subplot(1, 3, 2)
pos_positive = nx.spring_layout(G_sampling, seed=42, k=0.5)  # 为正样本图计算布局
visualize_graph(G_sampling, pos_positive, 'G_sampling Sample Graph')


# plt.plot(range(20))
plt.savefig("./Picture1.png") 



def visualize_subgraph(G, selected_node, title):
    subgraph = G.subgraph(list(G.neighbors(selected_node)) + [selected_node])
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', edge_color='red', node_size=20)
    plt.title(title)

# 从采样图中随机选择一个存在的节点
sampling_nodes = list(G_sampling.nodes())
if sampling_nodes:
    selected_node = random.choice(sampling_nodes)
else:
    selected_node = None

# 创建一个图形窗口，并设置大小
plt.figure(figsize=(12, 6))

# 第一张子图：原始图中 42 号节点的子图
plt.subplot(1, 2, 1)
visualize_subgraph(G, selected_node, f'Node {selected_node} Subgraph in Original Graph')

# 第二张子图：采样图中 42 号节点的子图
plt.subplot(1, 2, 2)
visualize_subgraph(G_sampling, selected_node, f'Node {selected_node} Subgraph in Sampling Graph')

plt.savefig("./Picture2.png") 



# # 显示图形窗口
# plt.show()