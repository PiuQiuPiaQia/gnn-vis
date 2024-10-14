import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from load_dataset import load_dataset

# 创建Cora图
G, data = load_dataset()

# 加载之前保存的节点嵌入向量
# embeddings = np.load("./cora_embeddings.npy")
embeddings = np.load("./amazon_embeddings.npy")

# print(embeddings)

# 初始化t-SNE对象
tsne = TSNE(n_components=2, random_state=0)
tsne_result = tsne.fit_transform(embeddings)

# 可视化t-SNE结果
plt.figure(figsize=(15, 7))

# 绘制原始图
plt.subplot(1, 2, 1)
plt.scatter(embeddings[:, 0], embeddings[:, 1], c='lightblue', edgecolor='none', s=5)
plt.title('Original Embedding Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 绘制边
edge_color = 'gray'
for edge in G.edges():
    source_idx = list(G.nodes()).index(edge[0])
    target_idx = list(G.nodes()).index(edge[1])
    # 由于原始特征可能不是二维的，这里仅示意，实际可能需要根据实际维度调整
    plt.plot([embeddings[source_idx, 0], embeddings[target_idx, 0]], [embeddings[source_idx, 1], embeddings[target_idx, 1]], color=edge_color, alpha=0.1)

# 绘制t-SNE降维后的图
plt.subplot(1, 2, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='lightgreen', edgecolor='none', s=5)
plt.title('t-SNE Reduced Space')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')

# 绘制边
for edge in G.edges():
    source_idx = list(G.nodes()).index(edge[0])
    target_idx = list(G.nodes()).index(edge[1])
    plt.plot([tsne_result[source_idx, 0], tsne_result[target_idx, 0]], [tsne_result[source_idx, 1], tsne_result[target_idx, 1]], color=edge_color, alpha=0.1)

# 保存图像到文件
# plt.savefig('cora-t-sne-comparison.png', dpi=300)  # dpi参数控制图像质量
plt.savefig('amazon-t-sne-comparison.png', dpi=300)  # dpi参数控制图像质量
plt.show()  # 显示图像


# ------------ 保存 t-SNE 结果到文件 ------------

# 保存 tsne_result 为 .npy 文件
# np.save("cora-tsne_results.npy", tsne_result)
np.save("amazon-tsne_results.npy", tsne_result)

# 打印消息确认保存成功
print("t-SNE results have been saved to 'tsne_results.npy'")