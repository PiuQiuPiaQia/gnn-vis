from blue_noise import blueNoise
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for r in [100000, 25000, 6000, 1250, 300]:
        t1 = time.time()

        # 读取 t-SNE 结果
        tsne_result = np.load("tsne_results.npy")
        points = [{'id': i, 'lat': float(point[0]), 'lng': float(point[1])} for i, point in enumerate(tsne_result)]

        # 进行蓝噪声采样
        samplePoints = blueNoise(points, r)

        # # 保存采样结果
        # recentBlueNoiseFilePath = f'./samplePoints-{r}-{len(samplePoints)}-{len(samplePoints) / len(points)}.json'
        # with open(recentBlueNoiseFilePath, 'w', encoding='utf-8') as f:
        #     logging.info(f'{r} sampling over, {(time.time() - t1) / 60} minutes')
        #     logging.info('-------------------')
        #     f.write(json.dumps(samplePoints))

        # 绘制采样点并保存图像
        plt.figure()
        plt.scatter([point['lng'] for point in samplePoints], [point['lat'] for point in samplePoints], s=10)
        plt.title(f'Blue Noise Sampling with {r} points')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f'./samplePoints-{r}.png')
        plt.show()
        plt.close()

        logging.info(f'Plot saved for {r} points')