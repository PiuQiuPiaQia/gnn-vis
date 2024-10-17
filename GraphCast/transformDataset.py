from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.data import Data  # 确保导入Data类
from era52Geometric import loadEra5Dateset

# 加载ERA5数据集
G, data = loadEra5Dateset()

# 显示原始数据
print(data)

# 使用Laplacian Eigenvector PE转换
transform = AddLaplacianEigenvectorPE(k=3)
data_transformed_Laplacian = transform(data)

# 显示转换后的数据
print(data_transformed_Laplacian)

# 使用Random Walk PE转换
transform2 = AddRandomWalkPE(walk_length=30)
data_transformed_Random = transform2(data)

# 显示第二次转换后的数据
print(data_transformed_Random)

# 保存转换后的数据到文件系统
data_transformed_Laplacian.save('data_transformed_Laplacian.pt')
data_transformed_Random.save('data_transformed_Random.pt')

# 之后你可以使用Data类的load方法重新加载这些数据
# new_data_transformed = Data.load('data_transformed_Laplacian.pt')
# new_data_transformed2 = Data.load('data_transformed_Random.pt')