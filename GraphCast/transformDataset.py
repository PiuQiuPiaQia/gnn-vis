from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.data import Data  # 确保导入Data类
from torch_geometric.utils import to_networkx

from era52Geometric import loadEra5Dateset

def tansformEra5():
    # 加载ERA5数据集
    G, data = loadEra5Dateset()

    # 显示原始数据
    # print(data)

    # 使用Laplacian Eigenvector PE转换
    transform = AddLaplacianEigenvectorPE(k=3, is_undirected=True)
    data_transformed_Laplacian = transform(data)
    # 创建图
    G = to_networkx(data)

    # 显示转换后的数据
    print(data_transformed_Laplacian)


    # 使用Random Walk PE转换
    # transform2 = AddRandomWalkPE(walk_length=30)
    # data_transformed_Random = transform2(data)
    # 显示第二次转换后的数据
    # print(data_transformed_Random)

    return G, data_transformed_Laplacian