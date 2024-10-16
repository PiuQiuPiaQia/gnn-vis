from torch_geometric.transforms import  AddLaplacianEigenvectorPE, AddRandomWalkPE
from era52Geometric import loadEra5Dateset

# AddLaplacianEigenvectorPE: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddLaplacianEigenvectorPE.html?highlight=transform+ran

G,data = loadEra5Dateset()

print(data)

# 假设 data 是一个 PyTorch Geometric Data 对象
transform = AddLaplacianEigenvectorPE(k=3)
data_transformed = transform(data)

print(data_transformed)

# 假设 data 是一个 PyTorch Geometric Data 对象
transform2 = AddRandomWalkPE(walk_length=30)
data_transformed2 = transform2(data)

print(data_transformed2)