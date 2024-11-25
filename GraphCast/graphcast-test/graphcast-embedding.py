import pickle


with open('./graphcast-graph/grid2mesh_graph.pkl', 'rb') as f:
    grid2mesh_graph = pickle.load(f)

print(grid2mesh_graph.nodes["grid_nodes"])
print(grid2mesh_graph.nodes["mesh_nodes"])
print(grid2mesh_graph)