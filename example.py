from kd_tree import *

points = [[1.5, 2], [1, 2], [3, 4], [5, 3], [4, 3]]
print(sorted(points, key = lambda points: points[0]))
numpy_points = np.array(points)

tree = KDTree(3, 2)

root = tree.get_root()

tree.root = tree.build_tree_from_points(points, 0)

query_point = [2, 2]
tree.nn_search(query_point, tree.root, 0)
print(tree.min)

tree.knn_search(query_point, tree.root, 0)
print(tree.results)

tree.plot_subplanes(tree.root, 0)