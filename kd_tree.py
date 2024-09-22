import sys
import matplotlib.pyplot as plt
import numpy as np
import heapq
%matplotlib inline

from typing import List

class KDNode:
  def __init__(self, data, left, right):
    self.data = data
    self.left = left
    self.right = right

  def __init__(self, data):
    self.data = data
    self.left = None
    self.right = None

class KDTree():
  def __init__(self, K, dim):
    self.K = K
    self.root = None
    self.axis = 0
    self.min = sys.maxsize
    self.results = []
    self.dim = dim

  def get_root(self):
    return self.root

  def dimension(self, point):
    return len(point)

  def distance(self, point1, point2):
    distance = 0
    dimension = self.dimension(point1)

    for i in range(dimension):
      distance += (point1[i] - point2[i]) * (point1[i] - point2[i])

    return distance

  def plane_distance(self, point1, point2, axis):
    return abs(point1[axis] - point2[axis])

  def build_tree_from_points(self, points, axis):
    if len(points) > 1:
      points.sort(key=lambda point: point[axis])
      mid = int(len(points) / 2)

      node = KDNode(points[mid])
      next_axis = (axis + 1) % self.dim

      node.left = self.build_tree_from_points(points[:mid], next_axis)
      node.right = self.build_tree_from_points(points[mid + 1:], next_axis)

      return node

    if len(points) == 1:
      return KDNode(points[0])

  def nn_search(self, query_point, curr, axis):
    if curr == None:
      return

    if self.distance(query_point, curr.data) < self.min:
      self.min = self.distance(query_point, curr.data)
    
    is_left = False
    next_axis = (axis + 1) % self.dim

    if query_point[axis] <= curr.data[axis]:
      self.nn_search(query_point, curr.left, next_axis)
      is_left = True
    else:
      self.nn_search(query_point, curr.right, next_axis)
  
    plane_dist = self.plane_distance(query_point, curr.data, axis)

    if self.min > plane_dist:
      if is_left == True: 
        self.nn_search(query_point, curr.right, next_axis)
      else:
        self.nn_search(query_point, curr.left, next_axis)

  
  def knn_search(self, query_point, curr, axis):
    if curr == None:
      return

    squared_dist = self.distance(query_point, curr.data)

    if len(self.results) >= self.K:
      if -squared_dist > self.results[0]:
        heapq.heapreplace(self.results, -squared_dist)
    else:
      heapq.heappush(self.results, -squared_dist)
    
    is_left = False
    next_axis = (axis + 1) % self.dim

    if query_point[axis] <= curr.data[axis]:
      self.knn_search(query_point, curr.left, next_axis)
      is_left = True
    else:
      self.knn_search(query_point, curr.right, next_axis)
  
    plane_dist = self.plane_distance(query_point, curr.data, axis)

    if self.min > plane_dist:
      if is_left == True: 
        self.nn_search(query_point, curr.right, next_axis)
      else:
        self.nn_search(query_point, curr.left, next_axis)
  
  def plot_subplanes(self, curr, axis):
    goes_through = curr.data[axis]
    if axis == 0:
      plt.axvline(x=goes_through, color='r', linewidth=0.4)
    elif axis == 1:
      plt.axhline(y=goes_through, color='b', linewidth=0.4)

    new_axis = (axis + 1) % self.dim

    if curr.left != None:
      self.plot_subplanes(curr.left, new_axis)
    if curr.right != None:
      self.plot_subplanes(curr.right, new_axis)
  
def plot_points(points: List[int]):
  plt.plot(points[0], points[1])
