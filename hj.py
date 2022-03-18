# import numpy as np
import math
import random
import networkx as nx
# import copy
# import matplotlib.pyplot as plt
# from itertools import combinations
# import csv
# from tqdm import tqdm


class Point:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.oldx = 0
        self.oldy = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.oldx = self.x
        self.oldy = self.y

    def make_point_random(self):
        self.x = random.randint(0, 5)
        self.y = random.randint(0, 5)
        self.oldx = self.x
        self.oldy = self.y

    def distance(self, another_point):
        return math.sqrt((self.x - another_point.x) ** 2 + (self.y - another_point.y) ** 2)

    def __str__(self):
        return 'x:' + str(self.x) + ',y:' + str(self.y)

    def extend_point(self, x_add=0, y_add=0):
        self.x += x_add
        self.y += y_add

    def at_time_t(self, another_point, t=0, total_time=100):
        self.x = (self.oldx * (total_time - t) + another_point.oldx * t) / total_time
        self.y = (self.oldy * (total_time - t) + another_point.oldy * t) / total_time


all_points = [Point(0,1), Point(60,1), Point(0, 0), Point(30,0)]
random_point_name = [(2, 3), (1, 0)]
random_point_neighbours = []
dict_point_neighbours = {}
for name in random_point_name:
    random_point_neighbours.append((all_points[name[0]], all_points[name[1]]))
    dict_point_neighbours[all_points[name[0]]] = all_points[name[1]]
keys = list(dict_point_neighbours.keys())
keys.sort(key=lambda y: y.y)
keys.sort(key=lambda y: y.x)
new_spanning_tree = nx.Graph()
for i in range(1, len(keys)):
    new_spanning_tree.add_edge(keys[i - 1], keys[i])
new_spanning_tree.add_edge(keys[-1], dict_point_neighbours[keys[-1]])
for i in range(len(keys) - 2, -1,-1):
    new_spanning_tree.add_edge(dict_point_neighbours[keys[i + 1]], dict_point_neighbours[keys[i]])
tree_edges = new_spanning_tree.edges()
maximum_cost = 0
for t in range(20):
    H = nx.Graph()
    pos = {}
    for neighbours in random_point_neighbours:
        neighbours[0].at_time_t(neighbours[1], t, 20)
    for i in range(4):
        p = all_points[i]
        pos[i] = (p.x, p.y)
        H.add_node(i)

    for edge in tree_edges:
        H.add_edge(edge[0], edge[1], weight=edge[0].distance(edge[1]))
    cost = H.size(weight="weight")
    if cost > maximum_cost:
        maximum_cost = cost
print(maximum_cost)

old_spanning_tree = nx.Graph()
old_spanning_tree.add_edge(all_points[0],all_points[2])
old_spanning_tree.add_edge(all_points[1],all_points[3])
old_spanning_tree.add_edge(all_points[2],all_points[3])
tree_edges = old_spanning_tree.edges()
maximum_cost = 0
for t in range(20):
    H = nx.Graph()
    pos = {}
    for neighbours in random_point_neighbours:
        neighbours[0].at_time_t(neighbours[1], t, 20)
    for i in range(4):
        p = all_points[i]
        pos[i] = (p.x, p.y)
        H.add_node(i)

    for edge in tree_edges:
        H.add_edge(edge[0], edge[1], weight=edge[0].distance(edge[1]))
    cost = H.size(weight="weight")
    if cost > maximum_cost:
        print(t,cost)
        maximum_cost = cost
print(maximum_cost)
