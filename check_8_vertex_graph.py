import numpy as np
import math
import random
import networkx as nx
import copy
import matplotlib.pyplot as plt
from itertools import combinations
import csv
from tqdm import tqdm


class point:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.oldx = 0
        self.oldy = 0

    def make_point_random(self):
        self.x = random.randint(0, 20)
        self.y = random.randint(0, 20)
        self.oldx = self.x
        self.oldy = self.y

    def distance(self, another_point):
        return math.sqrt((self.x - another_point.x) ** 2 + (self.y - another_point.y) ** 2)

    def __str__(self):
        return 'x:' + str(self.oldx) + ',y:' + str(self.oldy)

    def extend_point(self, x_add=0, y_add=0):
        self.x += x_add
        self.y += y_add

    def at_time_t(self, another_point, t=0, total_time=100):
        self.x = (self.oldx * (total_time - t) + another_point.oldx * t) / total_time
        self.y = (self.oldy * (total_time - t) + another_point.oldy * t) / total_time


def _expand(G, explored_nodes, explored_edges):
    """
    Expand existing solution by a process akin to BFS.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    explored_nodes: set of ints
        nodes visited

    explored_edges: set of 2-tuples
        edges visited

    Returns:
    --------
    solutions: list, where each entry in turns contains two sets corresponding to explored_nodes and explored_edges
        all possible expansions of explored_nodes and explored_edges

    """
    frontier_nodes = list()
    frontier_edges = list()
    for v in explored_nodes:
        for u in nx.neighbors(G, v):
            if not (u in explored_nodes):
                frontier_nodes.append(u)
                frontier_edges.append([(u, v), (v, u)])

    return zip([explored_nodes | frozenset([v]) for v in frontier_nodes],
               [explored_edges | frozenset(e) for e in frontier_edges])


def find_all_spanning_trees(G, root=0):
    """
    Find all spanning trees of a Graph.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    Returns:
    ST: list of networkx.Graph() instances
        list of all spanning trees

    """

    # initialise solution
    explored_nodes = frozenset([root])
    explored_edges = frozenset([])
    solutions = [(explored_nodes, explored_edges)]
    # we need to expand solutions number_of_nodes-1 times
    for ii in range(G.number_of_nodes() - 1):
        # get all new solutions
        solutions = [_expand(G, nodes, edges) for (nodes, edges) in solutions]
        # flatten nested structure and get unique expansions
        solutions = set([item for sublist in solutions for item in sublist])

    return [nx.from_edgelist(edges) for (nodes, edges) in solutions]


max_of_change = 0
for iterations in tqdm(range(10000)):
    G = nx.Graph()
    all_points = []
    pos = {}
    for i in range(8):
        p = point()
        p.make_point_random()
        all_points.append(copy.deepcopy(p))
        pos[all_points[i]] = (p.x, p.y)
        G.add_node(all_points[i])

    edges = list(combinations(all_points, 2))
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[0].distance(edge[1]))

    random_point_name = [(6, 3), (4, 1), (0, 2),(5,7)]
    random_point_neighbours = []
    for name in random_point_name:
        random_point_neighbours.append((all_points[name[0]], all_points[name[1]]))

    inf = 10000

    mmst_cost = inf
    for spanning_tree in find_all_spanning_trees(G, root=all_points[0]):
        tree_edges = spanning_tree.edges()
        max_cost = 0
        for t in range(21):
            H = nx.Graph()
            pos = {}
            for neighbours in random_point_neighbours:
                neighbours[0].at_time_t(neighbours[1], t, 20)
            for i in range(8):
                p = all_points[i]
                pos[i] = (p.x, p.y)
                H.add_node(i)

            for edge in tree_edges:
                H.add_edge(edge[0], edge[1], weight=edge[0].distance(edge[1]))
            cost = H.size(weight="weight")
            if cost > max_cost:
                max_cost = cost
                # max_graph=H.copy()
        if max_cost < mmst_cost:
            mmst_cost = max_cost
            # graph = max_graph.copy()
    # print(mmst_cost)

    # return to oringinal position
    for each in all_points:
        each.x = each.oldx
        each.y = each.oldy
    max_weight_matrix = [[0 for each in all_points] for each in all_points]
    for t in range(21):
        # H=nx.Graph()
        # pos={}
        for neighbours in random_point_neighbours:
            neighbours[0].at_time_t(neighbours[1], t, 20)
        for i in range(len(all_points)):
            for j in range(len(all_points)):
                if i != j:
                    weight = all_points[i].distance(all_points[j])
                    if weight > max_weight_matrix[i][j]:
                        max_weight_matrix[i][j] = weight

    H = nx.Graph()
    for i in range(len(all_points)):
        for j in range(len(all_points)):
            if i != j:
                H.add_edge(all_points[i], all_points[j], weight=max_weight_matrix[i][j])
    T = nx.minimum_spanning_tree(H)
    # print(T.size(weight="weight"))
    weightT = T.size(weight="weight")

    tree_edges = T.edges()
    max_cost = 0
    for t in range(21):
        H = nx.Graph()
        pos = {}
        for neighbours in random_point_neighbours:
            neighbours[0].at_time_t(neighbours[1], t, 20)
        for i in range(8):
            p = all_points[i]
            pos[i] = (p.x, p.y)
            H.add_node(i)

        for edge in tree_edges:
            H.add_edge(edge[0], edge[1], weight=edge[0].distance(edge[1]))
        cost = H.size(weight="weight")
        if cost > max_cost:
            max_cost = cost
    # print(max_cost)
    # print(max_cost / mmst_cost)
    if max_of_change < max_cost / mmst_cost:
        max_of_change = max_cost / mmst_cost
    with open('record8new.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        row = [str(x) for x in all_points]
        row.append(str(random_point_name))
        row.extend([mmst_cost, max_cost, max_cost / mmst_cost])
        # print(iterations, row)
        csvwriter.writerow(row)
print(max_of_change)
