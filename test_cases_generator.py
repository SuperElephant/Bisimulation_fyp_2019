# this file is use to generate the test case ([flag, nx.DiGraph, nx.DiGraph])
# dependent on bisim_PnT.py

import math
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def test_cases_generator(min_node_number=3):
    n = min_node_number
    G1 = nx.fast_gnp_random_graph(n,0.5,directed=True)
    # TODO: generate test case base on a random digraph

def random_labeled_digraph(n, m, p):
    '''
    return a random labeled directed graph
    modify from networkx.generators.random_graphs
    https://github.com/networkx/networkx/blob/master/networkx/generators/random_graphs.py

    :param n: number of node
    :param m: number of edges type
    :param p: probability for edge creation
    :return: graph G
    '''

    G = nx.DiGraph(nx.empty_graph(n))
    # Nodes in graph are from 0,n-1 (start with v as the first node index).
    v = 0
    w = -1
    lp = math.log(1.0 - p)
    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        if v == w:  # avoid self loops
            w = w + 1
        while v < n <= w:
            w = w - n
            v = v + 1
            if v == w:  # avoid self loops
                w = w + 1
        if v < n:
            G.add_edge(v, w, action=chr(random.randint(97,96+m)))  # add a random action in range

    return G

if __name__ == '__main__':
    G = random_labeled_digraph(5,3,0.7)
    # G = nx.DiGraph()
    # G.add_edge(1, 1, action='a')
    # G.add_edge(1, 2, action='a')
    # G.add_edge(2, 3, action='b')
    # G.add_edge(3, 1, action='c')
    # G.add_edge(3, 2, action='e')

    pos  = nx.spring_layout(G)
    nx.draw(G, pos)
    edge_labels = nx.get_edge_attributes(G, "action")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    plt.show()
