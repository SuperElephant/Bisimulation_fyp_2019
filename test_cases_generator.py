# this file is use to generate the test case ([flag, nx.DiGraph, nx.DiGraph])
# dependent on bisim_PnT.py

import math
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def test_cases_generator(min_node_number=5, edge_types=3, probability=0.1):
    n = min_node_number
    G1 = random_labeled_digraph(n, edge_types, probability)

    # TODO: generate test case base on a random digraph

def random_labeled_digraph(n, m, p):
    '''
    return a random labeled directed graph
    modify from networkx.generators.random_graphs
    https://github.com/networkx/networkx/blob/master/networkx/generators/random_graphs.py

    :param n: number of node
    :param m: number of edges type (a, b, c, ...)
    :param p: probability for edge creation
    :return: graph G
    '''
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    for v in range(n):
        for w in range(n):
            for e in range(m):
                if random.random() < p:
                    G.add_edge(v, w, label=chr(97+e))

    return G

if __name__ == '__main__':
    import pydot
    from graphviz import render
    from networkx.drawing.nx_pydot import write_dot
    G = random_labeled_digraph(5,3,0.1)
    write_dot(G, "test.dot")
    render('dot', 'png', 'test.dot')



    # G = nx.DiGraph()
    # G.add_edge(1, 1, action='a')
    # G.add_edge(1, 2, action='a')
    # G.add_edge(2, 3, action='b')
    # G.add_edge(3, 1, action='c')
    # G.add_edge(3, 2, action='e')

    # pos  = nx.spring_layout(G)
    # nx.draw(G, pos)
    # edge_labels = nx.get_edge_attributes(G, "action")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    # plt.show()
