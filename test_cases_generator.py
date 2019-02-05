# this file is use to generate the test case ([flag, nx.DiGraph, nx.DiGraph])
# dependent on bisim_PnT.py

import math
import random
import networkx as nx
import bisim_PnT as bi
import numpy as np
import networkx.algorithms.isomorphism as iso


def test_cases_generator(min_node_number=5, edge_types=3, probability=0.1):
    n = min_node_number
    G = random_labeled_digraph(n, edge_types, probability)
    partition = bi.BisimPnT([chr(97 + t) for t in range(edge_types)], G).coarsest_partition()
    print(probability)
    return [G, H]

    # TODO: generate test case base on a random digraph


def generate_all(node_number=5, edge_number=3):
    graphs = []
    for graph_code in xrange(pow(2, node_number * node_number * edge_number)):
        edge_tuples = get_edges_from(graph_code, node_number, edge_number)
        g = nx.MultiDiGraph(edge_tuples)
        if g.order() == node_number and nx.is_weakly_connected(g):
            graphs.append(g)
            print(g.edges())
            print()
    return graphs


def classify_graphs(graphs, labels):
    all_min = []
    classified = []
    for graph in graphs:
        exist = False
        t = bi.BisimPnT(labels, graph)
        min_graph = t.get_min_graph()[0]
        for i in xrange(len(all_min)):
            if graph_equal(all_min[i], min_graph):
                # vi.plot_graph(all_min[i],'inlist')
                # vi.plot_graph(min_graph, 'newgene')
                classified[i].append(graph)
                exist = True
                break
        if not exist:
            all_min.append(min_graph)
            classified.append([graph])
    return classified


def graph_equal(graph_a, graph_b):
    em = iso.categorical_multiedge_match('label', '#')
    return nx.is_isomorphic(graph_a, graph_b, edge_match=em)


def get_edges_from(graph_code, node_number, edge_types):
    graph_string = bin(graph_code)[2:]
    tuple_length = node_number * edge_types
    diff = node_number * tuple_length - len(graph_string)
    if diff > 0:
        graph_string = ('0' * diff + graph_string)
    graph_string = graph_string[::-1]
    tuples = []
    for i in xrange(len(graph_string)):
        if graph_string[i] == '1':
            tuples.append((i / tuple_length + 1,
                           i % tuple_length / edge_types + 1,
                           dict(label=chr(97 + i % tuple_length % edge_types))))
    return tuples


if __name__ == '__main__':
    import visualization as vi

    a = generate_all(3, 1)
    b = classify_graphs(a, ['a'])
    print(b)

    for i in range(0, 392, 30):
        vi.plot_graph(a[i], "test_case" + str(i))
    print len(a)

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
    # G.add_nodes_from(range(n))
    for v in range(n):
        for w in range(n):
            for e in range(m):
                if random.random() < p:
                    # if any(label == chr(97+e) for label in G.get_edge_data(v,w)):
                    G.add_edge(v, w, label=chr(97+e))

    return G
#
# if __name__ == '__main__':
#     import pydot
#     from graphviz import render,view
#     from networkx.drawing.nx_pydot import write_dot
#
#     # G = random_labeled_digraph(5,3,0.1)
#     G = nx.MultiDiGraph()
#     G.add_node(1,style="filled", colorscheme="dark28", color=1)
#     G.add_node(2,style="filled", colorscheme="dark28", color=2)
#
#     G.add_edge(1,1, label='a', colorscheme="paired12", color=2)
#     G.add_edge(1,1, label='a', colorscheme="paired12", color=2)
#     G.add_edge(1,1, label='b', colorscheme="paired12", color=1)
#     G.add_edge(1,2, label='a', colorscheme="paired12", color=2)
#     G.add_edge(1,2, label='b', colorscheme="paired12", color=1)
#
#     write_dot(G, "test.dot")
#     render('dot', 'png', 'test.dot')
#     view('test.dot.png')
#
#     # test_cases_generator()
#
#
#
#     # G = nx.DiGraph()
#     # G.add_edge(1, 1, action='a')
#     # G.add_edge(1, 2, action='a')
#     # G.add_edge(2, 3, action='b')
#     # G.add_edge(3, 1, action='c')
#     # G.add_edge(3, 2, action='e')
#
#     # pos  = nx.spring_layout(G)
#     # nx.draw(G, pos)
#     # edge_labels = nx.get_edge_attributes(G, "action")
#     # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
#     # plt.show()
