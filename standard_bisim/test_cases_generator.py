# this file is use to generate the test case ([flag, nx.DiGraph, nx.DiGraph])
# dependent on bisim_PnT.py

import math
import random
import csv
import networkx as nx
import bisim_PnT as bi
import numpy as np
import networkx.algorithms.isomorphism as iso
import time
from contextlib import contextmanager
import logging
import visualization as vi
import os
import argparse


def test_cases_generator(c_type="random", number=100, file_name="test_cases",
                         min_node_number=5, edge_type_number=3, probability=0.5, p_rate=0.5):
    labels = [chr(97 + i) for i in xrange(edge_type_number)]
    if not os.access('./data/', os.R_OK):
        os.mkdir('./data/')
    with open('./data/' + file_name + '.csv', 'w') as csvfile:
        print('write in path: ' + './data/' + file_name + '.csv')
        if c_type == 'random':
            fieldnames = ['g1', 'g2', 'bis']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for _ in xrange(number):
                if _ % (number/10) == 0: print("generating # %d" % _)
                g1 = ''
                while not g1 or not nx.is_weakly_connected(g1):
                    g1 = random_labeled_digraph(min_node_number, edge_type_number, random.random() * probability)
                g1_bin = get_bin_array_from_graph(g1, min_node_number, edge_type_number)
                bis = 1
                if random.random() > p_rate:
                    g2 = generate_random_similar(g1, edge_type_number)
                else:
                    g2 = random_labeled_digraph(min_node_number, edge_type_number, random.random() * probability)
                    k = bi.BisimPnT(labels, g1, g2)
                    bis = int(k.is_bisimilar())
                g2_bin = get_bin_array_from_graph(g2, min_node_number, edge_type_number)
                writer.writerow({fieldnames[0]: g1_bin, fieldnames[1]: g2_bin, fieldnames[2]: bis})

        elif c_type == 'all':
            fieldnames = ['g', 'type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            g = generate_all(min_node_number, edge_type_number)
            c = classify_graphs(labels)
            case_num = -1
            try:
                while True:
                    case_num = case_num + 1
                    if case_num % 100 == 0: print("generating # %d" % case_num)
                    graph = g.next()
                    graph_bin = get_bin_array_from_graph(graph, min_node_number, edge_type_number)
                    c.next()
                    c_type = c.send(graph)
                    writer.writerow({fieldnames[0]: graph_bin, fieldnames[1]: str(c_type)})
            except StopIteration:
                print("generating # %d" % case_num)
                pass

            # g = generate_all(2, 1)
            # c = classify_graphs(["a"])
            # try:
            #     while True:
            #         gr = g.next()
            #         c.next()
            #         t = c.send(gr)
            #         # print get_bin_array_from_graph(gr, 2, 1) + " " + str(t)
            # except StopIteration:
            #     pass


@contextmanager
def log_time(prefix=""):
    start = time.time()
    # logging.getLogger().setLevel(logging.DEBUG)
    try:
        yield
    finally:
        end = time.time()
        elapsed_seconds = float("%.2f" % (end - start))
        logging.debug('%s: elapsed seconds: %s', prefix, elapsed_seconds)


def generate_random_similar(graph, edge_type_number):
    with log_time("random_similar"):
        k = bi.BisimPnT([chr(97 + t) for t in range(edge_type_number)], graph)
        min_graph = k.get_min_graph()
        min_node_number = min_graph.order()
        partition = [{i} for i in range(min_node_number)]
        node_number = graph.order()
        if min_node_number == node_number:
            return random_relabel_nodes(graph)
        for i in xrange(min_node_number, node_number):
            partition[random.randint(0, min_node_number - 1)].add(i)
        result_graph = None
        # print min_node_number, node_number
        while not result_graph or result_graph.order() != graph.order():
            result_graph = nx.MultiDiGraph()
            for start_node_type in xrange(min_node_number):
                for end_node_type in min_graph.successors(start_node_type):  # for each origin type pair
                    for edge_label in [edge['label'] for edge in
                                       min_graph.get_edge_data(start_node_type, end_node_type).values()]:
                        # for each type edge
                        start_nodes = random.sample(partition[start_node_type],
                                                    random.randint(1, len(partition[start_node_type])))
                        for star_node in start_nodes:
                            end_nodes = random.sample(partition[end_node_type],
                                                      random.randint(1, len(partition[end_node_type])))
                            for end_node in end_nodes:
                                result_graph.add_edge(star_node, end_node, label=edge_label)

    return result_graph


def random_relabel_nodes(graph):
    n = graph.order()
    t = random.sample(range(n), n)
    result_graph = nx.relabel_nodes(graph, {i: t[i] for i in xrange(n)})
    return result_graph


def generate_all(node_number=5, edge_number=3):
    # graphs = []
    for graph_code in xrange(pow(2, node_number * node_number * edge_number)):
        g = get_graph_from(graph_code, node_number, edge_number)
        if g.order() == node_number and nx.is_weakly_connected(g):
            # graphs.append(g)
            # print(g.edges())
            yield g
    # return graphs


# def classify_graphs(graphs, labels):
# all_min = []
# classified = []
# for graph in graphs:
#     exist = False
#     t = bi.BisimPnT(labels, graph)
#     min_graph = t.get_min_graph()
#     for i in xrange(len(all_min)):
#         if graph_equal(all_min[i], min_graph):
#             # vi.plot_graph(all_min[i],'inlist')
#             # vi.plot_graph(min_graph, 'newgene')
#             classified[i].append(graph)
#             exist = True
#             break
#     if not exist:
#         all_min.append(min_graph)
#         classified.append([graph])
# return classified

def classify_graphs(labels):
    all_min = []
    classified = []
    while True:
        exist = False
        graph = yield
        t = bi.BisimPnT(labels, graph)
        min_graph = t.get_min_graph()
        for i in xrange(len(all_min)):
            if graph_equal(all_min[i], min_graph):
                # vi.plot_graph(all_min[i],'inlist')
                # vi.plot_graph(min_graph, 'newgene')
                yield i
                exist = True
                break
        if not exist:
            all_min.append(min_graph)
            yield len(all_min) - 1


def graph_equal(graph_a, graph_b):
    if graph_a.order() != graph_b.order() or len(graph_a.edges) != len(graph_b.edges):
        return False
    em = iso.categorical_multiedge_match('label', '#')
    return nx.is_isomorphic(graph_a, graph_b, edge_match=em)


def get_graph_from(graph_code, node_number, edge_types):
    graph_string = bin(graph_code)[2:]
    tuple_length = node_number * edge_types
    diff = node_number * tuple_length - len(graph_string)
    if diff > 0:
        graph_string = ('0' * diff + graph_string)
    elif diff < 0:
        graph_string = graph_string[0:node_number * tuple_length]

    bin_array = map(int, list(graph_string[::-1]))

    # tuples = []
    # for i in xrange(len(graph_string)):
    #     if graph_string[i] == '1':
    #         tuples.append((i / tuple_length,
    #                        i % tuple_length / edge_types,
    #                        dict(label=chr(97 + i % tuple_length % edge_types))))
    return get_graph_from_bin_array(bin_array, node_number, edge_types)


def get_graph_from_bin_array(bin_array, node_number, edge_types):
    tuple_length = node_number * edge_types
    tuples = []
    for i in xrange(len(bin_array)):
        if bin_array[i] == 1:
            tuples.append((i / tuple_length,
                           i % tuple_length / edge_types,
                           dict(label=chr(97 + i % tuple_length % edge_types))))
    return nx.MultiDiGraph(tuples)


def get_bin_array_from_graph(graph, node_number, edge_types):
    tuple_length = node_number * edge_types
    bin_array = [0] * tuple_length * node_number
    for edge in graph.edges(data=True):
        bin_array[edge[0] * tuple_length
                  + edge[1] * edge_types
                  + ord(edge[2]['label']) - 97] = 1
    # return ''.join(map(str,bin_array))
    # return bin_array
    return str(bin_array)[1:-1]


def random_labeled_digraph(node_number, edge_types_number, p):
    '''
    return a random labeled directed graph
    modify from networkx.generators.random_graphs
    https://github.com/networkx/networkx/blob/master/networkx/generators/random_graphs.py

    :param node_number: number of node
    :param edge_types_number: number of edges type (a, b, c, ...)
    :param p: probability for edge creation
    :return: graph G
    '''
    G = nx.MultiDiGraph()
    # G.add_nodes_from(range(n))
    for v in range(node_number):
        for w in range(node_number):
            for e in range(edge_types_number):
                if random.random() < p:
                    # if any(label == chr(97+e) for label in G.get_edge_data(v,w)):
                    G.add_edge(v, w, label=chr(97 + e))

    return G


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), os.path.pardir))

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, choices=('random', 'all'), default='random', dest='c_type',
                        help='Type of data set')
    parser.add_argument('-n', '--number', type=int, default=1000, dest='number',
                        help='The length of data set')
    parser.add_argument('-f', '--file_name', type=str, default='test_cases', dest='file_name',
                        help='Name of the output file')
    parser.add_argument('-v', '--node_number', type=int, default=3, dest='node_number',
                        help='Number of the nodes of the graph in the data set')
    parser.add_argument('-e', '--edge_type-number', type=int, default=2, dest='edge_type_number',
                        help='The total types of the edge in the graphs')
    parser.add_argument('-r', '--p_rate', type=float, default=0.5, dest='p_rate',
                        help='Rate of the positive cases over all cases')
    parser.add_argument('-p', '--probability', type=float, default=0.5, dest='probability',
                        help='The density of the random generate graphs')
    args = parser.parse_args()
    # print args

    print("=============== strat generating data ===============")
    test_cases_generator(c_type=args.c_type,
                         number=args.number,
                         min_node_number=args.node_number,
                         edge_type_number=args.edge_type_number,
                         file_name=args.file_name,
                         p_rate=args.p_rate,
                         probability=args.probability
                         )

    # type = "random", number = 100, file_name = "test_cases",
    # min_node_number=5, edge_type_number=3, probability=0.5

    import pydot
    from graphviz import render, view
    from networkx.drawing.nx_pydot import write_dot
    import visualization as vi

    # a = nx.MultiDiGraph()
    # a.add_edge(0, 1, label='a')
    # a.add_edge(0, 2, label='a')
    # a.add_edge(1, 3, label='b')
    # a.add_edge(2, 4, label='b')
    # vi.plot_graph(a, 'test')
    # b = generate_random_similar(a, 2)
    # vi.plot_graph(b, 'test1')
    #
    # print a

    # test_cases_generator(type="random", min_node_number=3, edge_type_number=2,file_name='random_pairs',number=10000)

    #
    # a = generate_all(3, 1)
    # b = classify_graphs(a, ['a'])
    # print(b)
    #
    # for i in range(0, 392, 30):
    #     vi.plot_graph(a[i], "test_case" + str(i))
    # print len(a)

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
