import pydot
import networkx as nx
import constants as const
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from networkx.drawing.nx_pydot import write_dot
from graphviz import render, view


def plotGraph(graph, file_name='test.dot'):
    write_dot(graph, file_name)
    render('dot', 'png', file_name)
    view(file_name + '.png')


def plotGraphWithPattition(graph, blocks=None, file_name='test.dot'):
    if not blocks:
        plotGraph(graph, file_name)
        return

    colored_graph = nx.MultiDiGraph()

    for i in xrange(len(blocks)):
        for node in blocks[i]:
            colored_graph.add_node(node, color=i + 1, style="filled", colorscheme=const.NODE_COLOR_SCHEME)

    for edge in graph.edges(data=True):
        colored_graph.add_edge(edge[0], edge[1], label=edge[2]["label"], color=ord(edge[2]["label"]) - 96,
                               colorscheme=const.EDGE_COLOR_SCHEME)
    plotGraph(colored_graph, file_name)

# def plotGraphWithPattition(graph, blocks=None, labels=list(), pos=None):
#
#     numOfActions = len(labels)
#     numOfBlocks = len(blocks)
#
#     plt.figure(1)
#
#     if not pos:
#         pos = nx.spring_layout(graph)
#
#     for i in xrange(numOfBlocks):
#         nx.draw_networkx_nodes(graph, pos, nodelist=blocks[i], node_color=[i] * len(blocks[i]),
#                                cmap=cm.BrBG, vmin=-1, vmax=numOfBlocks)
#     for i in xrange(numOfActions):
#         acts = []
#
#         for edge in graph.edges():
#             if (graph.get_edge_data(*edge)["label"] == labels[i]):
#                 acts.append(edge)
#
#         nx.draw_networkx_edges(graph, pos, edgelist=acts, edge_color=[i] * len(acts),
#                                edge_cmap=cm.rainbow, edge_vmin=-1, edge_vmax=numOfActions,
#                                width=2.0, arrowsize=20)
#     plt.show()
