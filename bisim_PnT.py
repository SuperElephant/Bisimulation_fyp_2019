# this file is base on:
# [1] Paige, R., & Tarjan, R. E. (1987). Three Partition Refinement Algorithms. SIAM Journal on
#     Computing, 16(6), 973-989. http://doi.org/10.1137/0216062
# [2] A. Hoffman, "ArielCHoffman/BisimulationAlgorithms,"Github, 2015.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# partition =[set('', ...), set('', '''), ...]

class BisimPnT:
    def __init__(self, actions, graph_g, graph_h):
        self.actions = actions  # type: list
        self.full_graph_U = nx.union_all([graph_g, graph_h], rename=('G-', 'H-'))  # type: nx.DiGraph

    # return the preimage of the block_s
    def preimage(self, block_s, action=None):
        preimage = []
        for node in self.full_graph_U.nodes():
            successors = self.full_graph_U.successors(node)
            if action is None:
                if any((successor in block_s) for successor in successors):
                    preimage.append(node)
            elif any((self.full_graph_U[node][successor]["action"] == action and successor in block_s)
                     for successor in successors):
                preimage.append(node)
        return set(preimage)

    # need change
    def split_block(self, block_b, block_s, action=None):
        result = []
        prim_result = []
        if action is None:
            for action in self.actions:
                prim_result = self.split_block(block_b, block_s, action)
                if len(prim_result) == 2:
                    block_b = prim_result[1]  # second compound split result
                result.append(prim_result[0])
            if len(prim_result) == 2:
                result.append(prim_result[1])
            return result
        else:
            b1 = set(block_b).intersection(set(self.preimage(block_s, action=action)))
            b2 = set(block_b) - b1
            if b1:
                result.append(b1)
            if b2:
                result.append(b2)
            return result

    def compound_blocks(self, partition_Q, partition_X):
        block_set_C = []
        for block_x in partition_X:
            if self.is_compound_to(block_x, partition_Q) is not False:
                block_set_C.append(block_x)
        return block_set_C

    def is_compound_to(self, block_s, partition):
        contained_blocks = []
        i = 0
        for block in partition:
            if block.issubset(block_s):
                contained_blocks.append(block)
                i += 1
                if i == 2:
                    if (contained_blocks[0] < contained_blocks[1]):
                        return contained_blocks[0]
                    else:
                        return contained_blocks[1]
        return False

    def plotGraph(self, blocks, pos=None):
        numOfActions = len(self.actions)
        numOfBlocks = len(blocks)

        plt.figure(1)

        if not pos:
            pos = nx.spring_layout(self.full_graph_U)

        for i in xrange(numOfBlocks):
            nx.draw_networkx_nodes(self.full_graph_U, pos, nodelist=blocks[i], node_color=[i] * len(blocks[i]),
                                   cmap=cm.BrBG, vmin=-1, vmax=numOfBlocks)
        for i in xrange(numOfActions):
            acts = []

            for edge in self.full_graph_U.edges():
                if (self.full_graph_U.get_edge_data(*edge)["action"] == self.actions[i]):
                    acts.append(edge)

            nx.draw_networkx_edges(self.full_graph_U, pos, edgelist=acts, edge_color=[i] * len(acts),
                                   edge_cmap=cm.rainbow, edge_vmin=-1, edge_vmax=numOfActions,
                                   width=2.0, arrowsize=20)
        plt.show()

    def coarsest_partition(self, plot=True):

        # Initial
        block_u = set(self.full_graph_U.nodes())

        partition_Q = self.split_block(block_u, block_u)
        partition_X = [block_u]
        block_set_C = [block_u]
        pos = nx.kamada_kawai_layout(self.full_graph_U)

        while len(block_set_C) != 0:
            # Step 1:
            # select refining block_b
            block_set_C = self.compound_blocks(partition_Q, partition_X)
            block_s = block_set_C.pop()  # a compound block of partition_X
            block_b = self.is_compound_to(block_s, partition_Q)  # a block of Q that contained in s

            # Step 2:
            # update X:
            partition_X.remove(block_s)
            partition_X.append(block_s - block_b)
            partition_X.append(block_b)  # split block S in X

            # if the rest is still not simple, put it back to C
            if self.is_compound_to(block_s - block_b, partition_Q) is not False:
                block_set_C.append(block_s - block_b)

            for action in self.actions:

                # step 3:
                # compute preimage of B
                preimage_b = self.preimage(block_b, action)
                # compute preimage of S - B
                preimage_s_sub_b = self.preimage(block_s - block_b, action)

                # step 4:
                # refine Q wirh respect to B
                new_partition_Q = []
                for block_d in partition_Q:
                    block_d1 = block_d.intersection(preimage_b)
                    block_d2 = block_d - block_d1
                    block_d11 = block_d1.intersection(preimage_s_sub_b)
                    block_d12 = block_d - preimage_s_sub_b

                    if len(block_d2) and block_d2 not in new_partition_Q:
                        new_partition_Q.append(block_d2)
                    if len(block_d11) and block_d11 not in new_partition_Q:
                        new_partition_Q.append(block_d11)
                    if len(block_d12) and block_d12 not in new_partition_Q:
                        new_partition_Q.append(block_d12)
                    partition_Q = new_partition_Q

            if plot:
                self.plotGraph(partition_Q, pos)
        return partition_Q

    def is_bisimilar(self):

        partition = self.coarsest_partition()

        for block in partition:
            if not any('H' in node_name for node_name in block) \
                    or not any('G' in node_name for node_name in block):
                return False

        return True


if __name__ == '__main__':
    # the following is the bisimulation example:
    # # exanple 1:
    # G = nx.DiGraph()
    # G.add_edge(1, 2, action='a')
    # G.add_edge(2, 3, action='b')
    # G.add_edge(2, 4, action='b')
    #
    # H = nx.DiGraph()
    # H.add_edge(1, 2, action='a')
    # H.add_edge(1, 4, action='a')
    # H.add_edge(2, 3, action='b')
    # H.add_edge(4, 5, action='b')
    #
    # actions = ['a', 'b', 'c']
    # k = BisimPnT(actions, G, H)
    # print("Example 1: ")
    # print(k.is_bisimilar())

    # # example 2
    # G = nx.DiGraph()
    # G.add_edge(1, 2, action='a')
    # G.add_edge(2, 3, action='a')
    # G.add_edge(2, 1, action='b')
    # G.add_edge(3, 2, action='b')
    #
    # H = nx.DiGraph()
    # H.add_edge(1, 2, action='a')
    # H.add_edge(1, 3, action='a')
    # H.add_edge(2, 4, action='a')
    # H.add_edge(2, 1, action='b')
    # H.add_edge(3, 5, action='a')
    # H.add_edge(3, 1, action='b')
    # H.add_edge(4, 3, action='b')
    # H.add_edge(5, 2, action='b')
    #
    # actions = ['a', 'b']
    # k = BisimPnT(actions, G, H)
    # print("Example 2: ")
    # print(k.is_bisimilar())
    #
    #
    # example 3
    G = nx.DiGraph()
    G.add_edge(1, 3, action='a')
    G.add_edge(1, 2, action='a')
    G.add_edge(2, 3, action='b')
    G.add_edge(3, 1, action='c')
    G.add_edge(3, 2, action='b')

    H = nx.DiGraph()
    H.add_edge(1, 3, action='a')
    H.add_edge(1, 4, action='a')
    H.add_edge(2, 3, action='b')
    H.add_edge(3, 1, action='c')
    H.add_edge(3, 4, action='c')
    H.add_edge(4, 5, action='b')
    H.add_edge(5, 1, action='c')
    H.add_edge(5, 2, action='c')

    actions = ['a', 'b', 'c']
    k = BisimPnT(actions, G, H)
    print("Example 3: ")
    print(k.is_bisimilar())
