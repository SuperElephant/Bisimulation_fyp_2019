# this file is base on:
# [1] Paige, R., & Tarjan, R. E. (1987). Three Partition Refinement Algorithms. SIAM Journal on
#     Computing, 16(6), 973-989. http://doi.org/10.1137/0216062
# [2] A. Hoffman, "ArielCHoffman/BisimulationAlgorithms,"Github, 2015.s

import networkx as nx
import numpy as np
import visualization as vi


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm

# partition =[set('', ...), set('', '''), ...]


class BisimPnT:
    def __init__(self, labels, graph_g, graph_h=None):
        self.labels = labels  # type: list
        if graph_h is None:
            self.full_graph_U = graph_g
        else:
            self.full_graph_U = nx.union_all([graph_g, graph_h], rename=('G-', 'H-'))  # type: nx.MultiDiGraph
        self.co_partition = []

    # return the preimage of the block_s
    def preimage(self, block_s, label=None):
        preimage = []
        for node in self.full_graph_U.nodes():
            successors = self.full_graph_U.successors(node)
            if label is None:
                if any((successor in block_s) for successor in successors):
                    preimage.append(node)
            elif type(self.full_graph_U) == nx.DiGraph \
                    and any((self.full_graph_U[node][successor]['label'] == label and successor in block_s)
                            for successor in successors):
                preimage.append(node)
            elif type(self.full_graph_U) == nx.MultiDiGraph \
                    and any((any(edge['label'] == label for edge in self.full_graph_U[node][successor].itervalues())
                             and successor in block_s)
                            for successor in successors):
                preimage.append(node)
        return set(preimage)

    # need change
    def split_block(self, block_b, block_s, label=None):
        result = []
        prim_result = []
        if label is None:
            if not (block_b and block_s):
                return []
            for label in self.labels:
                prim_result = self.split_block(block_b, block_s, label)
                if len(prim_result) == 2:
                    block_b = prim_result[1]  # second compound split result
                    result.append(prim_result[0])
            result.append(prim_result[-1])
            return result
        else:
            b1 = set(block_b).intersection(set(self.preimage(block_s, label=label)))
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
            # i=0
            # for block in partition_Q:
            #     if block.issubset(block_x):
            #         i += 1
            #     if i==2:
            #         block_set_C.append(block_x)
            #         break
        return block_set_C

    def is_compound_to(self, block_s, partition):
        contained_blocks = []
        i = 0
        for block in partition:
            if block.issubset(block_s):
                contained_blocks.append(block)
                i += 1
                if i == 2:
                    if contained_blocks[0] < contained_blocks[1]:
                        return contained_blocks[0]
                    else:
                        return contained_blocks[1]
        return False

    def coarsest_partition(self, plot=False):

        # Initial
        block_u = set(self.full_graph_U.nodes())

        partition_Q = self.split_block(block_u, block_u)
        partition_X = [block_u]
        block_set_C = [block_u]

        block_set_C = self.compound_blocks(partition_Q, partition_X)
        while len(block_set_C) != 0:
            # Step 1:
            # select refining block_b
            block_s = block_set_C.pop()  # a compound block of partition_X
            block_b = self.is_compound_to(block_s, partition_Q)  # a block of Q that contained in s

            # Step 2:
            # update X:
            partition_X.remove(block_s)
            partition_X.append(block_s - block_b)
            partition_X.append(block_b)  # split block S in X

            # if the rest is still not simple, put it back to C
            if self.is_compound_to(block_s - block_b, partition_Q) is not False:
                block_set_C.append(block_s)

            for label in self.labels:

                # step 3:
                # compute preimage of B
                preimage_b = self.preimage(block_b, label)
                # compute preimage of S - B
                preimage_s_sub_b = self.preimage(block_s - block_b, label)

                # step 4:
                # refine Q wirh respect to B
                new_partition_Q = []
                for block_d in partition_Q:
                    block_d1 = block_d.intersection(preimage_b)
                    block_d2 = block_d - block_d1
                    block_d11 = block_d1.intersection(preimage_s_sub_b)
                    block_d12 = block_d1 - preimage_s_sub_b

                    if len(block_d2) and block_d2 not in new_partition_Q:
                        new_partition_Q.append(block_d2)
                    if len(block_d11) and block_d11 not in new_partition_Q:
                        new_partition_Q.append(block_d11)
                    if len(block_d12) and block_d12 not in new_partition_Q:
                        new_partition_Q.append(block_d12)
                    partition_Q = new_partition_Q

            block_set_C = self.compound_blocks(partition_Q, partition_X)
            if plot:
                vi.plot_graph_with_partition(self.full_graph_U, partition_Q)

        self.co_partition = partition_Q
        return partition_Q

    def get_min_graph(self):
        if not self.co_partition:
            self.coarsest_partition()
        partition = [i.copy() for i in self.co_partition]
        if len(partition) == self.full_graph_U.order():
            return nx.MultiDiGraph(self.full_graph_U)
        else:
            G = nx.MultiDiGraph()

        for n in range(len(partition)):

            # for part in partition:
            possible_type = []
            while partition[n]:
                node = partition[n].pop()
                # new_partition.append({n + 1})
                for successor in self.full_graph_U.successors(node):
                    for i in range(len(partition)):
                        if successor in self.co_partition[i] and i not in possible_type:
                            possible_type.append(i)
                            links = {label.values()[0] for label in
                                     self.full_graph_U.get_edge_data(node, successor).values()}
                            for label in links:
                                G.add_edge(n, i, label=label)
        # return [G, new_partition]
        return G

    def is_bisimilar(self):

        partition = self.coarsest_partition()

        for block in partition:
            if not any('H' in node_name for node_name in block) \
                    or not any('G' in node_name for node_name in block):
                return False

        return True


if __name__ == '__main__':
    # the following is the bisimulation example:
    # =============== example 1 ===============
    # G = nx.MultiDiGraph()
    # G.add_edge(5, 2, label='a')
    # G.add_edge(2, 3, label='b')
    # G.add_edge(2, 4, label='b')
    # # G.add_edge(2, 4, label='c')
    #
    # H = nx.MultiDiGraph()
    # H.add_edge(1, 2, label='a')
    # H.add_edge(1, 4, label='a')
    # H.add_edge(2, 3, label='b')
    # H.add_edge(4, 5, label='b')
    # # H.add_edge(4, 5, label='c')
    #
    # labels = ['a', 'b']
    # # labels = ['a', 'b', 'c']
    #
    # k = BisimPnT(labels, G, H)
    # par = k.coarsest_partition()
    # vi.plot_graph_with_partition(k.full_graph_U,par,'example_1')
    # vi.plot_graph(BisimPnT(labels,H).get_min_graph(), 'mini_g')
    # print("Example 1: ")
    # print(par)
    # print(k.is_bisimilar())

    # # =============== example 2 ===============
    # G = nx.MultiDiGraph()
    # G.add_edge(1, 2, label='a')
    # G.add_edge(2, 3, label='a')
    # G.add_edge(2, 1, label='b')
    # G.add_edge(3, 2, label='b')
    #
    # H = nx.MultiDiGraph()
    # H.add_edge(1, 2, label='a')
    # H.add_edge(1, 3, label='a')
    # H.add_edge(2, 4, label='a')
    # H.add_edge(2, 1, label='b')
    # H.add_edge(3, 5, label='a')
    # H.add_edge(3, 1, label='b')
    # H.add_edge(4, 3, label='b')
    # H.add_edge(5, 2, label='b')
    #
    # labels = ['a', 'b']
    # k = BisimPnT(labels, G, H)
    # vi.plot_graph_with_partition(k.full_graph_U,k.coarsest_partition(), 'example_2')
    # vi.plot_graph(BisimPnT(labels, H).get_min_graph(), 'mini_g2')
    # print("Example 2: ")
    # print(k.is_bisimilar())


    # # =============== example 3 ===============
    # G = nx.MultiDiGraph()
    # G.add_edge(1, 3, label='a')
    # G.add_edge(1, 2, label='a')
    # G.add_edge(2, 3, label='b')
    # G.add_edge(3, 1, label='c')
    # G.add_edge(3, 2, label='c')
    #
    # H = nx.MultiDiGraph()
    # H.add_edge(1, 3, label='a')
    # H.add_edge(1, 4, label='a')
    # H.add_edge(2, 3, label='b')
    # H.add_edge(3, 1, label='c')
    # H.add_edge(3, 4, label='c')
    # H.add_edge(4, 5, label='b')
    # H.add_edge(5, 1, label='c')
    # H.add_edge(5, 2, label='c')
    #
    # labels = ['a', 'b', 'c']
    # k = BisimPnT(labels, G, H)
    # vi.plot_graph_with_partition(k.full_graph_U,k.coarsest_partition(), 'example_3')
    # vi.plot_graph(BisimPnT(labels, H).get_min_graph(), 'mini_g3')
    # print("Example 3: ")
    # print(k.is_bisimilar())

    # # =============== example 4 ===============
    # H = nx.MultiDiGraph()
    # H.add_edge(1, 3, label='a')
    # H.add_edge(1, 4, label='a')
    # H.add_edge(2, 3, label='b')
    # H.add_edge(3, 1, label='c')
    # H.add_edge(3, 4, label='c')
    # H.add_edge(4, 5, label='b')
    # H.add_edge(5, 1, label='c')
    # H.add_edge(5, 2, label='c')
    #
    # labels = ['a', 'b', 'c']
    # k = BisimPnT(labels, H)
    # par = k.coarsest_partition()
    # print(par)
    # vi.plot_graph_with_partition(k.full_graph_U,par, 'raw')
    # t = k.get_min_graph()
    # vi.plot_graph(t,'min_g')

    # # =============== example 5 ===============
    # a = nx.MultiDiGraph()
    # a.add_edge(0, 0, label='a')
    # a.add_edge(0, 2, label='b')
    # a.add_edge(1, 0, label='b')
    # a.add_edge(3, 0, label='a')
    # a.add_edge(3, 3, label='c')
    # a.add_edge(4, 4, label='a')
    # a.add_edge(4, 1, label='c')
    #
    # k = BisimPnT(['a', 'b', 'c'], a)
    # vi.plot_graph(a, 'raw')
    # par = k.coarsest_partition()
    # print(par)
    # vi.plot_graph_with_partition(a, par, 'with_p')
    # min_g = k.get_min_graph()
    # vi.plot_graph(min_g, 'min')
    pass
