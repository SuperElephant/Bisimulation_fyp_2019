
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

class BisimSimple:
    def __init__(self, actions, graph_g, graph_h):
        self.actions = actions
        self.full_graph_U = nx.union_all([graph_g,graph_h], rename=('G-', 'H-'))  # type: nx.DiGraph

    # return preimage of a set/block respect to certain action
    def preimage(self, block, action):
        preimage = []
        for node in self.full_graph_U.nodes():
            successors = self.full_graph_U.successors(node)
            if any((self.full_graph_U[node][successor]["action"] == action and successor in block
                   for successor in successors)):
                preimage.append(node)
        return set(preimage)

    def refine_block(self, block, refiner):
        """

        :type refiner: set
        :type block: set
        """
        include = block.intersection(refiner)
        exclude = block - include
        if len(include)==0 or len(exclude)==0:
            return False
        elif len(include) > len(exclude): # smaller at end
            return [include, exclude]
        else:
            return [exclude, include]

    def get_coarsest_partition(self, plot=True):

        # initial
        check_set = [set(self.full_graph_U.nodes())]
        partition = [set(self.full_graph_U.nodes())]

        while len(check_set) != 0:

            split_block = check_set[-1]
            new_check_set = []
            refined_blocks = False

            for action in self.actions:
                preimage = self.preimage(split_block, action)
                for block in check_set:
                    refined_blocks = self.refine_block(block,preimage)
                    if refined_blocks:
                        new_check_set += refined_blocks
                    else:
                        new_check_set += block
                check_set = new_check_set




            if refined_blocks==False:




            