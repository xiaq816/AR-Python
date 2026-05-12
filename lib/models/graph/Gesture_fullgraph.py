import sys
import numpy as np

sys.path.extend(['../'])
from . import tools

# 将左右手视为整图
num_node = 42
self_link = [(i, i) for i in range(num_node)]
ind_start = [5, 20, 6, 7, 20, 8, 9, 10, 20, 11, 12, 13, 20, 14, 15, 16, 20, 17, 18, 19]
ind_end =   [20, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]
r_ind_start = [x + 21 for x in ind_start]
r_ind_end = [x + 21 for x in ind_end]
ind_start = ind_start + r_ind_start
ind_end = ind_end + r_ind_end


inward_ori_index = []
for i in range(len(ind_start)):
    start = ind_start[i]
    end = ind_end[i]
    inward_ori_index.append((start, end))
# inward_ori_index = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (6,7), (0,8), (8,9), (9,10),
#                     (10,11), (0,12), (12,13), (13,14), (14,15), (0,16), (16,17), (17,18), (18,19)]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
