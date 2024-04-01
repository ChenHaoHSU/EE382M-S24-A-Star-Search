'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-07 16:42:12
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-08 16:04:28
'''
#######################################################################
# Implementation of A Star Search
# You need to implement initialize() and route_one_net()
# All codes should be inside A Star Search class
# Name: Chen-Hao Hsu
# UT EID: ch48458
#######################################################################

from typing import List, Tuple

import numpy as np

from .p2_routing_base import A_Star_Search_Base, GridAstarNode, PriorityQueue, AdvancedPriorityQueue

__all__ = ["A_Star_Search"]

class A_Star_Search(A_Star_Search_Base):
    def __init__(self) -> None:
        super().__init__()

    def _is_blockage(self, pos: Tuple[int, int]) -> bool:
        # Note the usage of blockage_map [y, x]
        return self.blockage_map[pos[1], pos[0]];

    def _is_grid(self, pos: Tuple[int, int]) -> bool:
        return (0 <= pos[0] < self.grid_size[0] and
                0 <= pos[1] < self.grid_size[1])

    def _get_neighbor_pos_list(
        self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        (x, y) = pos

        # Follow the specific order for neighbor exploration
        candidates = [
            (x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)
        ]

        # Collect valid neighbors
        neighbor_pos = []
        for xx, yy in candidates:
            if self._is_grid((xx, yy)) and not self._is_blockage((xx, yy)):
                neighbor_pos.append((xx, yy))

        return neighbor_pos

    def _get_node(self, pos: Tuple[int, int]) -> GridAstarNode:
        if pos not in self._nodes:
            # No need to set neighbors here. Neighbors are built on the fly
            self._nodes[pos] = GridAstarNode(
                pos=pos, cost_g=None, cost_f=None, bend_count=None,
                visited=False, parent=None, neighbors=[]
            )
        return self._nodes[pos]

    def _dist_to_target(self, pos: Tuple[int, int]) -> int:
        return self._find_manhattan_dist_to_target(pos, self._target_pos)

    def _relax(self, node: GridAstarNode, neighbor_node: GridAstarNode) -> None:
        """node: from-node
           neighbor_node: to-node (to be relaxed)
        """
        # Distance between node and neighbor_node
        dist = self._find_manhattan_dist_to_target(node.pos, neighbor_node.pos)

        # Three conditions for relaxation:
        #   1. neighbor_node is a new node (i.e. it has invalid cost_g)
        #   2. node.cost_g + dist < neighbor_node.cost_g
        #   3. node.cost_g + dist == neighbor_node.cost_g
        #      AND
        #      node.bend_count + has_bend(node, neighbor_node)
        #           < neighbor_node.bend_count
        if (neighbor_node.cost_g is None or
                node.cost_g + dist < neighbor_node.cost_g or
                    (node.cost_g + dist == neighbor_node.cost_g and
                        node.bend_count + self._has_bend(node, neighbor_node) <
                            neighbor_node.bend_count)):
            # Cost g: actual cost
            neighbor_node.cost_g = node.cost_g + dist
            # Cost h: use the manhattan distance to the target as the heuristic
            #         cost
            # Cost f: cost_f = cost_g + cost_h
            neighbor_node.cost_f = (neighbor_node.cost_g +
                                    self._dist_to_target(neighbor_node.pos))
            # Bend count
            neighbor_node.bend_count = (node.bend_count +
                                        self._has_bend(node, neighbor_node))
            # Set node as neighbor_node's parent
            neighbor_node.parent = node

            # Update PQ
            if self._pq.exist(neighbor_node):
                self._pq.update()
            else:
                self._pq.put(neighbor_node)

    def initialize(self):
        """Initialize necessary data structures before starting solving the
           problem
        """
        # TODO initialize any auxiliary data structure you need
        self._nodes = dict()  # Nodes (GridAstarNode) are created on the fly
        self._source_pos = (self.pin_pos_x[0], self.pin_pos_y[0])
        self._target_pos = (self.pin_pos_x[1], self.pin_pos_y[1])
        self._pq = AdvancedPriorityQueue()

    def route_one_net(
            self
    ) -> Tuple[List[Tuple[Tuple[int], Tuple[int]]], int, List[int], List[int]]:
        """route one multi-pin net using the A star search algorithm

        Return:
            path (List[Tuple[Tuple[int], Tuple[int]]]): the vector-wise routing
                path described by a list of (src, dst) position pairs
            wl (int): total wirelength of the routing path
            wl_list (List[int]): a list of wirelength of each routing path
            n_visited_list (List[int]): the number of visited nodes in the grid
            in each iteration
        """
        # TODO implement your A star search algorithm for one multi-pin net.
        # To make this method clean, you can extract subroutines as methods of
        # this class
        # But do not override methods in the parent class
        # Please strictly follow the return type requirement.

        # A path is found or not
        path_found = False
        # Number of visited nodes
        n_visited = 0

        # Initialize source
        source_node = self._get_node(self._source_pos)
        source_node.cost_g = 0
        source_node.cost_f = self._dist_to_target(self._source_pos)
        source_node.bend_count = 0
        source_node.visited = False
        source_node.parent = None
        # Insert source node to PQ
        self._pq.put(source_node)

        # Start searching
        while not self._pq.empty():
            # Extract min
            node = self._pq.get()
            node.visited = True
            n_visited += 1

            # Path found
            if node.pos == self._target_pos:
                path_found = True
                break

            # Relex neighbors
            for neighbor_pos in self._get_neighbor_pos_list(node.pos):
                neighbor_node = self._get_node(neighbor_pos)
                if neighbor_node.visited:
                    continue
                self._relax(node, neighbor_node)

        # Data to be returned
        path_list, wl = [], 0

        # Backtrack
        if path_found:
            # Backtrack from the target and merge paths
            path_list = self._merge_path(
                self._backtrack(self._nodes[self._target_pos]))

            # Calculate wirelength from path_list
            wl = sum([self._find_manhattan_dist_to_target(path[0], path[1])
                      for path in path_list])
        else:
            print('[Error] Cannot not find a path from source', self._source_pos,
                  'to target', self._target_pos)

        # Message
        print(f'[Info] Source: {self._source_pos}; Target: {self._target_pos}')
        print(f'[Info]   Wirelength: {wl}; #Visited: {n_visited}')

        return path_list, wl, [wl], [n_visited]

