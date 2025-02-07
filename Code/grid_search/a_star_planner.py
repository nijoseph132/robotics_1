'''
Created on 2 Jan 2022

@author: ucacsjj
'''

import math

from .dijkstra_planner import DijkstraPlanner
from .occupancy_grid import OccupancyGrid

class AStarPlanner(DijkstraPlanner):
    def __init__(self, occupancy_grid: OccupancyGrid):
        DijkstraPlanner.__init__(self, occupancy_grid)
    # Q2d:
    # Complete implementation of A*.
    def push_cell_onto_queue(self, cell):
        parent_cell = cell.parent 
        cost_to_come = self.compute_l_stage_additive_cost(parent_cell, cell) if parent_cell else 0
        cost_to_come += parent_cell.path_cost if parent_cell else 0
        cell.path_cost = cost_to_come
        cost_to_go = self.compute_heuristic(cell)
        total_path_cost = cost_to_come + cost_to_go
        self.priority_queue.put((total_path_cost, cell))

    def compute_heuristic(self, cell):
        x, y = cell.coords()
        x_g, y_g = self.goal.coords()
        return math.sqrt((x - x_g)**2 + (y - y_g)**2)
    
    def resolve_duplicate(self, cell, parent_cell):
        new_cost = parent_cell.path_cost + self.compute_l_stage_additive_cost(parent_cell, cell)
        cost_to_go = self.compute_heuristic(cell)
        if new_cost < cell.path_cost:
            cell.path_cost = new_cost
            cell.parent = parent_cell
            total_path_cost = cell.path_cost + cost_to_go
            self.priority_queue.put((total_path_cost, cell))