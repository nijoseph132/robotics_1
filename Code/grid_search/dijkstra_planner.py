'''
Created on 2 Jan 2022

@author: ucacsjj
'''

from collections import deque
from math import sqrt, dist
from queue import PriorityQueue

from .occupancy_grid import OccupancyGrid
from .planner_base import PlannerBase
from .search_grid import SearchGridCell

class DijkstraPlanner(PlannerBase):

    # This implements Dijkstra. The priority queue is the path length
    # to the current position.
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()  # type: ignore

    # Q1d:
    # Modify this class to finish implementing Dijkstra

    # add cell to queue, via correct priority
    def push_cell_onto_queue(self, cell: SearchGridCell):
        parent_cell = cell.parent 
        cost_to_come = self.compute_l_stage_additive_cost(parent_cell, cell)

        if parent_cell:
            cell.path_cost = parent_cell.path_cost + cost_to_come
        else:
            cell.path_cost = cost_to_come
        
        self.priority_queue.put((cell.path_cost, cell))
        

    # Check the queue size is zero
    def is_queue_empty(self) -> bool:
        return self.priority_queue.empty()

    # Simply pull from the front of the list
    def pop_cell_from_queue(self) -> SearchGridCell:
        cell = self.priority_queue.get(0)
        return cell[1]

    # update priority
    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        new_cost = parent_cell.path_cost + self.compute_l_stage_additive_cost(parent_cell, cell)
        if cell.path_cost > new_cost:
            cell.path_cost = new_cost
            self.priority_queue.put((cell.path_cost, cell))

