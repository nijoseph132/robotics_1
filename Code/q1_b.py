#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario
from p1.high_level_actions import HighLevelActionType
from p1.high_level_environment import HighLevelEnvironment, PlannerType

if __name__ == '__main__':
    
    # Create the scenario
    airport_map, drawer_height = full_scenario()

    print(airport_map)
    
    # Draw what the map looks like. This is optional and you
    # can comment it out
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()    
    airport_map_drawer.wait_for_key_press()
    
    # Create the gym environment
    # Q1b:
    # Evaluate breadth and depth first algorithms.
    # Check the implementation of the environment
    # to see how the planner type is used.
    airport_environment = HighLevelEnvironment(airport_map, PlannerType.BREADTH_FIRST)
    #airport_environment = HighLevelEnvironment(airport_map, PlannerType.DEPTH_FIRST)

    # Set to this to True to generate the search grid and
    # show graphics. If you set this to false, the
    # screenshot will not work.
    airport_environment.show_graphics(True)

    # Set to this to True to show step-by-step graphics.
    # This is potentially useful for debugging, but can be very slow.    
    airport_environment.show_verbose_graphics(False)

    # First specify the start location of the robot
    action = (HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
    observation, reward, done, info = airport_environment.step(action)
    
    if reward is -float('inf'):
        print('Unable to teleport to (1, 1)')
        
    # Get all the rubbish bins and toilets; these are places which need cleaning
    all_rubbish_bins = airport_map.all_rubbish_bins()
        
    # Q1b:
    # Modify this code to collect the data needed to assess the different algorithms
    # This code also shows how to dump the search grid. For your submitted coursework,
    # please DO NOT include all the graphs - just the important ones
    
    # Now go through them and plan a path sequentially

    bin_number = 1
    total_path_cost = 0
    total_cells_visited = 0
    
    for rubbish_bin in all_rubbish_bins:
        action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
        # observation - Goal coordinates
        # reward – path cost, using distance with each cell edge as 1 unit. (i.e. diagonals cost root 2)
        # done - False
        # info - Path object, (info.number_of_cells_visited)
        observation, reward, done, info = airport_environment.step(action)
        screen_shot_name = f'bin_{bin_number:02}.pdf'
        airport_environment.search_grid_drawer().save_screenshot(screen_shot_name)
        bin_number += 1

        total_path_cost += reward
        total_cells_visited += info.number_of_cells_visited

        try:
            input("Press enter in the command window to continue.....")
        except SyntaxError:
            pass  

    print("Total Path Cost:", total_path_cost)
    print("Total Cells Visited:", total_cells_visited)
    
