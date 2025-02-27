#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
import os
import sys

# save plots here
OUTPUT_DIR = '3d_plots'

# create output directory if it doesn't exist
def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"Created directory: {OUTPUT_DIR}")
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
            sys.exit(1)
    else:
        print(f"Directory already exists: {OUTPUT_DIR}")

if __name__ == '__main__':
        
    setup_directories()

    # Get the map for the scenario
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Q3d:
    # Configure the process model using different probabilities
    # list of p values
    p_values = [1, 0.9, 0.6, 0.3]

    for p in p_values:
        airport_environment.set_nominal_direction_probability(p)

        # Note that you can create multiple instances of the same object, with different
        # settings, and run them in the same programme. Therefore, you do not need to
        # create lots of separate scripts to run the code.

        # Create the policy iterator
        policy_solver = PolicyIterator(airport_environment)

        # Set up initial state
        policy_solver.initialize()
            
        # Bind the drawer with the solver
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        
        value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_function_drawer)
            
        # Compute the solution
        v, pi = policy_solver.solve_policy()

        policy_filename = os.path.join(OUTPUT_DIR, f"policy_iteration_results_p_{p}.pdf")
        policy_drawer.save_screenshot(policy_filename)
        value_function_filename = os.path.join(OUTPUT_DIR, f"value_function_iteration_results_p_{p}.pdf")
        value_function_drawer.save_screenshot(value_function_filename)
        
        # Wait for a key press
        #value_function_drawer.wait_for_key_press()
