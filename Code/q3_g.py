#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

from common.scenarios import *
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import \
    ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
from generalized_policy_iteration.value_iterator import ValueIterator
import numpy as np
import argparse
import copy

"""""
1. Default mode (no command-line arguments): run on the default (non-randomized) map once.
2. Multi-run mode (using --num_runs > 1 and --randomize flag): run multiple trials (e.g. 10 runs)
   with randomized obstacle layouts to compute average iteration counts.

Usage:
    python q3_g.py
    python q3_g.py --num_runs 10 --randomize
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Policy Iteration and Value Iteration")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
    )
    return parser.parse_args()


def run_single(airport_map, drawer_height):
    print("Running on the default map (non-randomized) once.")

    # Deep copy the airport map so that each algorithm gets an identical copy.
    airport_map_pi = copy.deepcopy(airport_map)
    airport_map_vi = copy.deepcopy(airport_map)

    # Policy Iteration
    environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
    environment_pi.set_nominal_direction_probability(0.8)
    policy_solver = PolicyIterator(environment_pi)
    policy_solver.initialize()
    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    value_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_drawer)

    print("Running Policy Iteration...")
    v_pi, pi_pi = policy_solver.solve_policy()
    print("Policy Iteration Stats:")
    print(f"  Evaluation iterations: {policy_solver._policy_eval_iterations}")
    print(f"  Improvement iterations: {policy_solver._policy_improve_iterations}")

    policy_drawer.save_screenshot("policy_iteration_policy_default.pdf")
    value_drawer.save_screenshot("policy_iteration_value_default.pdf")

    # ---- Value Iteration ----
    environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
    environment_vi.set_nominal_direction_probability(0.8)
    value_solver = ValueIterator(environment_vi)
    value_solver.initialize()
    policy_drawer_vi = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
    value_solver.set_policy_drawer(policy_drawer_vi)
    value_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
    value_solver.set_value_function_drawer(value_drawer_vi)

    print("Running Value Iteration...")
    v_vi, pi_vi = value_solver.solve_policy()
    print("Value Iteration Stats:")
    print(f"  Value sweeps: {value_solver._value_iterations}")

    policy_drawer_vi.save_screenshot("value_iteration_policy_default.pdf")
    value_drawer_vi.save_screenshot("value_iteration_value_default.pdf")


def run_multiple(airport_map, drawer_height, num_runs, randomize_flag):
    pi_eval_iterations = []
    pi_improve_iterations = []
    vi_value_iterations = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        if randomize_flag:
            # Create new map
            tmp_map = copy.deepcopy(airport_map)
            # Create a temporary env instance to randomize obstacles on the map.
            temp_env = LowLevelEnvironment(tmp_map, randomize_obstacles_flag=True, obstacle_probability=0.01)
            # Get the randomized map from the environment.
            randomized_map = temp_env.map()
            # Deep copy the randomized map for each algorithm.
            airport_map_pi = copy.deepcopy(randomized_map)
            airport_map_vi = copy.deepcopy(randomized_map)
        else:
            airport_map_pi = copy.deepcopy(airport_map)
            airport_map_vi = copy.deepcopy(airport_map)

        # Policy Iteration
        environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
        environment_pi.set_nominal_direction_probability(0.8)
        policy_solver = PolicyIterator(environment_pi)
        policy_solver.initialize()
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        value_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_drawer)

        print("Running Policy Iteration...")
        v_pi, pi_pi = policy_solver.solve_policy()
        pi_eval_iterations.append(policy_solver._policy_eval_iterations)
        pi_improve_iterations.append(policy_solver._policy_improve_iterations)

        policy_drawer.save_screenshot(f"policy_iteration_policy_run{run + 1}.pdf")
        value_drawer.save_screenshot(f"policy_iteration_value_run{run + 1}.pdf")

        # Value Iteration
        environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
        environment_vi.set_nominal_direction_probability(0.8)
        value_solver = ValueIterator(environment_vi)
        value_solver.initialize()
        policy_drawer_vi = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
        value_solver.set_policy_drawer(policy_drawer_vi)
        value_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
        value_solver.set_value_function_drawer(value_drawer_vi)

        print("Running Value Iteration...")
        v_vi, pi_vi = value_solver.solve_policy()
        vi_value_iterations.append(value_solver._value_iterations)

        policy_drawer_vi.save_screenshot(f"value_iteration_policy_run{run + 1}.pdf")
        value_drawer_vi.save_screenshot(f"value_iteration_value_run{run + 1}.pdf")

    avg_pi_eval = np.mean(pi_eval_iterations)
    avg_pi_improve = np.mean(pi_improve_iterations)
    avg_vi_iterations = np.mean(vi_value_iterations)

    print("\nAverage Iteration Stats over", num_runs, "runs:")
    print(f"  Policy Iteration: Avg Evaluation iterations: {avg_pi_eval:.2f}, "
          f"Avg Improvement iterations: {avg_pi_improve:.2f} (Total = {avg_pi_eval + avg_pi_improve:.2f})")
    print(f"  Value Iteration: Avg Value sweeps: {avg_vi_iterations:.2f}")


def main():
    args = parse_args()
    airport_map, drawer_height = full_scenario()

    # If num_runs is 1 and no randomization flag, run the default map once.
    if args.num_runs == 1 and not args.randomize:
        run_single(airport_map, drawer_height)
    else:
        run_multiple(airport_map, drawer_height, args.num_runs, args.randomize)


if __name__ == '__main__':
    main()