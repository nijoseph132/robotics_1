#!/usr/bin/env python3
import argparse
import copy
import numpy as np
import time
from common.scenarios import full_scenario
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_iterator import ValueIterator
from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

"""
python q3_g.py --num_runs 10 --randomize
python q3_g.py --num_runs 1
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Policy Iteration and Value Iteration")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--randomize", action="store_true")
    return parser.parse_args()


def run_single(airport_map, drawer_height):
    # run both algortihms on exact same map
    airport_map_pi = copy.deepcopy(airport_map)
    airport_map_vi = copy.deepcopy(airport_map)

    environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
    environment_pi.set_nominal_direction_probability(0.8)
    policy_solver = PolicyIterator(environment_pi)
    policy_solver.initialize()

    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)

    value_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_drawer)

    print("Running Policy Iteration...")
    start_time = time.time()
    v_pi, pi_pi = policy_solver.solve_policy()
    pi_time = time.time() - start_time

    print("Policy Iteration Stats:")
    print(f"  Evaluation iterations: {policy_solver._policy_eval_iterations}")
    print(f"  Improvement iterations: {policy_solver._policy_improve_iterations}")
    print(f"  Total runtime: {pi_time:.3f} seconds")

    policy_drawer.save_screenshot("policy_iteration_policy_default.pdf")
    value_drawer.save_screenshot("policy_iteration_value_default.pdf")

    # =============== Value Iteration ===============
    environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
    environment_vi.set_nominal_direction_probability(0.8)
    value_solver = ValueIterator(environment_vi)
    value_solver.initialize()

    policy_drawer_vi = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
    value_solver.set_policy_drawer(policy_drawer_vi)

    value_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
    value_solver.set_value_function_drawer(value_drawer_vi)

    print("Running Value Iteration...")
    start_time = time.time()
    v_vi, pi_vi = value_solver.solve_policy()
    vi_time = time.time() - start_time

    print("Value Iteration Stats:")
    print(f"  Value sweeps: {value_solver._value_iterations}")
    print(f"  Total runtime: {vi_time:.3f} seconds")

    policy_drawer_vi.save_screenshot("value_iteration_policy_default.pdf")
    value_drawer_vi.save_screenshot("value_iteration_value_default.pdf")


def run_multiple(airport_map, drawer_height, num_runs, randomize_flag):
    # Lists to accumulate iteration stats
    pi_eval_iterations = []
    pi_improve_iterations = []
    vi_value_iterations = []

    # Lists to store total runtimes
    pi_times = []
    vi_times = []

    for run in range(num_runs):
        if randomize_flag:
            tmp_map = copy.deepcopy(airport_map)
            temp_env = LowLevelEnvironment(tmp_map, randomize_obstacles_flag=True, obstacle_probability=0.01)
            randomized_map = temp_env.map()
            airport_map_pi = copy.deepcopy(randomized_map)
            airport_map_vi = copy.deepcopy(randomized_map)
        else:
            airport_map_pi = copy.deepcopy(airport_map)
            airport_map_vi = copy.deepcopy(airport_map)

        # =============== Policy Iteration ===============
        environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
        environment_pi.set_nominal_direction_probability(0.8)
        policy_solver = PolicyIterator(environment_pi)
        policy_solver.initialize()

        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        value_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_drawer)

        print("Running Policy Iteration...")
        start_time = time.time()
        v_pi, pi_pi = policy_solver.solve_policy()
        pi_time = time.time() - start_time

        pi_times.append(pi_time)
        pi_eval_iterations.append(policy_solver._policy_eval_iterations)
        pi_improve_iterations.append(policy_solver._policy_improve_iterations)

        policy_drawer.save_screenshot(f"policy_iteration_policy_run{run + 1}.pdf")
        value_drawer.save_screenshot(f"policy_iteration_value_run{run + 1}.pdf")

        # =============== Value Iteration ===============
        environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
        environment_vi.set_nominal_direction_probability(0.8)
        value_solver = ValueIterator(environment_vi)
        value_solver.initialize()

        policy_drawer_vi = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
        value_solver.set_policy_drawer(policy_drawer_vi)
        value_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
        value_solver.set_value_function_drawer(value_drawer_vi)

        print("Running Value Iteration...")
        start_time = time.time()
        v_vi, pi_vi = value_solver.solve_policy()
        vi_time = time.time() - start_time

        vi_times.append(vi_time)
        vi_value_iterations.append(value_solver._value_iterations)

        policy_drawer_vi.save_screenshot(f"value_iteration_policy_run{run + 1}.pdf")
        value_drawer_vi.save_screenshot(f"value_iteration_value_run{run + 1}.pdf")

    avg_pi_eval = np.mean(pi_eval_iterations)
    avg_pi_improve = np.mean(pi_improve_iterations)
    avg_pi_total = avg_pi_eval + avg_pi_improve
    avg_vi_iter = np.mean(vi_value_iterations)

    avg_pi_time = np.mean(pi_times)
    avg_vi_time = np.mean(vi_times)

    print("\nAverage Iteration Stats over", num_runs, "runs:")
    print(f"  Policy Iteration: Avg Eval= {avg_pi_eval:.2f}, Avg Improve= {avg_pi_improve:.2f}, "
          f"(Total= {avg_pi_total:.2f}), Time= {avg_pi_time:.3f}s")
    print(f"  Value Iteration: Avg Sweeps= {avg_vi_iter:.2f}, Time= {avg_vi_time:.3f}s")

    # ---- Print LaTeX Table ----
    print("\n========== LaTeX Table of Average Results ==========")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\hline")
    print(r"Algorithm & Eval Iter/Sweeps & Improve Iter & Total Iter & Time (s)\\")
    print(r"\hline")
    print(
        f"Policy Iteration & {avg_pi_eval:.2f} & {avg_pi_improve:.2f} & {avg_pi_total:.2f} & {avg_pi_time:.3f} \\\\"
    )
    print(f"Value Iteration & {avg_vi_iter:.2f} & -- & {avg_vi_iter:.2f} & {avg_vi_time:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Comparison of Policy Iteration vs Value Iteration over "
      f"{num_runs} runs with randomize={randomize_flag}.")
    print(r"\label{tab:pi_vs_vi_comparison}")
    print(r"\end{table}")
    print("=====================================================\n")


def main():
    args = parse_args()
    airport_map, drawer_height = full_scenario()

    if args.num_runs == 1 and not args.randomize:
        # Single run on the default map
        run_single(airport_map, drawer_height)
    else:
        # Multiple runs, randomised 
        run_multiple(airport_map, drawer_height, args.num_runs, args.randomize)


if __name__ == '__main__':
    main()