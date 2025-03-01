#!/usr/bin/env python3
import argparse
import copy
import numpy as np
import time
import os
import pickle
from common.scenarios import full_scenario
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_iterator import ValueIterator
from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

"""
commandline args:
python q3_g.py --num_runs 10 --randomize
python q3_g.py --num_runs 1
"""

# output dirs
OUTPUT_DIR = 'q3g_plots'
RESULTS_DIR = 'q3g_results'

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Policy Iteration and Value Iteration")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--randomize", action="store_true")
    return parser.parse_args()


def run_single(airport_map, drawer_height):
    # build cache id
    cache_id = "default"
    pi_cache_file = os.path.join(RESULTS_DIR, f"pi_results_{cache_id}.pkl")
    vi_cache_file = os.path.join(RESULTS_DIR, f"vi_results_{cache_id}.pkl")
    
    # Check if we have cached results
    if os.path.exists(pi_cache_file) and os.path.exists(vi_cache_file):
        print("Loading cached results for default run...")
        with open(pi_cache_file, 'rb') as f:
            pi_results = pickle.load(f)
        with open(vi_cache_file, 'rb') as f:
            vi_results = pickle.load(f)
            
        v_pi, pi_pi = pi_results['v'], pi_results['pi']
        v_vi, pi_vi = vi_results['v'], vi_results['pi']
        pi_time = pi_results['time']
        vi_time = vi_results['time']
        
        airport_map_pi = copy.deepcopy(airport_map)
        airport_map_vi = copy.deepcopy(airport_map)
        
        # ------Policy Iteration Setup------
        environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
        environment_pi.set_nominal_direction_probability(0.8)
        policy_solver = PolicyIterator(environment_pi)
        policy_solver.set_gamma(0.95)
        policy_solver.initialize()
        
        # Set policy and value from cached results
        policy_solver._policy = pi_pi
        policy_solver._value_function = v_pi
        
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        
        value_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_drawer)
        
        # ------Value Iteration Setup------
        environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
        environment_vi.set_nominal_direction_probability(0.8)
        value_solver = ValueIterator(environment_vi)
        value_solver.set_gamma(0.95)
        value_solver.initialize()
        
        # Set policy and value from cached results
        value_solver._policy = pi_vi
        value_solver._value_function = v_vi
        
        policy_drawer_vi = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
        value_solver.set_policy_drawer(policy_drawer_vi)
        
        value_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
        value_solver.set_value_function_drawer(value_drawer_vi)
        
        # Print cached stats
        print("Policy Iteration Stats (cached):")
        print(f"  Evaluation iterations: {pi_results['eval_iterations']}")
        print(f"  Improvement iterations: {pi_results['improve_iterations']}")
        print(f"  Total runtime: {pi_time:.3f} seconds")
        
        print("Value Iteration Stats (cached):")
        print(f"  Value sweeps: {vi_results['value_iterations']}")
        print(f"  Total runtime: {vi_time:.3f} seconds")

    # no cache, we have to run algorithms
    else:
        # run both algortihms on exact same map
        airport_map_pi = copy.deepcopy(airport_map)
        airport_map_vi = copy.deepcopy(airport_map)

        # ------Policy Iteration------
        environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
        environment_pi.set_nominal_direction_probability(0.8)
        policy_solver = PolicyIterator(environment_pi)
        policy_solver.set_gamma(0.95)
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
        print(f"  Evaluation iterations: {policy_solver.get_total_evaluation_steps()}")
        print(f"  Improvement iterations: {policy_solver.get_total_improvement_steps()}")
        print(f"  Total runtime: {pi_time:.3f} seconds")
        
        # Cache results
        pi_results = {
            'v': v_pi,
            'pi': pi_pi,
            'time': pi_time,
            'eval_iterations': policy_solver.get_total_evaluation_steps(),
            'improve_iterations': policy_solver.get_total_improvement_steps()
        }
        with open(pi_cache_file, 'wb') as f:
            pickle.dump(pi_results, f)

        # ------Value Iteration------
        environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
        environment_vi.set_nominal_direction_probability(0.8)
        value_solver = ValueIterator(environment_vi)
        value_solver.set_gamma(0.95)
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
        
        # Cache results
        vi_results = {
            'v': v_vi, 
            'pi': pi_vi,
            'time': vi_time,
            'value_iterations': value_solver._value_iterations
        }
        with open(vi_cache_file, 'wb') as f:
            pickle.dump(vi_results, f)

    # Save PDFs to output directory
    policy_drawer.save_screenshot(os.path.join(OUTPUT_DIR, "policy_iteration_policy_default.pdf"))
    value_drawer.save_screenshot(os.path.join(OUTPUT_DIR, "policy_iteration_value_default.pdf"))
    policy_drawer_vi.save_screenshot(os.path.join(OUTPUT_DIR, "value_iteration_policy_default.pdf"))
    value_drawer_vi.save_screenshot(os.path.join(OUTPUT_DIR, "value_iteration_value_default.pdf"))


def run_multiple(airport_map, drawer_height, num_runs, randomize_flag):
    # Lists to accumulate iteration stats
    pi_eval_iterations = []
    pi_improve_iterations = []
    vi_value_iterations = []

    # Lists to store total runtimes
    pi_times = []
    vi_times = []

    for run in range(num_runs):
        # build cache id
        cache_id = f"run{run+1}_random{randomize_flag}"
        pi_cache_file = os.path.join(RESULTS_DIR, f"pi_results_{cache_id}.pkl")
        vi_cache_file = os.path.join(RESULTS_DIR, f"vi_results_{cache_id}.pkl")
        
        # Generate map for this run
        if randomize_flag:
            print(f"Generating randomized map for run {run+1}")
            tmp_map = copy.deepcopy(airport_map)
            temp_env = LowLevelEnvironment(tmp_map, randomize_obstacles_flag=True, obstacle_probability=0.01)
            randomized_map = temp_env.map()
            
            airport_map_pi = copy.deepcopy(randomized_map)
            airport_map_vi = copy.deepcopy(randomized_map)
        else:
            airport_map_pi = copy.deepcopy(airport_map)
            airport_map_vi = copy.deepcopy(airport_map)

        # Check if we have cached results for this run
        if os.path.exists(pi_cache_file) and os.path.exists(vi_cache_file):
            print(f"Loading cached results for run {run+1}...")
            with open(pi_cache_file, 'rb') as f:
                pi_results = pickle.load(f)
            with open(vi_cache_file, 'rb') as f:
                vi_results = pickle.load(f)
                
            v_pi, pi_pi = pi_results['v'], pi_results['pi']
            v_vi, pi_vi = vi_results['v'], vi_results['pi']
            pi_time = pi_results['time']
            vi_time = vi_results['time']
            
            # Use cached stats
            pi_times.append(pi_time)
            pi_eval_iterations.append(pi_results['eval_iterations'])
            pi_improve_iterations.append(pi_results['improve_iterations'])
            vi_times.append(vi_time)
            vi_value_iterations.append(vi_results['value_iterations'])
            
            # ------Policy Iteration Setup------
            environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
            environment_pi.set_nominal_direction_probability(0.8)
            policy_solver = PolicyIterator(environment_pi)
            policy_solver.set_gamma(0.95)
            policy_solver.initialize()
            
            # Set policy and value from cached results
            policy_solver._policy = pi_pi
            policy_solver._value_function = v_pi
            
            policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
            value_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
            
            # ------Value Iteration Setup------
            environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
            environment_vi.set_nominal_direction_probability(0.8)
            value_solver = ValueIterator(environment_vi)
            value_solver.set_gamma(0.95)
            value_solver.initialize()
            
            # Set policy and value from cached results
            value_solver._policy = pi_vi
            value_solver._value_function = v_vi
            
            policy_drawer_vi = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
            value_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
            
            print(f"Using cached results for run {run+1}")

        # again no cache, so run algos
        else:
            # ------Policy Iteration------
            environment_pi = LowLevelEnvironment(airport_map_pi, randomize_obstacles_flag=False)
            environment_pi.set_nominal_direction_probability(0.8)
            policy_solver = PolicyIterator(environment_pi)
            policy_solver.set_gamma(0.95)
            policy_solver.initialize()

            policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
            policy_solver.set_policy_drawer(policy_drawer)
            value_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
            policy_solver.set_value_function_drawer(value_drawer)

            print(f"Running Policy Iteration for run {run+1}...")
            start_time = time.time()
            v_pi, pi_pi = policy_solver.solve_policy()
            pi_time = time.time() - start_time

            pi_times.append(pi_time)
            pi_eval_iterations.append(policy_solver.get_total_evaluation_steps())
            pi_improve_iterations.append(policy_solver.get_total_improvement_steps())
            
            # Cache results
            pi_results = {
                'v': v_pi,
                'pi': pi_pi,
                'time': pi_time,
                'eval_iterations': policy_solver.get_total_evaluation_steps(),
                'improve_iterations': policy_solver.get_total_improvement_steps()
            }
            with open(pi_cache_file, 'wb') as f:
                pickle.dump(pi_results, f)

            # ------Value Iteration------
            environment_vi = LowLevelEnvironment(airport_map_vi, randomize_obstacles_flag=False)
            environment_vi.set_nominal_direction_probability(0.8)
            value_solver = ValueIterator(environment_vi)
            value_solver.set_gamma(0.95)
            value_solver.initialize()

            policy_drawer_vi = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
            value_solver.set_policy_drawer(policy_drawer_vi)
            value_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
            value_solver.set_value_function_drawer(value_drawer_vi)

            print(f"Running Value Iteration for run {run+1}...")
            start_time = time.time()
            v_vi, pi_vi = value_solver.solve_policy()
            vi_time = time.time() - start_time

            vi_times.append(vi_time)
            vi_value_iterations.append(value_solver._value_iterations)
            
            # Cache results
            vi_results = {
                'v': v_vi, 
                'pi': pi_vi,
                'time': vi_time,
                'value_iterations': value_solver._value_iterations
            }
            with open(vi_cache_file, 'wb') as f:
                pickle.dump(vi_results, f)

        # Save PDFs to output directory (always do this, whether cached or new)
        policy_drawer.save_screenshot(os.path.join(OUTPUT_DIR, f"policy_iteration_policy_run{run + 1}.pdf"))
        value_drawer.save_screenshot(os.path.join(OUTPUT_DIR, f"policy_iteration_value_run{run + 1}.pdf"))
        policy_drawer_vi.save_screenshot(os.path.join(OUTPUT_DIR, f"value_iteration_policy_run{run + 1}.pdf"))
        value_drawer_vi.save_screenshot(os.path.join(OUTPUT_DIR, f"value_iteration_value_run{run + 1}.pdf"))

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

    # Save summary results
    summary_file = os.path.join(RESULTS_DIR, f"summary_runs{num_runs}_random{randomize_flag}.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump({
            'pi_eval_iterations': pi_eval_iterations,
            'pi_improve_iterations': pi_improve_iterations,
            'vi_value_iterations': vi_value_iterations,
            'pi_times': pi_times,
            'vi_times': vi_times,
            'avg_pi_eval': avg_pi_eval,
            'avg_pi_improve': avg_pi_improve,
            'avg_pi_total': avg_pi_total,
            'avg_vi_iter': avg_vi_iter,
            'avg_pi_time': avg_pi_time,
            'avg_vi_time': avg_vi_time,
            'num_runs': num_runs,
            'randomize_flag': randomize_flag
        }, f)

    # ----  LaTeX Table ----
    latex_file = os.path.join(OUTPUT_DIR, f"comparison_table_runs{num_runs}_random{randomize_flag}.tex")
    with open(latex_file, 'w') as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tabular}{l c c c c}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Algorithm & Eval Iter/Sweeps & Improve Iter & Total Iter & Time (s)\\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(
            f"Policy Iteration & {avg_pi_eval:.2f} & {avg_pi_improve:.2f} & {avg_pi_total:.2f} & {avg_pi_time:.3f} \\\\" + "\n"
        )
        f.write(f"Value Iteration & {avg_vi_iter:.2f} & -- & {avg_vi_iter:.2f} & {avg_vi_time:.3f} \\\\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Comparison of Policy Iteration vs Value Iteration over " + 
            f"{num_runs} runs with randomize={randomize_flag}." + "\n")
        f.write(r"\label{tab:pi_vs_vi_comparison}" + "\n")
        f.write(r"\end{table}" + "\n")
    
    print(f"LaTeX table written to {latex_file}")


def main():
    args = parse_args()
    setup_directories()
    airport_map, drawer_height = full_scenario()

    if args.num_runs == 1 and not args.randomize:
        # Single run on the default map
        run_single(airport_map, drawer_height)
    else:
        # Multiple runs, randomised 
        run_multiple(airport_map, drawer_height, args.num_runs, args.randomize)


if __name__ == '__main__':
    main()