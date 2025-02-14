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
import time 

def run_policy_iteration(theta, max_eval_steps):
    # Get the full scenario map.
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment.
    airport_environment = LowLevelEnvironment(airport_map)
    airport_environment.set_nominal_direction_probability(0.8)
    
    # Create & configure the policy iterator.
    policy_solver = PolicyIterator(airport_environment)
    policy_solver.set_theta(theta)
    policy_solver.set_max_policy_evaluation_steps_per_iteration(max_eval_steps)
    policy_solver.initialize()
    
    # Solve the policy iteration while measuring runtime.
    start_time = time.time()
    policy_solver.solve_policy()
    runtime = time.time() - start_time
    
    eval_iters = policy_solver._policy_eval_iterations
    improve_iters = policy_solver._policy_improve_iterations
    
    return runtime, eval_iters, improve_iters, drawer_height

def main():
    # Parameter settings to explore:
    theta_values = [0.1, 0.5, 1, 5, 10]      
    max_eval_steps_values = [5, 10, 20, 50, 500] 
    
    results = []
    for theta in theta_values:
        for max_eval in max_eval_steps_values:
            runtime, eval_iters, improve_iters, _ = run_policy_iteration(theta, max_eval)
            results.append((theta, max_eval, runtime, eval_iters, improve_iters))
    
    # print into latex table for rep
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{l l l l l}")
    print(r"\hline")
    print(r"Theta & Max Eval Steps & Runtime (sec) & Policy Eval Iters & Policy Improve Iters \\")
    print(r"\hline")
    for theta, max_eval, runtime, eval_iters, improve_iters in results:
        print(f"{theta} & {max_eval} & {runtime:.3f} & {eval_iters} & {improve_iters} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Policy iteration evaluation results for various $\theta$ values and maximum evaluation steps.}")
    print(r"\label{tab:policy_iteration_results}")
    print(r"\end{table}")
    
if __name__ == '__main__':
    main()