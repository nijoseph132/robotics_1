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
import math
import numpy as np
from p2.low_level_actions import LowLevelActionType
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_policy_evaluation_test(theta, eval_steps):
    # Initialize the airport environment with a nominal direction probability of 0.8
    # This means the agent will move in the intended direction 80% of the time
    airport_map, _ = full_scenario()
    airport_environment = LowLevelEnvironment(airport_map)
    airport_environment.set_nominal_direction_probability(0.8)

    # Configure the policy solver with the given parameters
    policy_solver = PolicyIterator(airport_environment)
    policy_solver.set_theta(theta)
    policy_solver.set_max_policy_evaluation_steps_per_iteration(eval_steps)
    policy_solver.set_gamma(0.9)
    policy_solver.set_max_policy_iteration_steps(100)
    
    policy_solver.set_policy_drawer(None)
    policy_solver.set_value_function_drawer(None)

    policy_solver.initialize()

    start_time = time.time()
    v, pi = policy_solver.solve_policy()
    end_time = time.time()
    
    return {
        'time': end_time - start_time,
        'evaluation_steps': policy_solver.get_total_value_function_evaluation_steps(),
        'policy': pi
    }

def analyze_value_function(v):
    # Extract and analyze non-NaN values from the value function grid
    # Returns mean and standard deviation of the values
    values = []
    for x in range(v.width()):
        for y in range(v.height()):
            val = v.value(x, y)
            if not math.isnan(val):
                values.append(val)
    return {
        'mean': np.mean(values),
        'std': np.std(values)
    }

def analyze_policy(pi):
    actions = []
    for x in range(pi.width()):
        for y in range(pi.height()):
            action = pi.action(x, y)
            if action != LowLevelActionType.NONE:
                actions.append(action)
    return {
        'action_distribution': np.bincount(actions) / len(actions)
    }

def compare_policies(baseline_policy, current_policy):
    # Calculate the similarity between two policies
    # Returns the fraction of matching actions in valid cells (excluding NONE actions)
    matches = 0
    total = 0
    for x in range(baseline_policy.width()):
        for y in range(baseline_policy.height()):
            base_action = baseline_policy.action(x, y)
            curr_action = current_policy.action(x, y)
            if base_action != LowLevelActionType.NONE:
                total += 1
                if base_action == curr_action:
                    matches += 1
    return matches / total if total > 0 else 0

def compare_action_distributions(dist1, dist2):
    return np.sum(np.abs(dist1 - dist2))

if __name__ == '__main__':
    # Create output directories if they don't exist
    os.makedirs('simulation_outputs', exist_ok=True)
    os.makedirs('simulation_results', exist_ok=True)
    
    # Define parameter ranges for the grid search
    thetas = [10, 1, 1e-1, 1e-3, 1e-6]        # Convergence thresholds
    eval_steps = [1, 3, 5, 10, 20, 50, 100]   # Maximum policy evaluation steps
    
    print("Computing baseline policy...")
    baseline_pickle_path = os.path.join('simulation_results', 'baseline_policy.pkl')
    
    # Load or compute the baseline policy using strict parameters
    if os.path.exists(baseline_pickle_path):
        print("Loading cached baseline policy...")
        with open(baseline_pickle_path, 'rb') as pkl_file:
            baseline_data = pickle.load(pkl_file)
            baseline_result = baseline_data['result']
            baseline_policy = baseline_result['policy']
    else:
        print("Computing new baseline policy...")
        # Use strict parameters (low theta, high eval_steps) for baseline
        baseline_result = run_policy_evaluation_test(1e-6, 100)
        baseline_policy = baseline_result['policy']
        
        # Cache the baseline policy for future runs
        with open(baseline_pickle_path, 'wb') as pkl_file:
            pickle.dump({
                'result': baseline_result,
                'parameters': {
                    'theta': 1e-6,
                    'eval_steps': 10,
                }
            }, pkl_file)

    # Grid search over all parameter combinations
    results = []
    for theta in thetas:
        for e_steps in eval_steps:
            # Generate unique filename for this parameter combination
            param_filename = f"policy_iter_t{theta:.0e}_e{e_steps}.pkl"
            pickle_path = os.path.join('simulation_results', param_filename)
            
            # Load cached results if available, otherwise compute new results
            if os.path.exists(pickle_path):
                print(f"Loading cached results for θ={theta:.0e}, eval_steps={e_steps}")
                with open(pickle_path, 'rb') as pkl_file:
                    cached_data = pickle.load(pkl_file)
                    result = cached_data['result']
                    similarity = cached_data['similarity']
            else:
                print(f"Testing θ={theta:.0e}, eval_steps={e_steps}")
                result = run_policy_evaluation_test(theta, e_steps)
                # Compare current policy with baseline to measure quality
                similarity = compare_policies(baseline_policy, result['policy'])

            results.append({
                'theta': theta,
                'eval_steps': e_steps,
                'time': result['time'],
                'total_eval_steps': result['evaluation_steps'],
                'similarity': similarity
            })
                
            with open('policy_iteration_results.txt', 'a') as f:
                f.write(f"Parameters:\n")
                f.write(f"Theta={theta:.0e}, eval_steps={e_steps}\n")
                f.write(f"Time: {result['time']:.3f}s\n")
                f.write(f"Total evaluation steps: {result['evaluation_steps']}\n")
                f.write(f"Similarity to baseline: {similarity:.4f}\n")
                f.write("-" * 50 + "\n\n")
                
                print(f"Time: {result['time']:.3f}s")
                print(f"Total evaluation steps: {result['evaluation_steps']}")
                print(f"Similarity to baseline: {similarity:.4f}")
                print("-" * 80)

    final_pickle_path = os.path.join('simulation_results', 'policy_iteration_results.pkl')
    with open(final_pickle_path, 'wb') as pkl_file:
        pickle.dump({
            'results': results,
            'baseline_policy': baseline_policy,
            'parameters': {
                'thetas': thetas,
                'eval_steps': eval_steps,
            }
        }, pkl_file)

    results.sort(key=lambda x: x['time'])

    print("\nTop 5 Fastest Parameter Combinations:")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. Time: {r['time']:.3f}s")
        print(f"θ={r['theta']:.0e}, eval_steps={r['eval_steps']}")
        print(f"Total evaluation steps: {r['total_eval_steps']}")
        print(f"Similarity to baseline: {r['similarity']:.4f}")

    print("\nGenerating analysis plots...")
    
    df = pd.DataFrame(results)

    input_vars = ['theta', 'eval_steps']
    output_vars = ['time', 'total_eval_steps', 'similarity']
    correlation_matrix = df[input_vars + output_vars].corr().loc[input_vars, output_vars]
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Generate visualization plots for analysis
    plt.figure(figsize=(15, 10))

    # Plot 1: Correlation heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Parameters and Metrics')

    # Plot 2: Removed Gamma vs Similarity analysis
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='eval_steps', y='similarity')
    plt.title('Eval Steps vs Similarity to Baseline')

    # Plot 3: Performance analysis
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='eval_steps', y='time', hue='theta')
    plt.title('Eval Steps vs Time (colored by theta)')

    # Plot 4: Convergence analysis
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='theta', y='total_eval_steps')
    plt.title('Theta vs Total Evaluation Steps')

    plt.tight_layout()
    
    plt.savefig('simulation_outputs/parameter_analysis.pdf')
    plt.savefig('simulation_outputs/parameter_analysis.png')
    
    print("\nKey Findings:")
    print("\nStrongest correlations:")
    correlations = correlation_matrix.unstack()
    sorted_correlations = correlations[correlations != 1.0].sort_values(ascending=False)
    print(sorted_correlations[:5])

    plt.show()