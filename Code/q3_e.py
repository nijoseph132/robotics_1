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
from p2.low_level_actions import LowLevelActionType

import os
import time
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Directory constants
OUTPUT_DIR = 'q3e_plots'
RESULTS_DIR = 'q3e_results'
TEXT_LOG_FILE = 'q3e_result_logs.txt'
LATEX_TABLE_PATH = 'policy_iteration_results.tex'
CSV_OUTPUT_PATH = 'analysis_metrics.csv'

# Parameter ranges
DEFAULT_THETAS = [1, 1e-1, 1e-3, 1e-6]
DEFAULT_EVAL_STEPS = [5, 10, 20, 50, 100]


# calculate decimal similarity between two policies
# use to compare to our 'optimal' policy, computed with tight parameters
def compare_policies(optimal_policy, current_policy):
    matches = 0
    valid_cells = 0
    
    for x in range(optimal_policy.width()):
        for y in range(optimal_policy.height()):
            optimal_action = optimal_policy.action(x, y)
            
            # Only compare cells where optimal policy has a meaningful action
            if optimal_action != LowLevelActionType.NONE:
                valid_cells += 1
                if optimal_action == current_policy.action(x, y):
                    matches += 1
                    
    return matches / valid_cells if valid_cells > 0 else 0.0


def get_optimal_policy():
    # either compute or load optimal policy
    baseline_pickle_path = os.path.join(RESULTS_DIR, 'baseline_policy.pkl')
    if os.path.exists(baseline_pickle_path):
        print("Loading cached baseline policy...")
        with open(baseline_pickle_path, 'rb') as pkl_file:
            baseline_data = pickle.load(pkl_file)
            baseline_result = baseline_data['result']
            baseline_policy = baseline_result['policy']
    else:
        print("Computing new baseline policy with strict parameters (theta=1e-6, eval_steps=10,000)...")
        strict_baseline_theta = 1e-6
        strict_baseline_eval_steps = 10000
        baseline_result = run_policy_evaluation_test(strict_baseline_theta, strict_baseline_eval_steps)
        baseline_policy = baseline_result['policy']
        # cache
        with open(baseline_pickle_path, 'wb') as pkl_file:
            pickle.dump({
                'result': baseline_result,
                'parameters': {
                    'theta': strict_baseline_theta,
                    'eval_steps': strict_baseline_eval_steps,
                }
            }, pkl_file)
    return baseline_policy


def run_policy_evaluation_test(theta, eval_steps):
    airport_map, _ = full_scenario()
    airport_environment = LowLevelEnvironment(airport_map)
    airport_environment.set_nominal_direction_probability(0.8)

    # configure PolicyIterator
    policy_solver = PolicyIterator(airport_environment)
    policy_solver.set_theta(theta)
    policy_solver.set_max_policy_evaluation_steps_per_iteration(eval_steps)
    policy_solver.set_gamma(0.9)

    # we dont want this to be hit 
    policy_solver.set_max_policy_iteration_steps(10000) 

    # No GUI updates to speed up
    policy_solver.set_policy_drawer(None)
    policy_solver.set_value_function_drawer(None)

    policy_solver.initialize()
    start_time = time.time()
    v, pi = policy_solver.solve_policy()
    end_time = time.time()

    return {
        'time': end_time - start_time,
        # number of times state space is swept in policy evaluation
        'total_evaluation_steps': policy_solver.get_total_state_space_sweeps(),
        # number of times policy iteration is run 
        'outer_iterations': policy_solver.get_total_iterations(),
        'policy': pi
    }


def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def initialize_text_log():
    """Initialize the text log file by clearing it if it exists."""
    if os.path.exists(TEXT_LOG_FILE):
        with open(TEXT_LOG_FILE, 'w'):
            pass 


def run_parameter_grid_search(thetas, eval_steps_list, baseline_policy):
    """Add baseline_policy parameter and similarity calculations"""
    results = []
    
    for theta in thetas:
        for e_steps in eval_steps_list:
            # Create param-specific pickle name
            param_filename = f"policy_iter_t{theta:.0e}_e{e_steps}.pkl"
            pickle_path = os.path.join(RESULTS_DIR, param_filename)

            if os.path.exists(pickle_path):
                print(f"Loading cached results for θ={theta:.0e}, eval_steps={e_steps}")
                with open(pickle_path, 'rb') as pkl_file:
                    cached_data = pickle.load(pkl_file)
                    result = cached_data['result']
                    similarity = cached_data.get('similarity', 0.0) 
            else:
                print(f"Testing θ={theta:.0e}, eval_steps={e_steps}")
                result = run_policy_evaluation_test(theta, e_steps)
                similarity = compare_policies(baseline_policy, result['policy'])

                with open(pickle_path, 'wb') as pkl_file:
                    pickle.dump({
                        'result': result,
                        'similarity': similarity
                    }, pkl_file)

            results.append({
                'theta': theta,
                'eval_steps': e_steps,
                'time': result['time'],
                'total_evaluation_steps': result['total_evaluation_steps'],
                'outer_iterations': result['outer_iterations'],
                'similarity': similarity
            })

            # Update log function to include similarity
            log_result_to_text_file(theta, e_steps, result, similarity)
            
            print(f"Time: {result['time']:.3f}s, "
                  f"Evaluation Sweeps: {result['total_evaluation_steps']}, "
                  f"Outer Iterations: {result['outer_iterations']}, "
                  f"Similarity: {similarity:.4f}")
            print("-" * 80)
            
    return results


def log_result_to_text_file(theta, eval_steps, result, similarity):
    with open(TEXT_LOG_FILE, 'a') as f:
        f.write(f"Parameters:\n")
        f.write(f"  Theta={theta:.0e}, eval_steps={eval_steps}\n")
        f.write(f"  Time: {result['time']:.3f}s\n")
        f.write(f"  Total evaluation sweeps: {result['total_evaluation_steps']}\n")
        f.write(f"  Outer policy iterations: {result['outer_iterations']}\n")
        f.write(f"  Similarity to baseline: {similarity:.4f}\n")
        f.write("-" * 50 + "\n\n")


def save_final_results(results, thetas, eval_steps_list, baseline_policy):
    # Save pickle with all results and parameters
    final_pickle_path = os.path.join(RESULTS_DIR, 'policy_iteration_results.pkl')
    with open(final_pickle_path, 'wb') as pkl_file:
        pickle.dump({
            'results': results,
            'baseline_policy': baseline_policy,
            'parameters': {
                'thetas': thetas,
                'eval_steps': eval_steps_list,
            }
        }, pkl_file)

    # convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Logged raw results to {CSV_OUTPUT_PATH}")
    
    return df


def generate_latex_table(results):
    """Modified to include similarity column"""
    with open(LATEX_TABLE_PATH, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l c c c c c}\n")  
        f.write("\\hline\n")
        f.write("Theta & EvalSteps & Time (s) & TotalEvalSteps & Policy Iteration Steps & Similarity \\\\\n")  
        f.write("\\hline\n")

        for row in results:
            f.write(f"{row['theta']:g} & {row['eval_steps']} & {row['time']:.3f} & "
                    f"{row['total_evaluation_steps']} & {row['outer_iterations']} & {row['similarity']:.4f} \\\\\n") 

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Results across different $\theta$ and eval\_steps, measuring time, total evaluation steps, policy iterations, and similarity to baseline.}\n")
        f.write("\\label{tab:theta_evalsteps_results}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table written to {LATEX_TABLE_PATH}")


def create_analysis_plots(df):
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", n_colors=len(df['theta'].unique()))
    
    # Plot 1: Heatmap for Time
    plt.figure(figsize=(8, 6))
    create_heatmap(df, 'time', 'YlGnBu_r', 'Runtime (seconds) by Parameters', plt.gca())
    plt.tight_layout()
    heatmap_path_png = os.path.join(OUTPUT_DIR, "runtime_heatmap.png")
    heatmap_path_pdf = os.path.join(OUTPUT_DIR, "runtime_heatmap.pdf")
    plt.savefig(heatmap_path_png)
    plt.savefig(heatmap_path_pdf)
    print(f"Saved heatmap to {heatmap_path_png}")
    
    # Plot 2: Policy Iteration Steps vs. Runtime
    plt.figure(figsize=(8, 6))
    create_scatter_plot(df, 'outer_iterations', 'time', 
                        'Policy Iteration Steps', 'Runtime (seconds)', 
                        'Policy Iteration Steps vs. Runtime', palette, plt.gca())
    plt.tight_layout()
    scatter_path_png = os.path.join(OUTPUT_DIR, "iterations_vs_runtime.png")
    scatter_path_pdf = os.path.join(OUTPUT_DIR, "iterations_vs_runtime.pdf")
    plt.savefig(scatter_path_png)
    plt.savefig(scatter_path_pdf)
    print(f"Saved scatter plot to {scatter_path_png}")
    
    plt.show()

def create_scatter_plot(df, x_col, y_col, x_label, y_label, title, palette, ax):
    """Create a scatter plot on the given axis."""
    sns.scatterplot(
        data=df, x=x_col, y=y_col, hue='theta',
        palette=palette, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def create_heatmap(df, value_col, cmap, title, ax):
    """Create a heatmap on the given axis."""
    pivot_table = df.pivot_table(index='theta', columns='eval_steps', values=value_col)
    fmt = ".3f" if value_col == 'similarity' else ".2f"
    sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt=fmt, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Eval Steps")
    ax.set_ylabel("Theta")

def main():
    setup_directories()
    initialize_text_log()
    
    # define parameter grids
    thetas = DEFAULT_THETAS
    eval_steps_list = DEFAULT_EVAL_STEPS
    
    # get optimal policy for comparison
    baseline_policy = get_optimal_policy()
    
    # gridsearch
    results = run_parameter_grid_search(thetas, eval_steps_list, baseline_policy)
    
    # save rsults
    df = save_final_results(results, thetas, eval_steps_list, baseline_policy)
    
    # generate LaTeX table
    generate_latex_table(results)
    
    # create analysis plots
    create_analysis_plots(df)
        

if __name__ == '__main__':
    main()