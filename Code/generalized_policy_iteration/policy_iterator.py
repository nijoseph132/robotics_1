'''
Created on 29 Jan 2022

@author: ucacsjj
'''

# This class implements the policy iterator algorithm.

import copy

from .dynamic_programming_base import DynamicProgrammingBase


class PolicyIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the policy evaluation algorithm
        # will be run before the for loop is exited.
        self._max_policy_evaluation_steps_per_iteration = 100
        
        
        # The maximum number of times the policy evaluation iteration
        # is carried out.
        self._max_policy_iteration_steps = 1000
        
        # Add tracking variables
        self._last_delta = float('inf')
        self._total_iterations = 0
        self._total_value_function_evaluation_steps = 0
        self._total_value_function_updates = 0

    # Perform policy evaluation for the current policy, and return
    # a copy of the state value function. Since this is a deep copy, you can modify it
    # however you like.
    def evaluate_policy(self):
        self._evaluate_policy()
        
        #v = copy.deepcopy(self._v)
        
        return self._v
        
    def solve_policy(self):
                            
        # Initialize the drawers if defined
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()

        # Reset termination indicators       
        policy_iteration_step = 0        
        policy_stable = False
        
        # Loop until either the policy converges or we ran out of steps        
        while (policy_stable is False) and \
            (policy_iteration_step < self._max_policy_iteration_steps):
            
            # Evaluate the policy
            self._evaluate_policy()

            # Improve the policy            
            policy_stable = self._improve_policy()
            
            # Update the drawers if needed
            if self._policy_drawer is not None:
                self._policy_drawer.update()
                
            if self._value_drawer is not None:
                self._value_drawer.update()
                
            # Update iteration counter
            self._total_iterations += 1
            policy_iteration_step += 1

        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()

        # Return the value function and policy of the solution
        return self._v, self._pi

        
    def _evaluate_policy(self):
        
        # Get the environment and map
        environment = self._environment
        map = environment.map()
        
        # Execute the loop at least once
        
        iteration = 0
        
        while True:
            
            delta = 0

            # Sweep systematically over all the states            
            for x in range(map.width()):
                for y in range(map.height()):
                    
                    # We skip obstructions and terminals. If a cell is obstructed,
                    # there's no action the robot can take to access it, so it doesn't
                    # count. If the cell is terminal, it executes the terminal action
                    # state. The value of the value of the terminal cell is the reward.
                    # The reward itself was set up as part of the initial conditions for the
                    # value function.
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue
                                       
                    # Unfortunately the need to use coordinates is a bit inefficient, due
                    # to legacy code
                    cell = (x, y)
                    
                    # Get the previous value function
                    old_v = self._v.value(x, y)

                    # Compute p(s',r|s,a)
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, \
                                                                                     self._pi.action(x, y))
                    
                    # Sum over the rewards
                    new_v = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))                        
                        
                    # Set the new value in the value function
                    self._v.set_value(x, y, new_v)
                    self._total_value_function_updates += 1
                                        
                    # Update the maximum deviation
                    delta = max(delta, abs(old_v-new_v))
 
            # Store the last delta value
            self._last_delta = delta
            
            # Increment counters
            iteration += 1
            self._total_value_function_evaluation_steps += 1
            
            print(f'Finished policy evaluation iteration {iteration}')
            
            # Terminate the loop if the change was very small
            if delta < self._theta:
                break
                
            # Terminate the loop if the maximum number of iterations is met. Generate
            # a warning
            if iteration >= self._max_policy_evaluation_steps_per_iteration:
                print('Maximum number of iterations exceeded')
                break

    def _improve_policy(self) -> bool:
        # Q3_c:
        # Implement the policy improvement step.
        # In this step, we update the policy by choosing, for each state, the action that
        # maximizes the expected return (i.e., the Q-value). This is based on the action value
        # function Q(s,a) = sum_{s'} P(s',r|s,a)[r + gamma * V(s')], as covered in Lecture 07.

        # Get the environment and map.
        environment = self._environment
        map = environment.map()

        # Assume the policy is stable until we find a state where it can be improved.
        policy_stable = True

        # Iterate over every state (cell) in the map.
        for x in range(map.width()):
            for y in range(map.height()):
                # Skip cells that are obstructions or terminal, as no action is taken there.
                if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                    continue
                
                # Define the current cell.
                cell = (x, y)
                
                # Retrieve the current action from the policy for this cell.
                old_action = self._pi.action(x, y)
                
                # Initialize variables for the best action and its Q-value.
                best_action = None
                best_q_value = float('-inf')
                
                # Iterate over all possible actions available in this state.
                # This loop implements the greedy policy improvement, as described in the slides.
                for action in environment.actions(cell):
                    # Obtain the next state and reward distribution for taking 'action' in 'cell'.
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, action)
                    
                    # Compute the expected return (Q-value) for this action.
                    # Q(s, a) = sum_{s'} P(s',r|s,a) * [r + gamma * V(s')]
                    q_value = 0
                    for i in range(len(p)):
                        next_state_coords = s_prime[i].coords()  # Get coordinates for the next state.
                        q_value += p[i] * (r[i] + self._gamma * self._v.value(next_state_coords[0], next_state_coords[1]))
                    
                    # If this action has a higher Q-value than the best seen so far, record it.
                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = action
                
                # If the best action is different from the current policy's action, update the policy.
                # This is the core of the policy improvement theorem:
                # A new policy that is greedy with respect to the current value function is an improvement.
                if best_action is not None and best_action != old_action:
                    self._pi.set_action(x, y, best_action)
                    # Mark that we made a change to the policy (i.e., it is not yet stable).
                    policy_stable = False
                
        # Return True if the policy did not change at all (i.e., it is stable and optimal).
        return policy_stable
                    
                
    def set_max_policy_evaluation_steps_per_iteration(self, \
                                                      max_policy_evaluation_steps_per_iteration):
            self._max_policy_evaluation_steps_per_iteration = max_policy_evaluation_steps_per_iteration


    def set_max_policy_iteration_steps(self, max_policy_iteration_steps):
        self._max_policy_iteration_steps = max_policy_iteration_steps
        
    def get_max_policy_iteration_steps(self):
        return self._max_policy_iteration_steps         
                
    # Add getter methods
    def get_last_delta(self):
        return self._last_delta

    def get_total_iterations(self):
        return self._total_iterations

    def get_total_value_function_evaluation_steps(self):
        return self._total_value_function_evaluation_steps
    
    def get_total_value_function_updates(self):
        return self._total_value_function_updates