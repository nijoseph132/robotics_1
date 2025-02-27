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
        
        
        # The maximum number of times the policy evaluation and improvement 
        # outer loop is carried out
        self._max_policy_iteration_steps = 1000
        
        # tracking variables
        self._last_delta = float('inf')
        # number of times policy iteration is run
        self._total_iterations = 0
        # number of times policy is improved
        self._policy_improvement_steps = 0
        # number of times policy is evaluated
        self._policy_evaluation_steps = 0

        # number of times state space is swept in policy evaluation
        self._state_space_sweeps = 0

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
            self._policy_evaluation_steps += 1

            # Improve the policy            
            policy_stable = self._improve_policy()
            self._policy_improvement_steps += 1
            
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
                    # Update the maximum deviation
                    delta = max(delta, abs(old_v-new_v))
 
            # Store the last delta value
            self._last_delta = delta

            self._state_space_sweeps += 1

            
            # Increment counters
            iteration += 1            
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
                
                # current cell/state
                cell = (x, y)
                
                # get action from policy
                old_action = self._pi.action(x, y)
                
                # Initialize variables for the best action and its Q-value.
                best_action = None
                best_q_value = float('-inf')
                
                # iterate over all possible actions
                for action in environment.actions(cell):
                    # get next state and reward distribution
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, action)
                    
                    # compute q-value
                    q_value = 0
                    for i in range(len(p)):
                        next_state_coords = s_prime[i].coords()
                        q_value += p[i] * (r[i] + self._gamma * self._v.value(next_state_coords[0], next_state_coords[1]))
                    
                    # if q-value is better than the best q-value, update best q-value and best action
                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = action

                # if the best action that we just found is different from the current policy's action, update the policy.
                # the policy is therefore not stable, so we set policy_stable to false so that policy iteration can continue.
                if best_action is not None and best_action != old_action:
                    self._pi.set_action(x, y, best_action)
                    policy_stable = False
        
        # if this is true, policy is stable and we can stop, otherwise we need to continue
        return policy_stable
                    
                
    def set_max_policy_evaluation_steps_per_iteration(self, \
                                                      max_policy_evaluation_steps_per_iteration):
            self._max_policy_evaluation_steps_per_iteration = max_policy_evaluation_steps_per_iteration


    def set_max_policy_iteration_steps(self, max_policy_iteration_steps):
        self._max_policy_iteration_steps = max_policy_iteration_steps
        
    def get_max_policy_iteration_steps(self):
        return self._max_policy_iteration_steps         
                
    def get_last_delta(self):
        return self._last_delta

    def get_total_iterations(self):
        return self._total_iterations

    def get_total_evaluation_steps(self):
        return self._policy_evaluation_steps
    
    def get_total_improvement_steps(self):
        return self._policy_improvement_steps
    
    def get_total_state_space_sweeps(self):
        return self._state_space_sweeps
