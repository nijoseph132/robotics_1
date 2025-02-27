'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from .dynamic_programming_base import DynamicProgrammingBase

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
        self._value_iterations = 0

    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3f:
    # Finish the implementation of the methods below.
    
    def _compute_optimal_value_function(self):

        environment = self._environment
        map_obj = environment.map()
        iteration = 0

        while iteration < self._max_optimal_value_function_iterations:
            delta = 0.0
            # loop over every non-obstruction, non-terminal state.
            for x in range(map_obj.width()):
                for y in range(map_obj.height()):
                    if map_obj.cell(x, y).is_obstruction() or map_obj.cell(x, y).is_terminal():
                        continue

                    # current state
                    cell = (x, y)
                    # store the current value for comparison.
                    old_value = self._v.value(x, y)
                    best_value = float('-inf')
                    # loop over all available actions.
                    for action in environment.get_actions(cell):
                        s_primes, rewards, probs = environment.next_state_and_reward_distribution((x, y), action)
                        q_value = 0.0
                        # accumulate the expected reward for this action.
                        for i in range(len(probs)):
                            next_state = s_primes[i].coords() 
                            q_value += probs[i] * (rewards[i] + self._gamma * self._v.value(next_state[0], next_state[1]))
                        best_value = max(best_value, q_value)
                    # update the value for state (x,y).
                    self._v.set_value(x, y, best_value)
                    delta = max(delta, abs(old_value - best_value))
            iteration += 1
            self._value_iterations = iteration
            if delta < self._theta:
                break

    def _extract_policy(self):

        environment = self._environment
        map_obj = environment.map()
        
        for x in range(map_obj.width()):
            for y in range(map_obj.height()):
                if map_obj.cell(x, y).is_obstruction() or map_obj.cell(x, y).is_terminal():
                    continue

                cell = (x, y)
                best_action = None
                best_q_value = float('-inf')
                for action in environment.get_actions(cell):
                    s_primes, rewards, probs = environment.next_state_and_reward_distribution((x, y), action)
                    q_value = 0.0
                    for i in range(len(probs)):
                        next_state = s_primes[i].coords()
                        q_value += probs[i] * (rewards[i] + self._gamma * self._v.value(next_state[0], next_state[1]))
                    if q_value > best_q_value:
                        best_q_value = q_value
                        best_action = action
                self._pi.set_action(x, y, best_action)