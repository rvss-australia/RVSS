import numpy as np
from RL_Support.gym_simple_gridworlds.envs.grid_env import GridEnv
from RL_Support.gym_simple_gridworlds.envs.grid_2dplot import plot_value_function, plot_policy
from collections import defaultdict
from copy import deepcopy
from matplotlib import pyplot as plt


def encode_policy(grid_env, policy_matrix=None):
    """
     Convert deterministic policy matrix into stochastic policy representation

     :param grid_env: MDP environment
     :param policy_matrix: Deterministic policy matrix (one action per state)

     :return: (dict of dict) Dictionary of dictionaries where each element corresponds to the
             probability of selection an action a at a given state s
     """

    height, width = grid_env.grid.shape

    if policy_matrix is None:

        policy_matrix = np.array([[3,      3,  3,  -1],
                                  [0, np.NaN,  0,  -1],
                                  [0,      2,  0,   2]])

    result_policy = defaultdict(lambda: defaultdict(float))

    for i in range(height):
        for j in range(width):
            s = grid_env.grid[i, j]
            if np.isnan(s) or grid_env.is_terminal_state(i, j):
                continue

            for a, _ in grid_env.ACTIONS.items():
                result_policy[int(s)][int(a)] = 0.0

            if policy_matrix[i, j] >= 0 or not np.isnan(policy_matrix[i, j]):
                result_policy[int(s)][int(policy_matrix[i, j])] = 1.0

    return result_policy

def define_random_policy(grid_env):
    """
    Define random deterministic policy for given environment

    :param grid_env: MDP environment
    :return: (matrix) Deterministic policy matrix
    """
    np.random.seed(grid_env.seed()[0])

    policy_matrix = np.array([np.random.choice(grid_env.get_actions(), 4).tolist(),
                              np.random.choice(grid_env.get_actions(), 4).tolist(),
                              np.random.choice(grid_env.get_actions(), 4).tolist()])

    for (x, y) in grid_env.terminal_states:
        policy_matrix[x, y] = -1

    for (x, y) in grid_env.obstacles:
        policy_matrix[x, y] = -1

    return policy_matrix

def one_step_look_ahead(grid_env, state, value_function):
    """
     Compute the action-value function for a state $s$ given the state-value function $v$.
     
     :param grid_env (GridEnv): MDP environment
     :param state (int): state for which we are looking one action ahead
     :param value_function (dict): state-value function associated to a given policy py
     
     :return: (np.array) Action-value function for all actions available at state s
    """
    action_values = []
    
    for action in grid_env.get_actions():
        discounted_value = 0
        for s_next in grid_env.get_states():
             discounted_value += grid_env.state_transitions[state, action, s_next] * value_function[s_next]
        
        q_a = grid_env.rewards[state, action] + grid_env.gamma * discounted_value
        action_values.append(q_a)
    
    return np.array(action_values)

def update_policy(grid_env, cur_policy, value_function):
    """
     Update a given policy based on a given value_function
     
     :param grid_env (GridEnv): MDP environment
     :param cur_policy (matrix form): Policy to update
     :param value_function: state-value function associated to a policy cur_policy
     
     :return: (dict) Updated policy
    """
    
    states = grid_env.get_states(exclude_terminal=True)
    
    for s in states:
        # Obtain state-action values for state s using the helper function one_step_look_ahead
        action_values = one_step_look_ahead(grid_env, s, value_function)
        
        # Find (row, col) coordinates of cell with index s
        row, col = np.argwhere(grid_env.grid == s)[0]
        
        cur_policy[row, col] = np.argmax(action_values)
        
    return cur_policy
    

def decode_policy(grid_env, policy=None):
    """
     Convert stochastic policy representation (dict of dict) to deterministic policy matrix
     :param grid_env: MDP environment
     :param policy: stochastic policy (probability of each action at each state)
     :return: (matrix) Deterministic policy matrix (one action per state)
     """

    height, width = grid_env.grid.shape
    policy_matrix = np.full((height, width), -1)

    for s, actions in policy.items():
        x, y = np.argwhere(grid_env.grid == s)[0]

        if not grid_env.is_terminal_state(x,y):
          action_keys = list(actions.keys())
          policy_matrix[x, y] = action_keys[np.argmax(list(actions.values()))]

    return policy_matrix
