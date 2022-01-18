import numpy as np
from gym_simple_gridworlds.envs.grid_env import GridEnv
from gym_simple_gridworlds.envs.grid_2dplot import plot_value_function, plot_policy
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


def policy_evaluation(env, policy):

    v = {s: 0.0 for s in env.get_states()}
    theta = 0.0001
    delta = 1000

    while delta > theta:
        delta = 0.0
        for s in v.keys():

            old_v = v[s]
            new_v = 0

            for action, probability in policy[s].items():
                state_sum = 0
                for s_next in env.get_states():
                    state_sum += env.state_transitions[s, action, s_next] * v[s_next]

                new_v += probability * (env.rewards[s, action] + env.gamma * state_sum)

            delta = max(delta, np.abs(old_v - new_v))
            v[s] = new_v
    return v


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
