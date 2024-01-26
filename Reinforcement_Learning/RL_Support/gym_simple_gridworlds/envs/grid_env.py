import gymnasium as gym
import numpy as np
import copy
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from RL_Support.gym_simple_gridworlds.envs.grid_2dplot import plot_grid_world, get_state_to_plot, plot_value_function
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class GridEnv(gym.Env):

    """Implementation of a Grid World MDP

    This class has the following attributes

    - grid: Grid represented as Numpy 2D array
    - gamma: discount factor
    - terminal_states: Grid coordinates of terminal states
    - obstacles: Grid coordinates of obstacles
    - action_space: Discrete action space
    - observation_space: Discrete state space
    - noise: Probability of an noisy actions. A noisy action will result in a noise% chance of a change in direction
    - rewards: Reward function (Matrix with dimensions number states x number of actions)
    - immediate_rewards: Immediate reward received at each state. These values are used to define the reward function
    - state_transitions: State transition function (Matrix with dimensions number states x number of actions x number states)
    - cur_state: Current location of agent in the grid

    """

    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)
    # 0: up, 1: down, 2: left, 3: right
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, height=3,
                 width=4,
                 terminal_states=None,
                 reward_terminal_states=None,
                 obstacles=None,
                 living_reward=-0.04,
                 gamma=0.9,
                 noise=0.0):

        """
        Initialize a new Grid World environment
        :param height (int): Number of rows in the grid
        :param width (int): Number of columns in the grid
        :param terminal_states (list of tuples): Grid coordinates (row, col) of terminal states
        :param reward_terminal_states (list of floats): Reward associated with each terminal state
        :param obstacles (list of tuples): Grid coordinates (row, col) of obstacles in grid
        :param living_reward (float): Immediate reward agent receives at each time step
        :param gamma (float): Discount factor
        :param noise (float): Probability of a change of direction when an action is taken (e.g., going left/right
                              when agent decides to move up/down)
        """

        if terminal_states is None:
            terminal_states = [(1, 3), (0, 3)]

        if reward_terminal_states is None:
            reward_terminal_states = [-1.0, 1.0]

        if obstacles is None:
            obstacles = [(1, 1)]

        # Check that number of rewards match number of terminal states
        assert len(terminal_states) == len(reward_terminal_states)

        self.gamma = gamma
        self.terminal_states = terminal_states
        self.obstacles = obstacles

        # Set of valid actions
        self.action_space = spaces.Discrete(4)

        # Set of states
        self.observation_space = spaces.Discrete((height * width) - len(self.obstacles))

        self.grid = np.zeros((height, width))
        self.noise = noise
        self.np_random = None

        idx = 0
        for i in range(0, height):
            for j in range(width):
                if self.is_obstacle(i, j):
                    self.grid[i, j] = np.nan
                    continue
                self.grid[i, j] = int(idx)
                idx += 1

        # Define transition matrix
        self.state_transitions = np.zeros((self.observation_space.n, self.action_space.n, self.observation_space.n))

        for i in range(height):
            for j in range(width):
                s = self.grid[i, j]
                if np.isnan(s):
                    continue

                for a in self.ACTIONS.keys():
                    self.state_transitions[int(s), a, :] = self.__calculate_transition_probability__(int(s), a)

        # Define immediate rewards
        self.immediate_rewards = np.full(self.observation_space.n, living_reward)

        # Set rewards for final states
        for r, (x, y) in zip(reward_terminal_states, self.terminal_states):
            s = self.grid[x, y]
            self.immediate_rewards[int(s)] = r

        # Define reward function based on immediate rewards
        self.rewards = np.full((self.observation_space.n, self.action_space.n), 0.0)
        self.__compute_reward_function__()

        self.seed()
        self.cur_state = (2, 0)
        self.idx_cur_state = 7

    def __calculate_transition_probability__(self, state, action):
        """
        Determine probability of transition of any state given current state and the agent's action
        :param state: Current state (cell) in the grid
        :param action: Action to be applied at current state
        :return: Array of dimension self.observation_space telling probability of transitioning from current_state to
                 any other state given current action
        """
        x, y = np.argwhere(self.grid == state)[0]
        prob = np.zeros(self.observation_space.n)

        if self.is_terminal_state(x, y):
            prob[int(state)] = 1.0
            return prob

        possible_states = self.__get_reachable_states__(state)

        next_s = possible_states[action]
        prob[next_s] += 1.0 - self.noise

        if self.noise:
            if action in (0, 1):
                # If moving up or down, mdp transitions with noise/2.0 probability to left or right
                prob[possible_states[2]] += self.noise / 2.0
                prob[possible_states[3]] += self.noise / 2.0
            else:
                # If moving up or down, mdp transitions with noise/2.0 probability to up or down
                prob[possible_states[0]] += self.noise / 2.0
                prob[possible_states[1]] += self.noise / 2.0

        return prob

    def __get_reachable_states__(self, state):
        """
        Determine grid locations that are reachable from current state given the set of actions
        :param state: Current state (cell) in the grid
        :return: List with cell indices of states that are reachable from state s given action set
        """
        x, y = np.argwhere(self.grid == state)[0]
        states = {}
        height, width = self.grid.shape

        for a, displacement in self.ACTIONS.items():
            next_row = max(0, min(x + displacement[0], height - 1))
            next_col = max(0, min(y + displacement[1], width - 1))

            next_s = self.grid[next_row, next_col]

            if np.isnan(next_s):
                next_s = state

            states[a] = int(next_s)
        return states

    def is_terminal_state(self, x, y):
        """
        Determine whether a given location in the grid is a terminal state
        :param x: row index
        :param y: col index
        :return: Bool indicating if specified location is a terminal state
        """
        return (x, y) in self.terminal_states

    def is_obstacle(self, x, y):
        """
        Determine whether a given location in the grid is a terminal state
        :param x: row index
        :param y: col index
        :return: Bool indicating if specified location is a terminal state
        """
        return (x, y) in self.obstacles

    def step(self, action):
        """
        Run one time step of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        :param action: idx of action provided by the agent
        :return: tuple (observation, reward, done, info)

            observation (int): agent's observation of the current state
            reward (float) : amount of reward returned
            done (bool): whether the terminal state has been reached
            info (dict): state transition probability value
        """

        # Get transition probabilities given current state and agent's action
        prob = self.state_transitions[int(self.idx_cur_state), action]

        # Select state to transition to given transition probabilities
        next_state = np.random.choice(np.arange(self.observation_space.n), 1, p=prob)

        # Get (x,y) coordinates associated to that state
        next_x, next_y = np.argwhere(self.grid == next_state)[0]
        # Have we reached terminal state
        done = self.is_terminal_state(next_x, next_y)

        # Get reward
        reward = self.immediate_rewards[int(self.idx_cur_state)]

        if done:
            reward += self.immediate_rewards[int(next_state)]

        self.cur_state = (next_x, next_y)
        self.idx_cur_state = int(next_state)

        return self.idx_cur_state, reward, done, {'prob': prob[next_state]}

    def __select_init_state__(self):
        """
        Randomly select the initial state of the agent in the grid
        :return: Grid coordinates of initial state
        """
        candidate = np.random.randint(self.observation_space.n)
        x, y = np.argwhere(self.grid == candidate)[0]

        while self.is_terminal_state(x, y):
            candidate = np.random.randint(self.observation_space.n)
            x, y = np.argwhere(self.grid == candidate)[0]

        return (x,y), int(candidate)

    def reset(self):
        """Resets the environment to an initial state and returns an initial
            observation.
        :return Initial location in the grid
        """
        self.cur_state, self.idx_cur_state = self.__select_init_state__()
        return self.idx_cur_state

    def render(self, mode='human'):
        """
        Render the environment
        :param mode: Display mode (currently only human mode is supported)
        :return:
        """
        return plot_grid_world(self)

    def close(self):
        pass

    def seed(self, seed=1):
        """Sets the seed for this env's random number generator(s).
        :return: Returns the list of seeds used in this env's random number generators.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_states(self, exclude_terminal=False):
        """Get list of indices of all states
        :param exclude_terminal: Bool indicating whether terminal states should be include in list
        :return: Returns the list of indices of non-terminal states.
        """
        height, width = self.grid.shape
        states = []
        for i in range(height):
            for j in range(width):
                s = self.grid[i, j]
                if np.isnan(s):
                    continue

                if self.is_terminal_state(i, j) and exclude_terminal:
                    continue

                states.append(int(s))

        return states

    def get_actions(self):
        """Get list of actions.
        :return: Returns the list of indices of all actions.
        """
        return list(self.ACTIONS.keys())

    def __compute_reward_function__(self):
        """
        Determine reward function based on state transitions and immediate rewards
        """
        height, width = self.grid.shape
        for i in range(height):
            for j in range(width):
                s = self.grid[i, j]
                if np.isnan(s) or self.is_terminal_state(i, j):
                    continue

                for a in self.ACTIONS.keys():
                    p = self.state_transitions[int(s), a, :]

                    self.rewards[int(s), a] = np.dot(p, self.immediate_rewards)

    def get_q_values(self, value_function):
        """
        Given a state-value function, computes the corresponding action-value function
        ":param value_function: Dict with values of all non-terminal states
        :return dict of dict of floats: Value for each state-action pair
        """
        states = self.get_states(exclude_terminal=True)
        actions = np.arange(self.action_space.n)

        q_values = {s: {a: 0} for s in states for a in actions}

        for s in states:
            for a in actions:
                q_values[s][a] = self.rewards[s, a] + self.gamma * \
                                 np.max([value_function[n_s] * self.state_transitions[s, a, n_s] for n_s in states])

        return q_values
