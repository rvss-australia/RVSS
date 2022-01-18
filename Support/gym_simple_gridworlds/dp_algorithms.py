import numpy as np

def policy_evaluation(grid_env, policy, threshold=0.00001):
    
    """
    This function computes the value function for a policy pi in a given environment grid_env.
    
    :param grid_env (GridEnv): MDP environment
    :param policy (dict - stochastic form): Policy being evaluated
    :param threshold (float): Convergence threshold used to stop main computation loop
    :return: (dict) State-values for all non-terminal states
    """
        
    # Obtain list of all states in environment
    v = {s: 0.0 for s in grid_env.get_states()}
    theta = threshold
    delta = 1000

    while delta > theta:
        delta = 0.0
        # For all states
        for s in v.keys():

            old_v = v[s]
            new_v = 0

            # For all actions
            for a, probability_a in policy[s].items():
                discounted_value = 0

                # For all states that are reachable from s with action a
                for s_next in grid_env.get_states():
                    discounted_value += grid_env.state_transitions[s, a, s_next] * v[s_next]

                # Compute new value for state s
                new_v += probability_a*(grid_env.rewards[s, a] + grid_env.gamma*discounted_value)

            v[s] = new_v
            delta = max(delta, np.abs(old_v - new_v))
        
    return v


def value_iteration(grid_env, threshold=0.00001):
    """
    This function iteratively computes optimal state-value function for a given environment grid_env. 
    It returns the optimal state-value function and its associated optimal policy
    
    :param grid_env (GridEnv): MDP environment
    :param threshoold (float): Convergence threshold
    :param plot (bool): Bool argument indicating if value function and policy should be displayed 
    :return: (tuple) Optimal state-value funciton (dict) and deterministic optimal policy (matrix)
    """
    
    #1. Get list of states in environment
    states = grid_env.get_states()
    
    #2. Initialize v function
    v = {s: 0.0 for s in grid_env.get_states()}
    
    #3. Set convergence threshold and error variable
    theta = threshold
    delta = 1000
    
    #4. Update v(s) until convergence
    while delta > theta:
        
        delta = 0
        for s in states:
            old_v = v[s]
            v[s] = np.max(one_step_lookahead(grid_env, s, v))
            delta = max(delta, abs(old_v - v[s]))
            
    #5. Compute deterministic policy given v(s)
    temp_policy = np.ones(grid_env.grid.shape) * -1
    optimal_policy = update_policy(grid_env, temp_policy, v)
    
            
    return v, optimal_policy   


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
        action_values = one_step_lookahead(grid_env, s, value_function)
        
        # Find (row, col) coordinates of cell with index s
        row, col = np.argwhere(grid_env.grid == s)[0]
        
        cur_policy[row, col] = np.argmax(action_values)
        
    return cur_policy


def one_step_lookahead(grid_env, state, value_function):
    """
     Compute the action-value function for a state $s$ given the state-value function $v$.
     
     :param grid_env (GridEnv): MDP environment
     :param state (int): state for which we are looking one action ahead
     :param value_function (dict): state-value function associated to a given policy py
     
     :return: (np.array) Action-value functions of actions available at state s
    """
    action_values = []
    
    for action in grid_env.get_actions():
        discounted_value = 0
        for s_next in grid_env.get_states():
             discounted_value += grid_env.state_transitions[state, action, s_next] * value_function[s_next]
        
        q_a = grid_env.rewards[state, action] + grid_env.gamma * discounted_value
        action_values.append(q_a)
    
    return np.array(action_values)


def policy_evaluation(grid_env, policy, threshold=0.00001):
    
    """
    This function computes the value function for a policy pi in a given environment grid_env.
    
    :param grid_env (GridEnv): MDP environment
    :param policy (dict - stochastic form): Policy being evaluated
    :return: (dict) State-values for all non-terminal states
    """
        
    # Obtain list of all states in environment
    v = {s: 0.0 for s in grid_env.get_states()}
    theta = threshold
    delta = 1000

    while delta > theta:
        delta = 0.0
        # For all states
        for s in v.keys():

            old_v = v[s]
            new_v = 0

            # For all actions
            for a, probability_a in policy[s].items():
                discounted_value = 0

                # For all states that are reachable from s with action a
                for s_next in grid_env.get_states():
                    # TODO: Complete the computation of the discounted value of successor states
                    discounted_value += grid_env.state_transitions[s, a, s_next] * v[s_next]

                # Compute new value for state s
                new_v += probability_a*(grid_env.rewards[s, a] + grid_env.gamma*discounted_value)

            v[s] = new_v
            delta = max(delta, np.abs(old_v - new_v))

        
    return v