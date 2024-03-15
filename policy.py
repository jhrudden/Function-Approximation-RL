import numpy as np
from typing import Callable
from gymnasium import Env

def argmax(q_values: np.ndarray, break_ties_randomly: bool = False) -> int:
    if break_ties_randomly:
        return np.random.choice(np.flatnonzero(q_values == q_values.max()))
    else:
        return np.argmax(q_values)

def create_epsilon_greedy_policy(env: Env, Q: Callable, epsilon: float, break_ties_randomly=False) -> np.ndarray:
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        env: environment
        Q: Q-function, callable function which takes a state and action and returns the Q-value
        epsilon: probability of selecting a random action
        break_ties_randomly: whether to break ties randomly when selecting the action with the highest Q-value

        
    
    Returns:
        policy: epsilon-greedy policy, a function that takes an observation and returns an action and policy distribution 
    """
    def policy_fn(observation):
        A = np.arange(env.action_space.n)
        estimates = [Q(observation, a).item() for a in A]
        estimates = np.array(estimates)
        policy = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
        best_action = argmax(estimates, break_ties_randomly=break_ties_randomly)
        policy[best_action] += 1 - epsilon
        action = np.random.choice(A, p=policy)
        return action, policy
    return policy_fn