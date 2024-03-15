from policy import create_epsilon_greedy_policy
from featurize import Featurizer

import numpy as np
import gymnasium as gym

def q_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Assuming Q is approximated by a linear function, predict Q for given state-action pair x and learned weight vector w.

    Args:
        x: state-action pair, shape (n_features, 1)
        w: weight vector, shape (n_features, 1)

    Returns:
        Q: predicted Q value for given state-action pair
    """
    return np.dot(x.T, w)

def q_gradient(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the gradient of the Q function with respect to the weight vector w.

    Args:
        x: state-action pair, shape (n_features, 1)
        w: weight vector, shape (n_features, 1)

    Returns:
        grad: gradient of the Q function with respect to the weight vector w, shape (n_features, 1)
    """
    return x

def semi_gradient_sarsa(
    env: gym.Env,
    num_episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    featurizer: Featurizer,
) -> np.ndarray:
    """Implement the semi-gradient SARSA algorithm.

    Args:
        env: environment
        num_episodes: number of episodes
        alpha: step size
        epsilon: probability of exploration
        gamma: discount factor
        featurizer: GridWorldTileFeaturizer
    
    Returns:
        w: learned weight vector, shape (n_features, 1)
    """
    nA = env.action_space.n
    w = np.zeros((featurizer.n_features, 1))
    q = lambda s, a: q_predict(featurizer.featurize(s, a), w)
    policy = create_epsilon_greedy_policy(env, q, epsilon, break_ties_randomly=True)

    steps_per_episode = np.zeros(num_episodes)


    for e_ndx in range(num_episodes):
        state, _ = env.reset()
        action, _ = policy(state)
        done = False
        truncated = False
        t = 0
        while not done and not truncated:
            next_state, reward, done, truncated, _ = env.step(action)
            x = featurizer.featurize(state, action)
            t += 1
            grad = q_gradient(x, w)
            q_x = q_predict(x, w)
            if done or truncated:
                w += alpha * (reward - q_x) * grad
                break
            
            next_action, _ = policy(next_state)
            x_next = featurizer.featurize(next_state, next_action)
            w += alpha * (reward + gamma * q_predict(x_next, w) - q_x) * grad
            state = next_state
            action = next_action

        steps_per_episode[e_ndx] = t
    
    return w, steps_per_episode



    
