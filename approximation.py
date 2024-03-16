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
    check_points: set = {},
    mean_td: bool = False
) -> np.ndarray:
    """Implement the semi-gradient SARSA algorithm.

    Args:
        env: environment
        num_episodes: number of episodes
        alpha: step size
        epsilon: probability of exploration
        gamma: discount factor
        featurizer: GridWorldTileFeaturizer
        check_points: set of episodes to save the learned weight vector
        mean_td: whether to use mean TD error as stopping criterion
    
    Returns:
        w: learned weight vector, shape (n_features, 1)
    """
    nA = env.action_space.n
    w = np.zeros((featurizer.n_features, 1))
    q = lambda s, a: q_predict(featurizer.featurize(s, a), w)
    policy = create_epsilon_greedy_policy(env, q, epsilon, break_ties_randomly=True)

    steps_per_episode = np.zeros(num_episodes)

    save_weights = None
    if check_points is not None:
        save_weights = np.zeros((len(check_points), w.shape[0]))
        saved = 0
    else:
        check_points = set()

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
            q_next = q_predict(x_next, w)

            td_error = reward + gamma * q_next - q_x
            if mean_td:
                w -= alpha * td_error * (gamma * q_gradient(x_next, w) - grad)
            else:
                w += alpha * td_error * grad
            state = next_state
            action = next_action

        if (e_ndx + 1) in check_points:
            save_weights[saved] = w.flatten()
            saved += 1

        steps_per_episode[e_ndx] = t
    
    return w, steps_per_episode, save_weights



    
