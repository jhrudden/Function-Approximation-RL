import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.registration import register
from enum import IntEnum
from typing import Tuple, Optional, List
import numpy as np
from numpy import ndarray

def register_env(id: str, entry_point: str, max_episode_steps: Optional[int] = None):
    """Register custom gym environment so that we can use `gym.make()`
    Note: the max_episode_steps option controls the time limit of the environment.
    You can remove the argument to make FourRooms run without a timeout.
    """
    register(id=id, entry_point=entry_point, max_episode_steps=max_episode_steps)

def get_four_rooms_env(goal_pos=(10, 10)):
    """
    Get the FourRooms environment
    Args:
        goal_pos (Tuple[int, int]): goal position
    Returns:
        env (FourRoomsEnv): FourRooms environment
    """
    try:
        spec = gym.spec('FourRooms-v0')
    except:
        register_env("FourRooms-v0", "env:FourRoomsEnv", max_episode_steps=459)
    finally:
        return gym.make('FourRooms-v0', goal_pos=goal_pos)


class FourRoomAction(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

def actions_to_dxdy(action: FourRoomAction) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        FourRoomAction.LEFT: (-1, 0),
        FourRoomAction.DOWN: (0, -1),
        FourRoomAction.RIGHT: (1, 0),
        FourRoomAction.UP: (0, 1),
    }
    return mapping[action]

def perpendicular_actions(action: FourRoomAction) -> List[FourRoomAction]:
    """
    Helper function to get the perpendicular actions to the given action
    Args:
        action (Action): taken action
    Returns:
        perpendicular_actions (List[Action]): Perpendicular actions to the given action
    """
    mapping = {
        FourRoomAction.LEFT: [FourRoomAction.DOWN, FourRoomAction.UP],
        FourRoomAction.DOWN: [FourRoomAction.LEFT, FourRoomAction.RIGHT],
        FourRoomAction.RIGHT: [FourRoomAction.DOWN, FourRoomAction.UP],
        FourRoomAction.UP: [FourRoomAction.LEFT, FourRoomAction.RIGHT],
    }
    return mapping[action]


class FourRoomsEnv(Env):
    """Four Rooms gym environment.

    This is a minimal example of how to create a custom gym environment. By conforming to the Gym API, you can use the same `generate_episode()` function for both Blackjack and Four Rooms envs.
    """

    def __init__(self, goal_pos=(10, 10)) -> None:
        super().__init__()
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.action_space = spaces.Discrete(len(FourRoomAction))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos

        return (self.agent_pos, {})

    def step(self, action: FourRoomAction) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 for more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        EPSILON = 0.1
        
        random_val = np.random.random()
        if random_val < EPSILON:
            action_taken = np.random.choice(perpendicular_actions(action))
        else:
            action_taken = action

        dx, dy = actions_to_dxdy(action_taken)
        next_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        if self._valid_position(next_pos):
            self.agent_pos = next_pos

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0

        return self.agent_pos, reward, done, False, {}
    
    def _valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Helper function to check if a position is valid
        Args:
            pos (Tuple[int, int]): position to check
        Returns:
            valid (bool): True if position is valid, False otherwise
        """
        return pos not in self.walls and 0 <= pos[0] < self.cols and 0 <= pos[1] < self.rows