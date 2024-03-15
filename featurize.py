import numpy as np
from gymnasium import Env
from typing import Callable, List, Tuple
# import deepcopy from copy
from copy import deepcopy

from env import get_four_rooms_env

class Featurizer:
    """
    Featurizer class
    """
    def __init__(self, include_bias: bool = True):
        self.include_bias = include_bias

    @property
    def n_features(self) -> int:
        """
        Number of features
        """
        raise NotImplementedError

    def featurize(self, state: tuple, action: int) -> np.ndarray:
        """
        Featurize the state

        Args:
            state (tuple): state of the environment
        
        Returns:
            features (np.ndarray): features of the state
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bias={self.include_bias})"

class CompositionFeaturizer(Featurizer):
    """
    FeaturizerComposer class
    """
    def __init__(self, featurizers: list, include_bias: bool = True):
        # make a deep copy of the featurizers
        self.featurizers = [deepcopy(f) for f in featurizers]

        if include_bias:
            self.featurizers[-1].include_bias = True
            
        for f in self.featurizers[:-1]:
            f.include_bias = False

    @property
    def n_features(self) -> int:
        """
        Number of features
        """
        return sum([f.n_features for f in self.featurizers])
        
    def featurize(self, state: tuple, action: int) -> np.ndarray:
        """
        Featurize the state

        Args:
            state (tuple): state of the environment
        
        Returns:
            features (np.ndarray): features of the state
        """
        return np.vstack([f.featurize(state, action) for f in self.featurizers])
    
    def __repr__(self) -> str:
        return f"Comp({', '.join([str(f) for f in self.featurizers])})"


class FunctionalFeaturizer:
    def __init__(self, func: Callable[[tuple], float], label:str, include_bias: bool = False):
        self.func = func
        self.label = label

    @property
    def n_features(self) -> int:
        return 1

    def featurize(self, state: tuple, action: int) -> np.ndarray:
        return np.array([self.func(state, action)])

    def __repr__(self) -> str:
        return f"{self.label}"

class StateActionFeaturizer(Featurizer):
    def __init__(self, include_bias: bool = True):
        super().__init__(include_bias)
    
    @property
    def n_features(self) -> int:
        return 3 + int(self.include_bias) # col, row, action, bias (optional)
    
    def featurize(self, state: tuple, action: int) -> np.ndarray:
        col, row = state
        vec = np.zeros((self.n_features, 1))
        vec[0] = col
        vec[1] = row
        vec[2] = action
        if self.include_bias:
            vec[-1] = 1
        return vec

class TabularFeaturizer(Featurizer):
    """
    Simple Featurizer creates a tabular representation of (s, a) pairs. 

    Actual featurization is done by the featurize method which returns a one-hot vector with shape (n_cells, 1) + (1, ) if include_bias is True.
    Hot index of the one-hot vector is the cell index in the state-action table.

    NOTE: A big assumption made here is that both state and action are discrete and included in the bounds. E.g for 11x11 grid with 4 actions, the bounds would be [(0, 11), (0, 11), (0,4)].
    
    Args:
        state_bounds (List[Tuple[float, float]]): list of tuples representing the bounds of each dimension of the state space
        action_bounds (Tuple[float, float]): tuple representing the bounds of the action space
        include_bias (bool): whether to include a bias term in the feature vector
    """
    def __init__(self, state_bounds: List[Tuple[float, float]], action_bounds: Tuple[float, float], include_bias: bool = True):
        super().__init__(include_bias)

        assert len(state_bounds) > 0, "State bounds must not be empty"
        assert isinstance(state_bounds, list) and np.all([len(bound) == 2 for bound in state_bounds]), "State bounds must be a list of tuples of length 2"
        assert len(action_bounds) == 2, "Action bounds must be a tuple of length 2"

        self.bounds = state_bounds + [action_bounds]
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.table_shape = [int(bound[1] - bound[0]) for bound in self.bounds]
        self.n_cells = np.prod(self.table_shape)
        self.table = np.arange(self.n_cells).reshape(self.table_shape)
    
    @property
    def n_features(self) -> int:
        return self.n_cells + int(self.include_bias)
    
    def featurize(self, state: tuple, action: int) -> np.ndarray:
        """ Turn state into a feature one-hot vector. Where the hot index of the one-hot vector is the cell index in state-action table.

        Args:
            state (tuple): state of the environment
        
        Returns:
            tile (np.ndarray): one-hot vector of the tile index
        """
        assert len(state) == len(self.state_bounds), "State must have the same number of dimensions as the state bounds"
        print(self.action_bounds)
        assert self.action_bounds[0] <= action < self.action_bounds[1], "Action must be within the bounds"
        assert np.all([bound[0] <= s < bound[1] for s, bound in zip(state, self.state_bounds)]), "State must be within the bounds"

        decomposed_state = [int((s - bound[0])) for s, bound in zip(state, self.state_bounds)]

        state_action = tuple(decomposed_state + [action])

        one_hot = np.zeros((self.n_features, 1))
        hot_index = self.table[state_action]
        print(f'hot_index: {hot_index} for state-action pair: {state_action}')
        one_hot[hot_index] = 1
        if self.include_bias:
            one_hot[-1] = 1
        return one_hot

    def __repr__(self) -> str:
        return f"Tabular (bias={self.include_bias})"



class GridWorldTabularFeaturizer(TabularFeaturizer):
    """
    Simple Featurizer for the GridWorld environment.
    The featurizer divides the grid into tiles originating from the bottom-left corner of the grid (0, 0).

    Args:
        env (Env): GridWorld environment with n_rows and n_cols attributes (NOTE: assumes that state space is (n_rows, n_cols) and action space is (n_actions, ))
        include_bias (bool): whether to include a bias term in the feature vector
    """
    def __init__(self, env: Env, include_bias: bool = True):
        assert hasattr(env.unwrapped, "n_rows"), "Environment must have a n_rows attribute"
        assert hasattr(env.unwrapped, "n_cols"), "Environment must have a n_cols attribute"
        self.n_rows = env.unwrapped.n_rows
        self.n_cols = env.unwrapped.n_cols
        super().__init__([(0, self.n_rows), (0, self.n_cols)], (0, env.action_space.n), include_bias)

class GridWorldTileFeaturizer(Featurizer):
    """
    Simple Featurizer for the GridWorld environment.
    The featurizer divides the grid into tiles originating from the bottom-left corner of the grid (0, 0).

    Args:
        n_rows (int): number of rows in the grid
        n_cols (int): number of columns in the grid
        tile_width (float): width of the tile
        tile_height (float): height of the tile
        offset_x (float): offset in the x direction
        offset_y (float): offset in the y direction
    """
    def __init__(self, env: Env, tile_width: float, tile_height: float, offset_x: float = 0, offset_y: float = 0, include_bias: bool = True):
        super().__init__(include_bias)
        assert hasattr(env.unwrapped, "n_rows"), "Environment must have a n_rows attribute"
        assert hasattr(env.unwrapped, "n_cols"), "Environment must have a n_cols attribute"
        self.n_rows = env.unwrapped.n_rows
        self.n_cols = env.unwrapped.n_cols


        assert self.n_rows > 0, "Number of rows must be positive"
        assert self.n_cols > 0, "Number of columns must be positive"
        assert tile_width > 0, "Tile width must be positive"
        assert tile_height > 0, "Tile height must be positive"
        assert offset_x <= tile_width and offset_x >= 0, "Offset x must be less than or equal to tile width and greater than or equal to 0"
        assert offset_y <= tile_height and offset_y >= 0, "Offset y must be less than or equal to tile height and greater than or equal to 0"
        
        self.nA = env.action_space.n
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.num_tiles = int((self.n_rows + 1) / tile_height) * int((self.n_cols + 1) / tile_width)

    @property
    def n_features(self) -> int:
        return self.num_tiles + self.nA + int(self.include_bias)
        
    def featurize(self, state: tuple, action: int) -> list:
        """ Turn state into a feature one-hot vector. Where the hot index of the one-hot vector is the tile index.

        Args:
            state (tuple): state of the environment
        
        Returns:
            tile (list): one-hot vector of the tile index
        """
        col, row = state
        assert 0 <= row < self.n_rows, f"Row index {row} is out of bounds"
        assert 0 <= col < self.n_cols, f"Column index {col} is out of bounds"

        one_hot = np.zeros((self.n_features, 1))
        tile_index = int((row + self.offset_y) / self.tile_height) * int(np.ceil(self.n_cols / self.tile_width))  + int((col + self.offset_x) / self.tile_width)
        one_hot[tile_index] = 1
        one_hot[self.num_tiles + action] = 1

        if self.include_bias:
            one_hot[-1] = 1
        return one_hot
    
    def __repr__(self) -> str:
        base = f"Tile {self.tile_width}x{self.tile_height}"
        if self.offset_x or self.offset_y:
            base += f" offset {self.offset_x}x{self.offset_y}"
        
        return f"{base} (bias={self.include_bias})"
    
