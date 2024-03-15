import numpy as np
from gymnasium import Env
from typing import Callable, List, Tuple, Union
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
    Hot index of the one-hot vector is the cell index in the state-action table. Cell index is calculated from the distance of cell from top-left corner of the grid based on a left-to-right, top-to-bottom ordering.

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
        assert self.action_bounds[0] <= action < self.action_bounds[1], "Action must be within the bounds"
        assert np.all([bound[0] <= s < bound[1] for s, bound in zip(state, self.state_bounds)]), "State must be within the bounds"

        decomposed_state = [int((s - bound[0])) for s, bound in zip(state, self.state_bounds)]

        state_action = tuple(decomposed_state + [action])

        one_hot = np.zeros((self.n_features, 1))
        hot_index = self.table[state_action]
        one_hot[hot_index] = 1
        if self.include_bias:
            one_hot[-1] = 1
        return one_hot

    def __repr__(self) -> str:
        return f"Tabular (bias={self.include_bias})"

class TileFeaturizer(Featurizer):
    """
    Simple Featurizer creates a tile representation of (s, a) pairs based on the tile dimensions and offsets.
    Tiles map a possibly continuous state space to a discrete one. However, action space is assumed to be discrete.

    Args:
        state_bounds (List[Tuple[float, float]]): list of tuples representing the bounds of each dimension of the state space
        action_bounds (Tuple[int, int]): tuple representing the bounds of the action space
        tile_dims (List[float]): dimensions of the tile
        offset (List[float]): offset of the tile
        include_bias (bool): whether to include a bias term in the feature vector
    """
    def __init__(self, state_bounds: List[Tuple[float, float]], action_bounds: Tuple[int, int], tile_dims: Union [List[float], float], offset: Union[List[float], float] = None, include_bias: bool = True):
        super().__init__(include_bias)
        assert len(state_bounds) > 0, "State bounds must not be empty"
        assert isinstance(state_bounds, list) and np.all([len(bound) == 2 for bound in state_bounds]), "State bounds must be a list of tuples of length 2"
        assert len(action_bounds) == 2, "Action bounds must be a tuple of length 2"

        self.n_dims = len(state_bounds)
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.nA = action_bounds[1] - action_bounds[0]

        assert isinstance(tile_dims, list) or isinstance(tile_dims, float), "Tile dimensions must be a list or a float"
        if isinstance(tile_dims, float):
            tile_dims = [tile_dims] * self.n_dims
        assert len(tile_dims) == len(state_bounds), "Tile dimensions must have the same number of dimensions as the state bounds"
        assert np.all([0 < tile_dim for tile_dim in tile_dims]), "Tile dimensions must be positive"

        if offset is not None:
            assert isinstance(offset, list) or isinstance(offset, float) or offset is None, "Offset must be a list, a float or None"

            if isinstance(offset, float):
                offset = [offset] * self.n_dims
            assert len(offset) == len(state_bounds), "Offset must have the same number of dimensions as the state bounds"
            assert np.all([0 <= offset < tile_dim for offset, tile_dim in zip(offset, tile_dims)]), "Offset must be within the tile dimensions"
        else:
            offset = [0] * self.n_dims        

        self.tile_dims = tile_dims
        self.offsets = offset
        self.tiles_per_dim =[int((bound[1] - bound[0] + 1) / tile_dim) for bound, tile_dim in zip(self.state_bounds, self.tile_dims)]
        self.num_tiles = np.prod(self.tiles_per_dim)
    
    @property
    def n_features(self) -> int:
        return self.num_tiles + self.nA + int(self.include_bias)
    
    def featurize(self, state: tuple, action: int) -> np.ndarray:
        """ Turn state into a feature one-hot vector. Where the hot index of the one-hot vector is the tile index.
        Tile index is calculated from the distance of the state from the bottom-left corner of the grid (with self.state_bounds dims) based on a left-to-right, top-to-bottom ordering.

        Args:
            state (tuple): state of the environment
            action (int): action index
        
        Returns:
            tile (np.ndarray): one-hot vector of the tile index
        """
        assert len(state) == self.n_dims, "State must have the same number of dimensions as the state bounds"
        assert self.action_bounds[0] <= action < self.action_bounds[1], "Action must be within the bounds"
        assert np.all([bound[0] <= s < bound[1] for s, bound in zip(state, self.state_bounds)]), "State must be within the bounds"
        one_hot = np.zeros((self.n_features, 1))

        decomposed_state = [int((s - bound[0] + offset) / tile_dim) for s, bound, tile_dim, offset in zip(state, self.state_bounds, self.tile_dims, self.offsets)]
        tile_index = np.ravel_multi_index(decomposed_state, self.tiles_per_dim)

        # add tile encoding
        one_hot[tile_index] = 1

        # add action encoding
        one_hot[self.num_tiles + action] = 1

        # add bias term
        if self.include_bias:
            one_hot[-1] = 1
        return one_hot
    
    def __repr__(self) -> str:
        base = f"Tile {'x'.join([str(dim) for dim in self.tile_dims])}"
        if self.offsets:
            base += f" offset {'x'.join([str(offset) for offset in self.offsets])}"
        
        return f"{base} (bias={self.include_bias})"

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
    
class GridWorldTileFeaturizer(TileFeaturizer):
    """
    Simple Featurizer for the GridWorld environment.
    The featurizer divides the grid into tiles originating from the bottom-left corner of the grid (0, 0).

    Args:
        env (Env): GridWorld environment with n_rows and n_cols attributes (NOTE: assumes that state space is (n_rows, n_cols) and action space is (n_actions, ))
        tile_dims (List[float]): dimensions of the tile
        offset (List[float]): offset of the tile
        include_bias (bool): whether to include a bias term in the feature vector
    """
    def __init__(self, env: Env, tile_dims: Union [List[float], float], offset: Union[List[float], float] = None, include_bias: bool = True):
        assert hasattr(env.unwrapped, "n_rows"), "Environment must have a n_rows attribute"
        assert hasattr(env.unwrapped, "n_cols"), "Environment must have a n_cols attribute"
        self.n_rows = env.unwrapped.n_rows
        self.n_cols = env.unwrapped.n_cols
        super().__init__([(0, self.n_rows), (0, self.n_cols)], (0, env.action_space.n), tile_dims, offset, include_bias)
        