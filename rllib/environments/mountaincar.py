import numpy as np

from .environment import Environment

from typing import Tuple

from gym import spaces


class Mountaincar(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    Barrowed from Phil Thomas's RL course Fall 2019. Written by Blossom Metevier and Scott Jordan
    """

    def __init__(self):
        self._name = "MountainCar"
        self._action = None
        self._reward = 0
        self._isEnd = False

        self._gamma = 1.0
        ranges = np.zeros((2,2))
        ranges[0, :] = [-1.2, 0.5]
        ranges[1, :] = [-0.07, 0.07]
        self.observation_space = spaces.Box(ranges[:, 0], ranges[:, 1], dtype=np.float64)
        self.action_space = spaces.Discrete(3)

        self._x = 0.  # horizontal position of car
        self._v = 0.  # horizontal velocity of the car

        # dynamics
        self._g = 0.0025  # gravity coeff
        self._ucoef = 0.001  # gravity coeff
        self._h = 3.0  # cosine frequency parameter
        self._t = 0.0  # total time elapsed  NOTE: USE must use this variable

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array([self._x, self._v])

    @property
    def gamma(self) -> float:
        return self._gamma

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        u = float(action - 2)
        x, v = state
        v += self._ucoef * u - self._g * np.cos(self._h * x)
        v = np.clip(v, self.observation_space.low[1], self.observation_space.high[1])
        x += v

        return np.array([x, v])

    def R(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        returns a reward for the transition (state,action,next_state)
        """
        return -1.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        self._action = action
        state = np.array([self._x, self._v])
        next_state = self.nextState(state, action)
        self._x, self._v = next_state

        self._reward = self.R(state, action, next_state)
        self._t += 1.0
        self._isEnd = self.terminal()

        return self.state, self._reward, self.isEnd

    def reset(self):
        """
        resets the state of the environment to the initial configuration
        """
        self._x = -0.5
        self._v = 0.
        self._isEnd = False
        self._t = 0.0
        self._action = None

    def terminal(self) -> bool:
        """
        terminates the episode if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        if self._t >= 5000:
            return True
        if self._x >= 0.05:
            return True
        return False
