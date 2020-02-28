import numpy as np

from .environment import Environment

from typing import Tuple

from gym import spaces


class Cartpole(Environment):
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
        self._name = "Cartpole"
        self._action = None
        self._reward = 0
        self._isEnd = False

        self._gamma = 1.0
        ranges = np.zeros((4,2))
        ranges[0, :] = [-3, 3]
        ranges[1, :] = [-10., 10.]
        ranges[2, :] = [-np.pi / 12, np.pi / 12]
        ranges[3, :] = [-np.pi, np.pi]
        self.observation_space = spaces.Box(ranges[:, 0], ranges[:, 1], dtype=np.float64)
        self.action_space = spaces.Discrete(2)

        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: USE must use this variable

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array([self._x, self._v, self._theta, self._dtheta])

    @property
    def gamma(self) -> float:
        return self._gamma

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        F = 10 if action else -10  # force exerted by the agent
        h = self._dt  # step size (euler forward)

        x, v, theta, dtheta = state

        mass = self._mc + self._mp
        sinth = np.sin(theta)
        costh = np.cos(theta)

        num = self._g * sinth + costh * ((-F - self._mp * self._l * dtheta ** 2 * sinth) / mass)
        ddtheta = num / (self._l * (4 / 3 - (self._mp * costh ** 2) / mass))

        dv = (F + self._mp * self._l * (dtheta ** 2 * sinth - ddtheta * costh)) / mass


        # update by adding the derivatives (euler forward)
        theta += h * dtheta
        dtheta += h * ddtheta
        x += h * v
        v += h * dv

        return np.array([x, v, theta, dtheta])

    def R(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        returns a reward for the transition (state,action,next_state)
        """
        return 1.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        self._action = action
        state = np.array([self._x, self._v, self._theta, self._dtheta])
        next_state = self.nextState(state, action)
        self._x, self._v, self._theta, self._dtheta = next_state

        self._reward = self.R(state, action, next_state)
        self._t += self._dt
        self._isEnd = self.terminal()

        return self.state, self._reward, self.isEnd

    def reset(self):
        """
        resets the state of the environment to the initial configuration
        """
        self._x = 0.
        self._v = 0.
        self._theta = 0.
        self._dtheta = 0.
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
        if self._t > 20 - 1e-8:
            return True
        if np.abs(self._theta) > (np.pi / 12.0):
            return True
        if np.abs(self._x) >= 3:
            return True  # cart hits end of track
        return False


def bound(val, limit):
    if val < -limit: return -limit
    if val > limit: return limit
    return val
