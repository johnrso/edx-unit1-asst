from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    """
    A Model object is a function mapping states R^N and inputs R^N to states R^N.
    A model object is responsible for maintaining its own state variables.
    All Model objects must implement a Model.forward() method.
    This function is responsible for the mapping described above.
    See the forward method for more information.
    """

    def __init__(self,
                 act_dim,
                 obs_dim,
                 initial_state=None,
                 sigma=1,
                 seed=0):
        """
        bottom text
        """
        np.random.seed(seed)
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.sigma = sigma
        self._state = np.random.random(self.obs_dim) if initial_state is None else initial_state

    @abstractmethod
    def forward(self, input=None, state=None):
        """
        This method implements the forward dynamics of the system.
        """
        pass

    def state(self):
        return self._state

class SampleModel(Model):
    """
    A simple model that implements a noisy linear relationship.
    """

    def forward(self, input=None, state=None):
        m = 3

        if state is None:
            state_0 = self._state
        else:
            state_0 = state

        state_1 = state_0 * m
        state_1 = state_1 + np.random.normal(0, self.sigma, self.obs_dim)

        if state is None:
            self._state = state_1
        return state_0, state_1

class Complex1DModel(Model):
    """
    A complex model with non-linear relationships
    """

    def forward(self, input=None, state=None):
        if state is None:
            state_0 = self._state
        else:
            state_0 = state

        state_1 = 5 * state_0 ** 3 - 3 * state_0 ** 2 + state_0 + 20
        state_1 = state_1 + np.random.normal(0, self.sigma, self.obs_dim)

        if state is None:
            self._state = state_1
        return state_0, state_1

def MSE_loss(x, y):
    return np.mean((x - y).T @ (x - y))

class Motion2DModel(Model):
    A = np.array([[1, .037, 0, 0],
                  [0, .63, 0, 0],
                  [0, 0, 1, .037],
                  [0, 0, 0, .63]])

    b = np.array([[0, 0],
                  [.24, 0],
                  [0, 0],
                  [0, .24]])

    def forward(self, input=None, state=None):
        if state is None:
            state_0 = self._state
        else:
            state_0 = state

        if input is None:
            input = np.zeros(self.act_dim)

        state_1 = self.A @ state_0
        state_1 = state_1 + (self.b @ input)
        state_1 = state_1 + np.random.normal(0, self.sigma, self.obs_dim)

        if state is None:
            self._state = state_1
        return state_0, state_1
