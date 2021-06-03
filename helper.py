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

    obs_dim = None
    act_dim = None

    def __init__(self, sigma=1, seed=0):
        """
        bottom text
        """
        np.random.seed(seed)

        self.sigma = sigma
        self.state = np.random.random(obs_dim)

    @abstractmethod
    def forward(self, input, state=None):
        """
        This method implements the forward dynamics of the system.
        """
        pass

    def _state(self):
        return self.state

class SampleModel(Model):
    obs_dim = 1
    act_dim = 1

    def forward(self, input, state=None):
        m = 3

        state_1 =  self.state * m + input

        return state_1 + np.random.normal() 
