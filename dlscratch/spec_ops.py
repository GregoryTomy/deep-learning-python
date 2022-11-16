"""
Defines the specific operations used in the neural network.
"""
import numpy as np
from numpy import ndarray
from typing import List
from operations import ParamOperation, Operation

class WeightMultiply(ParamOperation):
    """
    Weight multiplication operation for a neural network
    """
    def __init__(self, W:ndarray):
        """
        Initialize Operatiowith param = W.
        """
        super().__init__(W)
    
    def get_output(self) -> ndarray:
        """
        Compute output
        """
        return self._input @ self._param
    
    def get_input_grad(self, output_grad: ndarray) -> ndarray:
        """
        Compute input gradient
        """
        return output_grad @ self._param.T
    
    def get_param_grad(self, output_grad: ndarray) -> ndarray:
        """Compute parameter gradient"""
        return self._input.T @ output_grad
        
class BiasAdd(ParamOperation):
    """
    Computes bias addition
    """
    def __init__(self, B: ndarray):
        """
        Initialize Operation and check appropriate shape
        """
        assert B.shape[0] == 1
        super().__init__(B)
    
    def get_output(self) -> ndarray:
        """
        Compute output
        """
        return self._input + self._param
    
    def get_input_grad(self, output_grad: ndarray) -> ndarray:
        """Compute input iradient"""
        return np.ones_like(self._input) * output_grad
    
    def get_param_grad(self, output_grad: ndarray) -> ndarray:
        """Compute parameter gradient"""
        param_grad = np.ones_like(self._param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
    
class Sigmoid(Operation):
    """Sigmoid activation function"""
    def __init__(self) -> None:
        super().__init__()
    
    def get_output(self) -> ndarray:
        """Compute output"""
        return 1.0 / (1.0 + np.exp(-1.0 * self._input))
    
    def get_input_grad(self, output_grad: ndarray) -> ndarray:
        """Compute input gradient"""
        sigmoid_backward = self._output * (1 - self._output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
