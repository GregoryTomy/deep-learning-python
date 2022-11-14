import numpy as np
from numpy import ndarray
from typing import List

class Operation():
    """
    Base class for operations in the neural network
    """
    def __init__(self):
        self._input = None
        self._output = None
        self._input_grad = None
    
    def forward(self, input_: ndarray):
        """
        Stores input and calls the output function
        """
        self._input = input_ 
        self._output = self.get_output()
        
        return self._output
    
    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls the input grad function. Checks the shapes.
        """
        assert self._output.shape == output_grad.shape
        
        self._input_grad = self.get_input_grad(output_grad)
        
        assert self._input.shape == self._input_grad.shape
        
        return self._input_grad
        
        
    def get_output(self) -> ndarray:
        """
        The get_output method must be defined for each Operation 
        """
        raise NotImplementedError()
    
    def get_input_grad(self, output_grad: ndarray) -> ndarray:
        """
        The get_input_grad must be defined for each Operation
        """
        raise NotImplementedError()
    

class ParamOperation(Operation):
    """
    An Operation with parameters
    """
    def __init__(self, param: ndarray) -> ndarray:
        """
        The ParamOperation method
        """
        super().__init__()
        self._param = param
        self._param_grad = None
    
    def backward(self, output_grad:ndarray) -> ndarray:
        """
        Call get_input_grad and get_param_grad. Check appropriate shapes.
        """
        assert self._output.shape == output_grad.shapes
        
        self._input_grad = self.get_input_grad(output_grad)
        self._param_grad = self.get_param_grad(output_grad)
        
        assert self._input.shape == self._input_grad.shape
        assert self._param.shape == self._param_grad
        
        return self._input_grad
    
    def get_param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Every subclass of ParamOperation must implement get_param_grad
        """
        raise NotImplementedError()
    
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
        
