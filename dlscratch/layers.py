"""
Layer class sends the input forward through a series of Operations and keeps a record
of the values inovlved. 
"""
import numpy as np
from numpy import ndarray
from typing import List
from operations import ParamOperation, Operation
from spec_ops import *

class Layer(object): 
    """A layer of neurons in a neural network"""
    def __init__(self, neurons: int):
        """
        The number of neurons. Roughly corresponds to
        the "width" of the layer.
        """
        self._neurons = neurons
        self._first = True
        self._params: List[ndarray] = []
        self._param_grads: List[ndarray] = []
        self._operations: List[Operation] = []
        self._input = None
        self._output = None
        self._seed = None
        
    def setup_layer(self, input_: ndarray) -> None: # changed from num_int
        """
        The setup_layer function must be implemented for
        each layer.
        """
        raise NotImplementedError()
    
    def forwrad(self, input_: ndarray) -> ndarray:
        """
        Passes input forward through a series of operations
        """
        if self._first:
            self.setup_layer(input_)
            self._first = False
        
        self._input = input_
        
        for operation in self._operations:
            input_ = operation.forward(input_)
        
        self._output = input_
        
        return self._output
    
    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Passes output_grad backward tghrough a series of operations.
        Checks appropriate shapes.
        """
        assert self._output.shape == output_grad
        
        for operation in reversed(self._operations):
            output_grad = operation.backward(output_grad)
            
            input_grad = output_grad
            self.get_param_grads()
            
            return input_grad
        
    def get_param_grads(self) -> ndarray:
        """
        Extracts the param_grads from a layer's operations.
        """
        for operation in self._operations:
            if issubclass(operation.__class__, ParamOperation):
                self._param_grads.append(operation._param_grad) 
                
    def get_params(self) -> ndarray:
        """
        Extracts the params from a layer's operations.
        """
        for operation in self._operations:
            if issubclass(operation.__class__, ParamOperation):
                self._params.append(operation._param) 
               
    class Dense(Layer):
        """A fully connectred layer that inherits from Layer"""
        def __init__(self, neurons: int,
                     activation: Operation = Sigmoid()) -> None:
            """
            Requires an activation function upon intitialization
            """
            super().__init__(neurons)
            self._activation = activation
            
        def setup_layer(self, input_: ndarray) -> None:
            """
            Defines the operations of a fully connected layer
            """
            if self._seed:
                np.random.seed(self._seed) # brought in from NN?
                
            self._params = []
            
            # weights
            self._params.append(np.random.randn(input_.shape[1],
                                                self._neurons))
            # bias
            self._params.append(np.random.randn(1, self._neurons))
            
            self._operations = [WeightMultiply(self._params[0]),
                                BiasAdd(self._params[1]),
                                self._activation]
            
            return None