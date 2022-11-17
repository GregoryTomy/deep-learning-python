import numpy as np
from numpy import ndarray

class Loss(object):
    """The loss of a neural network"""
    
    def __init__(self):
        self._prediction = None
        self._target = None
        self._input_grad = None
    
    def forward(self, prediction: ndarray, 
                target: ndarray) -> float:
        """ 
        Computes the actual loss value.
        """
        assert prediction.shape == target.shape

        self._prediction = prediction
        self._target = target
        
        loss_value = self.get_output()
        
        return loss_value
    
    def backward(self) -> ndarray:
        """ 
        Computes gradient of the loss value w.r.t the input to the 
        loss function.
        """
        self._input_grad = self.get_input_grad()
        
        assert self._prediction.shape == self._input_grad
        
        return self._input_grad
    
    def get_output(self) -> float:
        """ 
        Every sublass of "Loss" must implement the get_output function.
        """
        raise NotImplementedError()
        
    def get_input_grad(self) -> ndarray:
        """ 
        Every sublass of "Loss" must implement the get_input_grad function.
        """
        raise NotImplementedError()
    
class MeanSquaredError(Loss):
    """ 
    Mean squared loss function
    """
    def __init__(self):
        super().__init__()
    
    def get_output(self) -> float:
        """ 
        Computes the squared error loss for each observation
        """
        loss = np.sum(np.power(self._prediction - self._target, 2)) /self._prediction.shape[0]

        return loss
    
    def get_input_grad(self) -> ndarray:
        """
        Computes the loss gradient w.r.t. to the input for MSE loss.
        """
        return 2.0 * (self._prediction - self._target) / self._prediction.shape[0]