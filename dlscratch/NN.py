import numpy as np
from numpy import ndarray
from typing import List
from layers import Layer
from loss import Loss

class NeuralNetwork(object):
    """ 
    The class of neural network
    """
    
    def __init__(self, layers: List[Layer],
                 loss: Loss, seed: float = 1):
        """
        Initilize the layers and set loss value.
        """
        self._layers  = layers
        self._loss = loss
        self._seed = seed
        
        if seed:
            for layer in self._layers:
                setattr(layer, "_seed", self._seed)
    
    def forward(self, x_batch: ndarray) -> ndarray:
        """ 
        Passes data forward through a series of layers
        """
        x_out = x_batch
        for layer in self._layers:
            x_out = layer.forward(x_out)
            
        return x_out
    
    def backward(self, loss_grad:ndarray) -> None:
        """
        Passes data backward through a series of layers
        """
        grad = loss_grad
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        
        return None
    
    def train_batch(self, x_batch: ndarray, y_batch:ndarray) -> float:
        """ 
        Passes data forward through the layers. Computes the loss.
        Passes data backward through the layers.
        """
        predictions = self.forward(x_batch)
        loss = self._loss.forward(predictions, y_batch)
        self.backward(self._loss.backward())
        
        return loss
    
    def params(self):
        """Get the parameters for the network"""
        for layer in self._layers:
            yield from layer._params # yield from is cool Python 3.3 