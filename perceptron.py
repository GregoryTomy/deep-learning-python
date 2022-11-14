# %%
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap

# %%
iris = load_iris()
cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
data = pd.DataFrame(data= iris.data, columns = cols)
data['target'] = iris.target
data.head()
# %%
data.drop(columns=['sepal width (cm)', 'petal width (cm)'], inplace=True)
drop_indx = data[data['target'] == 1].index
data.drop(drop_indx, inplace=True)
data.head()

# Normalize the data
scaler = MinMaxScaler()
normalized= scaler.fit_transform(data)
data = pd.DataFrame(normalized, columns=data.columns)
data.head()

# %%
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'],
            color=np.where(data['target'] == 0, 'r', 'g'))
# %%
X = data[['sepal length (cm)', 'petal length (cm)']].values # (100, 2)
y = data['target'].to_numpy().reshape(100,1) # (100, 1)


###################################################################################################
###################################################################################################

# %%
def sigmoid(X: ndarray) -> ndarray:
    """
    The activation function of the perceptron
    """
    return 1 / (1 + np.exp(-X))

def neuron_output(X: ndarray, weights: ndarray) -> ndarray:
    """
    Returns the predicted values (forward propogation)
    """
    assert X.shape[1] == weights.shape[0]
    
    return sigmoid(X @ weights)

def gradient(X: ndarray, neuron_output: ndarray, y: ndarray) -> ndarray:
    """
    Calculates the gradient of the loss function with respect to
    the weights. (backward propagation)
    """
    return X.T @ ((neuron_output - y) * neuron_output * (1 - neuron_output))
                  
def train_network(X: ndarray, y:ndarray,
                  learning_rate: float, epochs: int) -> (ndarray, ndarray):
    """
    Function that trains the perceptron.
    """
    # add bias term to the X matrix
    X = np.c_[np.ones(len(X)), X]
    num_samples, num_features = X.shape
    weights = np.random.rand(num_features, 1)
    print(f'Orignal random weights: {weights}')

    
    for _ in range(epochs):
        y_hat = neuron_output(X, weights)
        grad = gradient(X, y_hat, y)
        
        # update weights
        weights = weights - (learning_rate*grad)

    print(f'Trained weights {weights}')
    
    trained_outputs = neuron_output(X, weights)
    
    return weights, trained_outputs                
###################################################################################################
###################################################################################################

# %%
# Training the model
np.random.seed(0) 
weights, trained_outputs = train_network(X, y, 3, 80000)

# %%
print(f'Final error after training: {np.sum(trained_outputs - y)}')
trained_outputs = np.round(trained_outputs)
print(all(trained_outputs == y))

###################################################################################################
###################################################################################################

# %%
x1 = [min(X[:,1]), max(X[:,1])]
m= -weights[1]/weights[2]
c= -weights[0]/weights[2]
x2 = m*x1 + c

fig = plt.figure(figsize=(10,8))
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'],
            color=np.where(data['target'] == 0, 'r', 'g'))
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Perceptron Algorithm')
plt.plot(x1, x2, 'y-')
 # %%
