import numpy as np
import pandas as pd

class SingleNeuron(object):
    """
    A class used to represent a single artificial neuron capable of performing regression and classification.

    ...

    Attributes
    ----------
    activation_function : function
        The activation function applied to the preactivation linear combination.

    w_ : numpy.ndarray
        The weights and bias of the single neuron. The last entry being the bias. 
        This attribute is created when the train method is called.

    errors_: list
        A list containing the mean sqaured error computed after each iteration 
        of stochastic gradient descent per epoch. 

    Methods
    -------
    train(self, X, y, alpha = 0.005, epochs = 50)
        Iterates the stochastic gradient descent algorithm through each sample 
        a total of epochs number of times with learning rate alpha. The data 
        used consists of feature vectors X and associated labels y. 

    predict(self, X)
        Uses the weights and bias, the feature vectors in X, and the 
        activation_function to make a y_hat prediction on each feature vector. 
    """

    def __init__(self, activation_function, classification = True):
        self.activatoin_function = activation_function
        self.classification = classification

    def train(self, X, y, alpha = 0.005, epochs = 50):
        if not self.classification:
            self.w_ = np.random.rand(1 + X.shape[1])
            self.errors_ = []
            N = X.shape[0]

            for _ in range(epochs):
                errors = 0
                for xi, target in zip(X, y):
                    error = (self.predict(xi) - target)
                    self.w_[:-1] -= alpha*error*xi
                    self.w_[-1] -= alpha*error
                    errors += .5*(error**2)
                self.errors_.append(errors/N)
            return self
        
        else:
            self.w_ = np.random.rand(1 + X.shape[1])
        
            self.errors_ = []
        
            for _ in range(self.epochs):
                errors = 0
                for xi, target in zip(X, y):
                    update = self.eta * (self.predict(xi) - target)
                    self.w_[:-1] -= update*xi
                    self.w_[-1] -= update
                    errors += int(update != 0)
                if errors == 0:
                    return self
                else:
                    self.errors_.append(errors)
                
            return self
        
    def predict(self, X):
        if not self.classification:
            preactivation = np.dot(X, self.w_[:-1]) + self.w_[-1]
            return self.activation_function(preactivation)
        
        else:
            return np.where(self.net_input(X) >= 0.0, 1, -1)