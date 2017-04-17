# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:30:37 2017

@author: user
"""

import random
import numpy as np

class Network:
    """This class represents a neural network. You can initialize
    a network by calling the class with a list representing the
    structur of the layers of the network. The list should contain
    integers specifying the number of neurons in the layer of the
    network corresponding to the index of the integer in the list.
    The function 'forward' can be called with an input vector to let
    the network make a prediction and the function 'train' can be
    called on a data set which should be a list of tuples of samples
    and labels. If a test data set, of similar structure to the training
    data set, is included when calling 'train' the instance variable
    'self.progress' can be called after training.""" 
    
    def __init__(self, sizes):
        """Initialize a random neural network with len(size) layers,
        including the input layer, where len(size[i]) is the number
        of neurons in layer i"""
        self.progress = None
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1],
                                                             sizes[1:])]
        
    
    def forward(self, x):
        """Calculate the output 'signal' of the neural network for input 'x'"""
        signal = x
        for w,b in zip(self.weights, self.biases):
            weighted_sum  = np.dot(w, signal) + b
            signal = sigmoid(weighted_sum)
        return signal
    
    
    def train(self, training_data, n_iter, eta, test_data = None,
              mini_batch_size = 1):
        """Train the neural network over the data set 'training_data' for
        'n_iter' times. The weights and biases get updated by the average
        of the partial derivate. Also save the progress in the list
        'progress'. The datasets 'training_data' and 'test_data' should be
        lists containing tuples of an input vector and the corresponding
        correct output vector."""
        training_data_size = len(training_data)
        
        if test_data:
            self.progress = []
            self.progress.append(self.test(test_data))
            print(self.progress[-1])
        
        for _ in range(n_iter):
            for i in range(0,training_data_size-mini_batch_size+1,
                           mini_batch_size):
                self.backpropagate_mini_batch(training_data[i:i+mini_batch_size],
                                              mini_batch_size, eta)
            random.shuffle(training_data)
            if test_data:
                self.progress.append(self.test(test_data))
                print(self.progress[-1])
            
    def backpropagate_mini_batch(self, mini_batch, mini_batch_size, eta):
        """First sum over the partial derivative for the weights and
        biases for each (features, label) pair in 'mini_batch'. Afterwards
        update the weights and biases by the average of the partial
        derivatives"""
        sum_nabla_b = [0]*len(self.weights)
        sum_nabla_W = [0]*len(self.weights)
        for x, y in mini_batch:
            nabla_b , nabla_W = self.backpropagate(x,y)
            addlist(sum_nabla_b, nabla_b)
            addlist(sum_nabla_W, nabla_W)
        addlist(self.biases, sum_nabla_b, -eta/mini_batch_size)
        addlist(self.weights, sum_nabla_W, -eta/mini_batch_size)
        
        
    def backpropagate(self, x, y):
        """Use the backpropagtion algorithm to calculate the the
        partial derivatives of the cost function for the weights,
        stored in 'nabla_w', and for the biases, stored in 'deltas'
        for feature vector x and label y. The list 'signals' contains
        vectors with outputs of each layer of neurons (the input
        layer included). The list 'weighted_sums' comtains for each
        layer (except the input layer) a vector with elements the
        weighted sum over the previous layer for the corresponding
        neuron."""
        signal=x
        signals = [signal]
        weighted_sums = []
        for w,b in zip(self.weights, self.biases):
            weighted_sum = np.dot(w, signal) + b
            weighted_sums.append(weighted_sum)
            signal = sigmoid(weighted_sum)
            signals.append(signal)
            
        error = signals[-1]-y
        delta = error*sigmoid_prime(weighted_sums[-1])
        deltas = [delta]
    
        for i in range(1, len(self.weights)):
            delta = np.dot(self.weights[-i].transpose(),
                           delta)*sigmoid_prime(weighted_sums[-i-1])
            deltas.insert(0, delta)
            
        nabla_w = []
        for i in range(len(self.weights)):
            nabla_w.append(np.dot(deltas[i],signals[i].transpose()))
        return (deltas, nabla_w)

    def test(self, test_data):
        """Return the number of correct predictions for the 'test_data' by the
        neural network."""
        hits = 0
        for x, y in test_data:
            if np.argmax(self.forward(x))==np.argmax(y):
                    hits+=1
        return hits

    
"""Helper funtions"""    
    
def addlist(list1, list2, rate = 1):
    """add 'rate'*'list2' to 'list1'"""
    for i in range(len(list1)):
        list1[i] = list1[i]+rate*list2[i]
    
def sigmoid(z):
    """the sigmoid function"""
    return 1.0/(1.0+np.exp(-z)) 
        
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
