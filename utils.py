# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:17:11 2017

@author: admin
"""

import pandas as pd
import numpy as np

"""a module to read and create datasets representing pictures and 
logic values"""


def load_MNIST_data_wrapper():
    """Returns lists containing tuples pairs (pixels, digit) representing a picture in the MNIST
    data set. pixels is a vector of lenght 784 which contains the consecutive rows of the 
    corresponding 28x28 picture in the MNIST data set. If i the digit represented by pixels then 
    digit is the ith basis vector. The three returned list training_data,validation_data and 
    test_data contain 50,000, 10,000 and 10,000 tuples respectively"""
    df_train = pd.read_csv('mnist_train.csv', header = None)
    df_test = pd.read_csv('mnist_test.csv', header = None)
    
    training_labels = df_train.iloc[:50000,0].values
    validation_labels = df_train.iloc[50000:,0].values
    test_labels = df_test.iloc[:,0].values
    
    training_labels = vectorized(training_labels)
    validation_labels = vectorized(validation_labels)
    test_labels = vectorized(test_labels)
    
    training_image = df_train.iloc[:50000, 1:].values*(1/255)
    validation_image = df_train.iloc[50000:,1:].values*(1/255)
    test_image = df_test.iloc[:,1:].values*(1/255)
    
    training_data = list(zip(training_image, training_labels ))
    validation_data = list(zip(validation_image, validation_labels))
    test_data = list(zip(test_image, test_labels))
    
    for i in range(len(training_data)):
        training_data[i][0].shape = (784, 1)
        training_data[i][1].shape = (10, 1)
    
    for i in range(len(validation_data)):
        validation_data[i][0].shape = (784, 1)
        validation_data[i][1].shape = (10, 1)
        
    for i in range(len(test_data)):
        test_data[i][0].shape = (784, 1)
        test_data[i][1].shape = (10, 1)
    
    return (training_data, validation_data, test_data)

def vectorized(digits):
    """return a matrix vector_digits where vector_digits[i] is the digits[i]th 
    basis vector"""
    n = len(digits)
    vector_digits = np.zeros((n, 10))
    for i in range(n):
        vector_digits[i,digits[i]]=1
    return vector_digits  

def load_logic():
    """Return the list X with 2x1 array representing (True,False) values. The lists AND and 
    OR are list containing the results from mapping X true the logic gates of AND and OR"""
    X = [[0,0],[0,1],[1,0],[1,1]]
    AND = [[1,0],[1,0],[1,0],[0,1]] 
    OR = [[1,0],[0,1],[0,1],[0,1]]
    X = [np.array(x) for x in X]
    AND = [np.array([x]) for x in AND]
    OR = [np.array([x]) for x in OR]

    for i in range(4):
        X[i].shape= (2,1)
        AND[i].shape= (2,1)
        OR[i].shape= (2,1)
        
    return (X ,AND ,OR)
