#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:07:58 2022

@author: himilco
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear_function(X, W, b):
    linear_y_hat = np.dot(X , W) + b
    return linear_y_hat

class Neurone:
    '''Logistical regression single neuron'''
    def __init__(self, nb_features):
        print("Creating new neuron")
        self.W, self.b = self.init_param(nb_features)
        
    def forward(self, X, Y):
        W = self.W
        b = self.b
        Z =linear_function(X, W, b)
        Y_hat = sigmoid(Z)
        m = X.shape[1]
        cost = - 1/m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return cost , Y_hat

    def backward(self, Y, Y_hat, X, learning_rate):
        W = self.W
        b = self.b
        m = X.shape[1]
        d_W = 1/m *(np.dot(X.T,(Y_hat - Y)))
        d_b = 1/m * np.sum(Y_hat - Y)
        n_W = W - learning_rate * d_W
        n_b = b - learning_rate * d_b
        return n_W, n_b

    def init_param(self, dim):
        W = np.zeros((dim, 1))
        b = 0
        return W, b
    
    def compute_performance(self, X, Y_):
        Y = Y_.T
        W = self.W
        b = self.b
        Z =linear_function(X, W, b)
        Y_hat = sigmoid(Z)
        correct_prediction_tab = (Y_hat > 0.5) == Y
        correct_prediction_count = np.count_nonzero(correct_prediction_tab == True)
        percentage = (correct_prediction_count * 100)/ correct_prediction_tab.size
        return percentage  
    
    def predict(self, X):
        Z =linear_function(X, self.W, self.b)
        Y_hat = sigmoid(Z)
        return Y_hat
    
    def train(self, X, Y_, epochs, lr):
        Y = Y_.T
        costs = []
        for i in range(0, epochs):
            cost = 0
            W = self.W
            b = self.b
            cost, Y_hat = self.forward(X, Y)
            costs.append(cost)
            self.W, self.b = self.backward(Y, Y_hat, X, lr)
        return costs, W, b
    




