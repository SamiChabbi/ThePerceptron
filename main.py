#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:24:37 2022

@author: himilco
"""
import numpy as np 
import  matplotlib.pyplot as plt 
import h5py 
import Perceptron 
from PIL import Image

def load_dataset():
    with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('datasets/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes 

x_train, y_train, x_test, y_test, classes = load_dataset()
print("Size of the training dataset : ",y_train.shape)
print("Size of the testing dataset : ",y_test.shape)
print("Size of the image : ",x_train[0].shape)
print("shape of training dataset : ",x_train.shape)
img = x_train[11]
x_train.shape
flat_x_train = x_train.reshape(x_train.shape[0], -1)
flat_x_test = x_test.reshape(x_test.shape[0], -1)
print("flat_x_train.shape", flat_x_train.shape)
print("flat_x_test.shape", flat_x_test.shape)
train_set_x = flat_x_train / 255
test_set_x = flat_x_test / 255

X = train_set_x
Y = y_train
i = 2

n_r = Perceptron.Neurone(X.shape[1])
costs, W, b = n_r.train(X, Y, 2000, 0.1)
perf = n_r.compute_performance(test_set_x, y_test)
prediction = n_r.predict(train_set_x[i])
plt.imshow(x_test[i])
plt.show()
print("performance : ",perf)
print("prediction : ",round(float(prediction), 4))
plt.plot(costs, np.linspace(0,len(costs) - 1,len(costs)))
plt.show()