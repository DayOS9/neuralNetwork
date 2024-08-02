import random

import numpy as np

# import tkinter as tk

# import seaborn as sb
# from matplotlib import pyplot as plt


class Network(object):
    def __init__(self, sizes):
        # layers will be the size of the list called "sizes"
        self.num_layers = len(sizes)
        # storing the list
        self.sizes = sizes
        # here we are creating a list of vectors which are one column of row size of sizes[i] not including the input layer
        # we use randn because it gives us vals with mean of 0 and st dev of 1
        self.biases = [
            np.zeros((y, 1)) for y in sizes[1:]
        ]  # setting biases to zero is sufficient
        # He initialization done for the weights for it to be compatible with relu activation
        self.weights = [
            np.random.randn(y, x) * np.sqrt(2 / x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def forwardprop(self, a):
        # using relu function as the activiation function
        # for each node in the current layer, perfrom dot product and add the bias which will be passed to activation function
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = reLU(np.dot(w, a) + b)
        # for the last layer, to keep the values between 0 and 1, we will use softmax function (essentially a probability)
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            test_data = list(test_data)  # Convert zip object to list in Python 3
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            # create a list of mini batches ( a list containing lists in which consist of tuples (data, label) )
            mini_batches = [
                training_data[j : j + mini_batch_size]
                for j in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(
                    "Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test)
                )
            else:
                print("Epoch {0} complete".format(i))

    def update_mini_batch(self, mini_batch, eta):
        # the creation of the weights and biases but this time just zeroed out
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]
        for x, y in mini_batch:
            # for each img and label in minibatch, we will get back a new set of biases and weights
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # we will simply add up all the weights and biases together from all the images in the mini batch
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # the calculations for the new weights and biases calculated after finishing the mini batch
        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward propogation
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = reLU(z)
            activations.append(activation)
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        # back propogation
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = reLU_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forwardprop(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def draw_evaluate(self):
        pass

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def reLU(z):
    return np.maximum(0, z)


def reLU_prime(z):
    return z > 0


def softmax(z):  # this will be used for the output layer
    exp_z = np.exp(z - np.max(z))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=0)
