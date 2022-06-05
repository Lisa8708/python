# -*- encoding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import numpy as np
import random
tf.enable_eager_execution()
tf.random.set_random_seed(22)
np.random.seed(22)


# mlp: multi-layer preceptron

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

class MLP:
    def __init__(self, sizes):
        """
        :param sizes: [784, 30, 10]
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        # w: [ch_out, ch_in] [783, 30], [30, 10]
        self.weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(ch1) for ch1 in sizes[1:]]

    def forward(self, x):
        """
        :param x: [784, 1]
        :return:[10, 1]
        """
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b
            x = sigmoid(z)

        return x

    def backprop(self, x, y): #
        """
        :param x: [784, 1]
        :param y: [10, 1] one_hot_encoding
        :return:[]
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # 1.forward
        activations = [x]
        zs = []
        activation = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)

            zs.append(z)
            activations.append(activation)
        loss = np.power(activation[-1]-y, 2).sum()

        # 2.backward
        # 2.1 compute gradient on output layer
        # [10, 1]
        delta = activations[-1]*(1-activations[-1])*(activations[-1]-y)
        nabla_b[-1] = delta
        # activation: [30, 1]
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # compute hidden gradient
        for l in range(2, self.num_layers + 1):
            l = -l
            z = zs[l]
            a = activations[l]
            # delta_j
            # [10, 30]
            delta = np.dot(self.weights[l+1].T, delta)*a*(1-a)

            nabla_b[l] = delta
            # [30, 1] @ [784, 1].T => [30, 784]
            nabla_w[l] = np.dot(delta, activations[l-1].T)
        return nabla_w, nabla_b

    def train(self, training_data, epochs, batch_size, lr, test_data):
        """
        :param training_data: list of (x, y)
        :param epochs: 1000
        :param batch_size: 10
        :param lr: 0.01
        :param test_data: list of (x, y)
        :return:
        """
        #if test_data:
        n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]

            # for every batch in current data
            loss = 0
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, lr)
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)}/{n_test}, loss:{loss}')
            else:
                print(f'Epoch {j} complete')


    def update_mini_batch(self, batch, lr):
        """
        :param batch: list of (x, y)
        :param lr: 0.01
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        loss = 0

        # for every sample in current batch
        for (x, y) in batch:
            # list of every w/b gradient
            # [w1, w2, w3]
            nabla_w_, nabla_b_ = self.backprop(x, y)
            nabla_w = [accu+cur for accu, cur in zip(nabla_w, nabla_w_)]
            nabla_b = [accu+cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_
        nabla_w = [w/len(batch) for w in nabla_w]
        nabla_b = [b/len(batch) for b in nabla_b]
        loss = loss/len(batch)

        # w = w-lr*nable_w
        self.weights = [w - lr*w for w in nabla_w]
        return loss

    def evaluate(self, test_data):
        """
        # y is not one-hot
        :param test_data: list of (x, y)
        :return:
        """
        result = [(np.argmax(self.forward(x)), y) for x, y in test_data]
        correct = sum(int(pred==y) for pred, y in result)
        return correct

def main():
    train_x, train_y, test_x, test_y = datasets.minist.load()



if __name__ == '__main__':
    main()