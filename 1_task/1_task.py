import math

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, weights, bias, act_func):
        self.bias = bias
        self.weights = weights
        self.act_func = act_func

    def count(self, inputs):
        results = np.dot(self.weights, inputs) + self.bias
        return self.act_func(results)


class BasicNetwork:
    def __init__(self, layer_config):
        self.network_ = []
        for layer in layer_config:
            self.network_.append([])
            for neuron in layer:
                self.network_[-1].append(neuron)

    def predict(self, inputs):
        prev_input = inputs
        for layer in self.network_:
            next_input = []
            for neuron in layer:
                next_input.append(neuron.count(prev_input))
            prev_input = next_input
        return prev_input

    def mse(self, inputs, actual_result):
        result = self.predict(inputs)
        return (np.square(np.asarray(result) - np.asarray(actual_result))).mean()


def task_1():
    print("Task 1")
    inputs = [1, 0]
    layers_config = [
        [Neuron([0.45, -0.12], 0, sigmoid), Neuron([0.78, 0.13], 0, sigmoid)],
        [Neuron([1.5, -2.3], 0, sigmoid)]
    ]
    nn = BasicNetwork(layers_config)
    print(f"Result: {nn.predict(inputs)}")
    print(f"MSE: {nn.mse(inputs, actual_result=1)}")


train_sets = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]


def task_2():
    print("\nTask 2")
    layers_config = [
        [Neuron([0.9, 0.8], 0, sigmoid), Neuron([-0.5, 0.37], 0, sigmoid)],
        [Neuron([0.5, -0.72], 0, sigmoid)]
    ]
    nn = BasicNetwork(layers_config)
    sum_error = 0
    for train_set in train_sets:
        sum_error += nn.mse(train_set[0], actual_result=train_set[1])
        print(f"Result: {nn.predict(train_set[0])}")
        print(f"MSE: {nn.mse(train_set[0], actual_result=train_set[1])}")
    sum_error /= 4
    print(f"Sum error: {sum_error}")


def task_3():
    print("\nTask 3")
    layers_config = [
        [Neuron([0.9, 0.8], 0.5, sigmoid), Neuron([-0.5, 0.37], -0.1, sigmoid)],
        [Neuron([0.5, -0.72], -0.1, sigmoid)]
    ]
    nn = BasicNetwork(layers_config)
    sum_error = 0
    for train_set in train_sets:
        sum_error += nn.mse(train_set[0], actual_result=train_set[1])
        print(f"Result: {nn.predict(train_set[0])}")
        print(f"MSE: {nn.mse(train_set[0], actual_result=train_set[1])}")
    sum_error /= 4
    print(f"Sum error: {sum_error}")


if __name__ == '__main__':
    task_1()
    task_2()
    task_3()
