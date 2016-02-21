import random
import math

# This class is wriiten so as to implement the NN using backpropagation algorithm 
# Took help from various online resources

# sigmoid function
def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x)

# derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

class neuralNet:
	#default value of alpha = 0.5
    def __init__(self, num_inputs, num_hidden, num_outputs, alpha = 0.5):
        
        self.num_inputs = num_inputs + 1

# It defines a neuron layer
# this can be used to create more hidden layers
class NeuronLayer:
    def __init__(self, num_neurons, bias):

        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def total_net_input(self,inputs):
        total = 0
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]
        return total + self.bias        

    def calculate_output(self, inputs):
        self.output = sigmoid(total_net_input(inputs))
        return self.output
 

nn = neuralNet(2, 2, 1)
