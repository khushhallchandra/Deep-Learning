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
 

nn = neuralNet(2, 2, 1)
