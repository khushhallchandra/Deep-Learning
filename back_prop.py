import random
import math

# This class is wriiten so as to implement the NN using backpropagation algorithm 
# Took help from various online resources

# sigmoid function
def sigmoid(x):
   return 1.0/(1.0 + math.exp(-x)

# derivative of the sigmoid function
def sigmoid_prime(x):
   return (sigmoid(x) * ( 1.0 - sigmoid(x)))

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

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs            

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

    def pd_error_wrt_total_net_input(self, target_output):
        return self.pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)        
 

nn = neuralNet(2, 2, 1)
