import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def forward(self, inputs):
        # Forward pass through the network
        hidden_input = np.dot(inputs, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)
        output = np.dot(hidden_output, self.weights_hidden_output)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Define input pattern and desired outputs
Q = np.array([1, 2, 3])
Z = np.array([[1, -1], [-2, 2]])

# Define weights for input-hidden and hidden-output layers
weights_input_hidden = np.array([[2.2, -1.2, 0.5], [-3.2, 4, -2], [2.25, -2.2, 1.5]])
weights_hidden_output = np.array([[3, -0.5], [-2.5, 1.9], [-2, 1.5]])

# Create a neural network with the provided weights
input_size = Q.shape[0]
hidden_size = weights_input_hidden.shape[1]
output_size = Z.shape[1]

# Create the neural network
network = NeuralNetwork(input_size, hidden_size, output_size)

# Set the weights to the provided values
network.weights_input_hidden = weights_input_hidden
network.weights_hidden_output = weights_hidden_output

# Perform forward pass
output = network.forward(Q)

print("Input pattern (Q):", Q)
print("Desired outputs (Z):", Z)
print("Predicted outputs:", output)
