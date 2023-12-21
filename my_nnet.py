from keras.models import Sequential
from keras.layers import Dense
from keras import activations
import tensorflow as tf
import z3

"""
    Extract weights and bias from a layer
"""
def get_info(layer: Dense):
    weights = layer.kernel_initializer.value
    bias = layer.bias_initializer.value
    return (weights, bias)

def my_symbolic_execution(dnn: Sequential):
    """
    Return symbolic states from a dnn
    Args:
        dnn (type: Sequential)
    """

    constraint_list = []
    layers = dnn.layers

    # Extract info from the first layer to define a list of inputs
    first_layer = layers[0]
    weights, _ = get_info(first_layer)
    num_inputs = len(weights)
    input_list = [z3.Real('i' + str(i)) for i in range(num_inputs)]

    # Iterate through layers to determine the neurons in each layer
    for index in range(len(layers) - 1):
        # Get the layer info
        layer = layers[index]
        weights, bias = get_info(layer)
        num_neurons = len(weights[0])

        # Define the neuron list
        neuron_list = [z3.Real('n' + str(index) + '_' + str(x)) for x in range(num_neurons)]
        for i in range(len(neuron_list)):
            # Find the sum weights of a neuron
            total = weights[0][i] * input_list[0]
            for input in range(1, len(input_list)):
                total += (weights[input][i] * input_list[input])

            # Add bias and clean up the formula
            total += bias[i][0]
            total = z3.simplify(total)

            # Append the constraint into the list
            constraint_list.append(neuron_list[i] == z3.If(total <= 0, 0, total))

        # The current neuron list becomes the inputs for the next layer
        input_list = neuron_list

    # Extract info from the last layer to define a list of outputs
    last_layer = layers[-1]
    weights, bias = get_info(last_layer)
    num_outputs = len(weights[0])
    output_list = [z3.Real('o' + str(i)) for i in range(num_outputs)]

    for i in range(len(output_list)):
        # Find the sum weights of the final outputs
        total = weights[0][i] * input_list[0]
        for input in range(1, len(input_list)):
            total += (weights[input][i] * input_list[input])
        total += bias[i][0]
        total = z3.simplify(total)

        # Append the outputs into the list
        constraint_list.append(output_list[i] == total)

    return z3.And(constraint_list)

def readNNet(filename: str) -> Sequential:
    """
    Return a DNN based on the information from the provided file
    Args:
        file (type: str): filename that provides information about NNet
    """
    file = open(filename)

    # Skip header lines
    line = file.readline()
    while line.startswith("//"):
        line = file.readline()

    parse_line = line.split(",")
    num_layers = int(parse_line[0])
    num_inputs = int(parse_line[1])

    line = file.readline()
    parse_line = line.split(",")

    layers = []
    for i in range(num_layers + 1):
        layers.append(int(parse_line[i]))

    # Skip line containing "0,"
    file.readline()

    # Read the normalization information
    line = file.readline()
    inputMins = [float(x) for x in line.strip().split(",")[:-1]]

    line = file.readline()
    inputMaxes = [float(x) for x in line.strip().split(",")[:-1]]

    line = file.readline()
    means = [float(x) for x in line.strip().split(",")[:-1]]

    line = file.readline()
    ranges = [float(x) for x in line.strip().split(",")[:-1]]

    model = Sequential()
    for i in range(1, num_layers + 1):
        nodes = layers[i]

        weights = []
        biases = []

        for _ in range(nodes):
            line = file.readline()
            weights.append([float(x) for x in line.strip().split(",")[:-1]])

        for _ in range(nodes):
            line = file.readline()
            biases.append(float(line.strip().split(",")[0]))
        if i == 1:
            dense = Dense(units = nodes,
                        input_shape = (num_inputs, ),
                        kernel_initializer = tf.constant_initializer(weights),
                        bias_initializer = tf.constant_initializer(biases),
                        dtype='float64')
        # the output layer has no activation function
        elif i == num_layers:
            dense = Dense(units = nodes,
                activation = None,
                kernel_initializer = tf.constant_initializer(weights),
                bias_initializer = tf.constant_initializer(biases),
                dtype='float64')
        # the rest
        else:
            dense = Dense(units = nodes,
                activation = activations.relu,
                kernel_initializer = tf.constant_initializer(weights),
                bias_initializer = tf.constant_initializer(biases),
                dtype='float64')

        model.add(dense)
    return model

readNNet("acasxu_run2a_1_1_batch_2000.nnet")
