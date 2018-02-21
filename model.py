import numpy

# Activation functions
SIGMOID = 'sigmoid'
RELU = 'relu'


class Layer(object):
    def __init__(self, units, activation='relu', **kwargs):
        if 'input_dim' in kwargs:
            self.input_dim = (kwargs.pop('input_dim'),)
        self.units = units
        self.activation = activation


class MLP(object):
    def __init__(self, config):
        """
        :param config: a python dictionary providing the configuration of the
        neural network. It contains two
        entries:
        input_dim: int, the dimension of the input
        layers: list of int, the number of neurons in each hidden layer.
        E,g., [10, 20] means the
        first hidden layer has 10 neurons and the second has 20.
        """
        self.sizes = []
        self.sizes = [config['input_dim']] + config['layers'] + [1]

        # self.num_layers = len(config)
        # layers = config['layers']
        self.input_dim = config['input_dim']

        # self.layers = []
        # # Input layer (+1 unit for bias)
        # self.layers.append(numpy.zeros(config['input_dim'] + 1))
        # # Hidden layer(s)
        # for i in layers[:]:
        #     self.layers.append(numpy.zeros(i))
        # # Output layer
        # self.layers.append(numpy.zeros(1))

        self.biases = [numpy.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in
                        zip(self.sizes[:-1], self.sizes[1:])]
        #
        # self.add(Layer(layers[0], input_dim=config['input_dim']))
        # for n in layers[1:]:
        #     self.add(Layer(n))
        # self.add(Layer(1, activation='sigmoid'))

        # self.biases = [numpy.random.randn(y, 1) for y in self.layers[1:]]
        # self.weights = [numpy.random.randn(y, x) for x, y in
        #                 zip(self.layers[:-1], self.layers[1:])]
        # for x , y in(zip(self.layers[:-1], self.layers[1:])):
        #     print(str(x) + " : " + str(y))
        print(self.biases)
        # print(self.weights)
        return

    # def add(self, layer):
    #     pass

    # compute the output of the neural network.
    def __call__(self, data):
        """
        :param data: numpy ndarray of size (n x m) (i.e., a matrix) with dtype
        float32. Each row is a data point
        (total n data points) and each column is a feature (m features).
        :return:  numpy ndarray of size (n,), the output from the neuron in
        the output layer for the n data
        points.
        """
        # for k in range(data.shape[0]):
        #     input = data[k, :self.input_dim]
        #     hidden = numpy.zeros(self.hidden_len,)
        #     for i in range(self.hidden_len):
        # for k in range(data.shape[0]):
        #     data_input = data[k, : self.input_dim]
        output = list()
        for data_input in data:
            data_input = numpy.reshape(data_input, [data_input.shape[0], 1])
            for b, w in zip(self.biases[:-1], self.weights[:-1]):
                data_input = sigmoid(numpy.dot(w, data_input) + b)
            for b, w in zip(self.biases[-1:], self.weights[-1:]):
                data_input = relu(numpy.dot(w, data_input) + b)
            output.append(data_input)
        return output

    # Compute the gradients of the parameters for the total loss
    def compute_gradients(self, data, label):
        """
        :param data: numpy ndarray of size (n x m) (i.e., a matrix) with dtype
        float32. Each row is a data point
        (total n data points) and each column is a feature (m features).
        :param label:  numpy ndarray of size (n,) (i.e., a vector) with dtype
        float32. The labels of the n data points.
        :return:list of tuples where each tuple contains the partial
        derivatives (two numpy ndarrays) for
        the kernel and the bias of a layer. For example, suppose the
        network has two hidden layers, the
        list would look like: [(dedk_1, dedb_1), (dedk_2, dedb_2),
        (dedk_o, dedb_o)] where dedk_o is the
        partial derivative for the kernel of the output neuron and dedb_o
        is the derivative for the neuron’s
        bias. (Note the number of tuples in the returned list is: # of
        hidden layers + 1.)
        """
        return

    def get_params(self):
        """
        :return: the parameters of the neural network as a list of tuples [
        (kernel_1, bias_1), (kernel_2,
        bias_1), …]. Note the parameters for the output neuron should be
        the last tuple in the list. To
        make it uniform, return the bias of the output neuron as a vector
        of single entry.
        """
        return

    # Set the parameters of the network
    def set_params(self, ps):
        """
        :param ps: a list of tuples in the form of [(kernel_1, bias_1),
        (kernel_2, bias_2), …] (last tuple for the
        output neuron, the bias will be given as a vector of single
        entry). Calling the function sets the
        parameters of the network to the values given in ps.
        """


# Sigmoid function
def sigmoid(z):
    """
    :param z: a vector or Numpy array
    :return:
    """
    return 1.0 / (1.0 + numpy.exp(-z))


# relu function
def relu(z):
    """
    :param z: a vector or Numpy array
    :return:
    """
    return numpy.maximum(z, 0, z)
