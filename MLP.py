import numpy as np

# Activation functions
SIGMOID = 'sigmoid'
RELU = 'relu'


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
        self.num_layers = len(config)
        self.layers = config['layers']
        self.input_dim = config['input_dim']
        # print(self.input_dim)
        # print(self.layers)
        return

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
        return

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
    return 1.0 / (1.0 + np.exp(-z))


# relu function
def relu(z):
    """
    :param z: a vector or Numpy array
    :return:
    """
    return np.maximum(z, 0, z)


# Create network:
nn = MLP({'input_dim': 100, 'layers': [10, 20]})
print(nn.input_dim)
print(nn.layers)

# # Train:
# for epoch in range(n_iter):
#     loss = 0
#     for b in make_batches():
#     # make_batches return the indices of a mini batch
#         batch_X = data[b]
#         batch_y = label[b]
#         loss += numpy.mean(numpy.square(nn(batch_X) – batch_y))
#         gs = nn.compute_gradients(batch_X, batch_y)
#         ps = nn.get_params()
#         u = [(p[0]-lr*g[0], p[1]-lr*g[1]) for p, g in zip(ps, gs)]
#         # lr is the learning rate
#         nn.set_params(u)
#     print(loss/len(label))
#
# # Make predictions:
# p = (nn(data)>0.5).astype('float32')
