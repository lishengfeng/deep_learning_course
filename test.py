import numpy as np
from model import MLP

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def select_data(X, y, i, n, ny):
    ixs = np.nonzero(y == i)[0]
    xn = X[ixs[np.random.permutation(len(ixs))[:n]]]
    yn = ny * np.ones((n,))
    return xn, yn


X0, y0 = select_data(x_train, y_train, 2, 1000, 0)
X1, y1 = select_data(x_train, y_train, 6, 1000, 1)
data = np.concatenate([X0, X1], axis=0)
label = np.concatenate([y0, y1], axis=0)

# shuffle data
ixs = np.random.permutation(len(label))
data = data[ixs].reshape((len(label), -1))
label = label[ixs]


def make_batches(total, batch_size):
    start = 0
    batches = []
    while start < total:
        batches.append(slice(start, min(total, start + batch_size)))
        start += batch_size
    return batches


# Create network:
nn = MLP({'input_dim': 28 * 28, 'layers': [16, 16]})


# Train:
def train(data, label, n_iter=60, lr=0.001, batch_size=32):
    # totalloss = np.mean(np.square(nn(data) - label))
    # print(totalloss)
    for epoch in range(n_iter):
        loss = 0
        batchs = make_batches(data.shape[0], batch_size)
        for b in batchs:
            # make_batches return the indices of a mini batch
            batch_X = data[b]
            batch_y = label[b]
            loss += np.mean(np.square(nn(batch_X) - batch_y))
            gs = nn.compute_gradients(batch_X, batch_y)
            ps = nn.get_params()
            u = [(p[0] - lr * g[0], p[1] - lr * g[1]) for p, g in zip(ps, gs)]
            # lr is the learning rate
            nn.set_params(u)
        # print(loss / len(label))
    # totalloss = np.mean(np.square(nn(data) - label))
    # print(totalloss)


before = nn(data)
print('before training, size of items > 0.5: ' + str((before > 0.5).sum()))
#
train(data, label)
#
after = nn(data)
print('after training, size of items > 0.5: ' + str((after > 0.5).sum()))
# Make predictions:
# p = (nn(data) > 0.5).astype('float32')
