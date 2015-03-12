from load_data import load_color
import numpy as np
import theano
import theano.tensor as tensor
from theano.tensor.signal.downsample import max_pool_2d
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import svd


def _relu(X):
    return X * (X > 1E-6)


def conv_layer(input_variable, filter_shape, pool_shape, stride, random_state):
    np_filters = 0.2 * (random_state.rand(
        *filter_shape).astype(theano.config.floatX) - 0.5)
    filters = theano.shared(np_filters)
    np_biases = np.zeros(filter_shape[0]).astype(theano.config.floatX)
    biases = theano.shared(np_biases)
    params = [filters, biases]
    conv = tensor.nnet.conv2d(input_variable, filters, subsample=stride)
    conv += biases.dimshuffle('x', 0, 'x', 'x')
    out = _relu(conv)
    pooled = max_pool_2d(out, pool_shape, ignore_border=True)
    return pooled, params


def fc_layer(input_variable, layer_shape, random_state):
    np_W = 0.2 * (random_state.rand(
        *layer_shape).astype(theano.config.floatX) - 0.5)
    W = theano.shared(np_W)
    np_b = np.zeros(layer_shape[1]).astype(theano.config.floatX)
    b = theano.shared(np_b)
    params = [W, b]
    out = _relu(tensor.dot(input_variable, W) + b)
    return out, params


def softmax_layer(input_variable, layer_shape, random_state):
    np_W = 0.2 * (random_state.rand(
        *layer_shape).astype(theano.config.floatX) - 0.5)
    W = theano.shared(np_W)
    np_b = np.zeros(layer_shape[1]).astype(theano.config.floatX)
    b = theano.shared(np_b)
    out = tensor.dot(input_variable, W) + b
    params = [W, b]

    e = tensor.exp(out - out.max(axis=1, keepdims=True))
    softmax = e / e.sum(axis=1, keepdims=True)

    # Gradient of softmax not defined... again!
    #softmax = tensor.nnet.softmax(out)
    return softmax, params


def softmax_cost(y_hat_sym, y_sym):
    return -tensor.mean(
        tensor.log(y_hat_sym)[tensor.arange(y_sym.shape[0]), y_sym])


def minibatch_indices(X, minibatch_size, lb=None, ub=None):
    if lb is None:
        lb = 0
    if ub is None:
        ub = len(X)
    minibatch_indices = np.arange(lb, ub, minibatch_size)
    minibatch_indices = np.asarray(list(minibatch_indices) + [ub])
    start_indices = minibatch_indices[:-1]
    end_indices = minibatch_indices[1:]
    return zip(start_indices, end_indices)


class ZCA(BaseEstimator, TransformerMixin):
    """
    Identical to CovZCA up to scaling due to lack of division by n_samples
    S ** 2 / n_samples should correct this but components_ come out different
    though transformed examples are identical.
    """
    def __init__(self, n_components=None, bias=.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

    def fit(self, X, y=None):
        if self.copy:
            X = np.array(X, copy=self.copy)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = svd(X, full_matrices=False)
        components = np.dot(VT.T * np.sqrt(1.0 / (S ** 2 + self.bias)), VT)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed

X_t, y_t = load_color()
shp = X_t.shape
X_train = X_t[:20000].reshape(20000, -1).astype('float32')
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
X_t = (X_t[:].reshape(len(X_t), -1) - mean) / std
"""
# ZCA not working...
zca = ZCA()
print("Performing ZCA preprocessing...")
X_train = X_t[:20000]
zca.fit(X_train)
X_t = zca.transform(X_t)
print("ZCA complete")
"""
X_t = X_t.reshape(*shp)

X = tensor.tensor4('X')
y = tensor.ivector('y')
params = []
random_state = np.random.RandomState(1999)

# n_filters, n_dim, kernel_width, kernel_height
filter_shape = (32, 3, 3, 3)
pool_shape = (2, 2)
stride = (2, 2)
out, l_params = conv_layer(X, filter_shape, pool_shape, stride, random_state)
params += l_params

filter_shape = (64, filter_shape[0], 2, 2)
pool_shape = (2, 2)
stride = (1, 1)
out, l_params = conv_layer(out, filter_shape, pool_shape, stride,
                           random_state)
params += l_params

filter_shape = (128, filter_shape[0], 2, 2)
pool_shape = (2, 2)
stride = (1, 1)
out, l_params = conv_layer(out, filter_shape, pool_shape, stride,
                           random_state)
params += l_params
shp = out.shape
out = out.reshape((shp[0], shp[1] * shp[2] * shp[3]))

shape = (512, 128)
out, l_params = fc_layer(out, shape, random_state)
params += l_params

shape = (shape[1], 64)
out, l_params = fc_layer(out, shape, random_state)
params += l_params

shape = (shape[1], 2)
out, l_params = softmax_layer(out, shape, random_state)
params += l_params

cost = softmax_cost(out, y)
grads = tensor.grad(cost, params)
# NaNs
#cost = cost + 0.1 * tensor.sum([tensor.sum(grad_i ** 2) for grad_i in grads])
#grads = tensor.grad(cost, params)

minibatch_size = 10
learning_rate = 0.01 / minibatch_size
updates = [(param_i, param_i - learning_rate * grad_i)
           for param_i, grad_i in zip(params, grads)]

train_function = theano.function([X, y], cost, updates=updates)
predict_function = theano.function([X], out)

epochs = 100
for n in range(epochs):
    loss = []
    for i, j in minibatch_indices(X_t, minibatch_size, lb=0, ub=20000):
        X_nt = X_t[i:j]
        # Random horizontal flips with probability 0.5
        flip_idx = np.where(random_state.rand(len(X_nt)) > 0.5)[0]
        X_nt[flip_idx] = X_nt[flip_idx][:, :, :, ::-1]
        l = train_function(X_nt, y_t[i:j])
        loss.append(l)
    loss = np.mean(loss)
    train_y_pred = []
    for i, j in minibatch_indices(X_t, minibatch_size, lb=0, ub=20000):
        train_y_hat = predict_function(X_t[i:j])
        y_p = np.argmax(train_y_hat, axis=1)
        train_y_pred.extend(list(y_p))
    valid_y_pred = []
    for i, j in minibatch_indices(X_t, minibatch_size, lb=20000, ub=25000):
        valid_y_hat = predict_function(X_t[i:j])
        y_p = np.argmax(valid_y_hat, axis=1)
        valid_y_pred.extend(list(y_p))
    train_y_pred = np.array(train_y_pred)
    valid_y_pred = np.array(valid_y_pred)
    print("Epoch %i" % n)
    print("Train Accuracy % f" % np.mean((y_t[0:20000].flatten() ==
                                          train_y_pred.flatten()).astype(
                                              "float32")))
    print("Valid Accuracy % f" % np.mean((y_t[20000:25000].flatten() ==
                                          valid_y_pred.flatten()).astype(
                                              "float32")))
    print("Loss %f" % loss)
