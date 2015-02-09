from load_data import load_color
import numpy as np
import theano
import theano.tensor as tensor
from theano.tensor.signal.downsample import max_pool_2d


def _relu(X):
    return X * (X < 1E-6)


def conv_layer(input_variable, filter_shape, pool_shape, random_state):
    np_filters = 0.05 * random_state.randn(
        *filter_shape).astype(theano.config.floatX)
    filters = theano.shared(np_filters)
    np_biases = np.zeros(filter_shape[0]).astype(theano.config.floatX)
    biases = theano.shared(np_biases)
    params = [filters, biases]
    conv = tensor.nnet.conv2d(input_variable, filters) + biases.dimshuffle(
        'x', 0, 'x', 'x')
    out = _relu(conv)
    pooled = max_pool_2d(out, pool_shape, ignore_border=True)
    return pooled, params


def fc_layer(input_variable, layer_shape, random_state):
    np_W = 0.05 * random_state.randn(*layer_shape).astype(theano.config.floatX)
    W = theano.shared(np_W)
    np_b = np.zeros(layer_shape[1]).astype(theano.config.floatX)
    b = theano.shared(np_b)
    params = [W, b]
    out = _relu(tensor.dot(input_variable, W) + b)
    return out, params


def softmax_layer(input_variable, layer_shape, random_state):
    np_W = 0.05 * random_state.randn(*layer_shape).astype(theano.config.floatX)
    W = theano.shared(np_W)
    np_b = np.zeros(layer_shape[1]).astype(theano.config.floatX)
    b = theano.shared(np_b)
    out = tensor.dot(input_variable, W) + b
    params = [W, b]
    softmax = tensor.nnet.softmax(out)
    return softmax, params


def softmax_cost(y_hat_sym, y_sym):
    return -tensor.mean(
        tensor.log(y_hat_sym)[tensor.arange(y_sym.shape[0]), y_sym])


def minibatch_indices(X, minibatch_size):
    minibatch_indices = np.arange(0, len(X), minibatch_size)
    minibatch_indices = np.asarray(list(minibatch_indices) + [len(X)])
    start_indices = minibatch_indices[:-1]
    end_indices = minibatch_indices[1:]
    return zip(start_indices, end_indices)


train, valid, test = load_color(reshaped=False)
train_x, train_y = train
valid_x, valid_y = valid
test_x, test_y = test

train_x = train_x.astype(theano.config.floatX)
test_y = test_y.astype('int32')
valid_x = valid_x.astype(theano.config.floatX)
valid_y = valid_y.astype('int32')
test_x = test_x.astype(theano.config.floatX)
test_y = test_y.astype('int32')

X = tensor.tensor4('X')
y = tensor.ivector('y')
params = []
random_state = np.random.RandomState(1999)

# n_filters, n_dim, kernel_width, kernel_height
l1_filter_shape = (10, 3, 3, 3)
l1_pool_shape = (2, 2)
l1_out, l1_params = conv_layer(X, l1_filter_shape, l1_pool_shape, random_state)
params += l1_params

l2_filter_shape = (10, l1_filter_shape[0], 3, 3)
l2_pool_shape = (1, 1)
l2_out, l2_params = conv_layer(l1_out, l2_filter_shape, l2_pool_shape,
                               random_state)
params += l2_params

l3_filter_shape = (10, l2_filter_shape[0], 3, 3)
l3_pool_shape = (1, 1)
l3_out, l3_params = conv_layer(l2_out, l3_filter_shape, l3_pool_shape,
                               random_state)
params += l3_params
shp = l3_out.shape
l3_out = l3_out.reshape((shp[0], shp[1] * shp[2] * shp[3]))

l4_shape = (3610, 100)
l4_out, l4_params = fc_layer(l3_out, l4_shape, random_state)
params += l4_params

l5_shape = (l4_shape[1], 2)
softmax_out, softmax_params = softmax_layer(l4_out, l5_shape, random_state)
params += softmax_params

cost = softmax_cost(softmax_out, y)
grads = tensor.grad(cost, params)
learning_rate = 0.01
updates = [(param_i, param_i - learning_rate * grad_i)
           for param_i, grad_i in zip(params, grads)]

train_function = theano.function([X, y], cost, updates=updates)
predict_function = theano.function([X], softmax_out)

epochs = 1000
minibatch_size = 100
for n in range(epochs):
    loss = []
    for i, j in minibatch_indices(train_x, minibatch_size):
        l = train_function(train_x[i:j], train_y[i:j])
        loss.append(l)
    loss = np.mean(loss)
    train_y_pred = []
    for i, j in minibatch_indices(train_x, minibatch_size):
        train_y_hat = predict_function(train_x[i:j])
        y_p = np.argmax(train_y_hat, axis=1)
        train_y_pred.extend(list(y_p))
    valid_y_pred = []
    for i, j in minibatch_indices(valid_x, minibatch_size):
        valid_y_hat = predict_function(valid_x[i:j])
        y_p = np.argmax(valid_y_hat, axis=1)
        valid_y_pred.extend(list(y_p))
    train_y_pred = np.array(train_y_pred)
    valid_y_pred = np.array(valid_y_pred)
    print("Epoch %i" % n)
    print("Train Accuracy % f" % np.mean(train_y == train_y_pred))
    print("Valid Accuracy % f" % np.mean(valid_y == valid_y_pred))
    print("Loss %f" % loss)

import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
raise ValueError()
