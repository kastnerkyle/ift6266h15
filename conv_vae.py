# Kyle Kastner
# License: MIT
"""
VAE in a single file.
Bringing in code from IndicoDataSolutions and Alec Radford (NewMu)
"""
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from optimizers import rmsprop, sgd_nesterov
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import conv2d
import tempfile
import gzip
import cPickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave
from time import time
import os
import sys
sys.setrecursionlimit(40000)


def hard_tanh(X):
    return T.clip(X, -1., 1.)


def relu(X):
    return X * (X > 1E-6)


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


def conv_layer(input_variable, filter_shape, pool_shape, random_state):
    # This is a mode='same' convolution
    np_filters = 0.1 * (random_state.rand(
        *filter_shape).astype(theano.config.floatX) - 0.5)
    filters = theano.shared(np_filters)
    np_biases = np.zeros(filter_shape[0]).astype(theano.config.floatX)
    biases = theano.shared(np_biases)

    # Assume square filters
    s = int(np.floor(filter_shape[-1] / 2.))
    conv = conv2d(input_variable, filters, border_mode='full')[:, :, s:-s, s:-s]
    params = [filters, biases]
    conv += biases.dimshuffle('x', 0, 'x', 'x')

    # batch_normalization
    n, n_params = normalization_layer(conv, filter_shape)
    params += n_params
    out = relu(n)
    pooled = max_pool_2d(out, pool_shape)
    return pooled, params


def deconv_layer(input_variable, filter_shape, pool_shape, random_state,
                 activation="relu"):
    # This is a mode='same' convolution
    np_filters = 0.1 * (random_state.rand(
        *filter_shape).astype(theano.config.floatX) - 0.5)
    filters = theano.shared(np_filters)
    np_biases = np.zeros(filter_shape[0]).astype(theano.config.floatX)
    biases = theano.shared(np_biases)

    if pool_shape[-1] > 1:
        pooled = depool_2d(input_variable, factor=pool_shape[-1])
    else:
        pooled = input_variable
    # Assume square filters
    s = int(np.floor(filter_shape[-1] / 2.))
    conv = conv2d(pooled, filters, border_mode='full')[:, :, s:-s, s:-s]
    params = [filters, biases]
    conv += biases.dimshuffle('x', 0, 'x', 'x')

    # batch_normalization
    n, n_params = normalization_layer(conv, filter_shape)
    params += n_params
    if activation == "relu":
        out = relu(n)
    elif activation == "hard_tanh":
        out = hard_tanh(n)
    # assume square pool_shape
    return out, params


def depool_2d(X, factor=2):
    """
    perforated upsample
    http://www.brml.org/uploads/tx_sibibtex/281.pdf

    Modified from Alec Radford (NewMu)
    """
    output_shape = (X.shape[1], X.shape[2] * factor, X.shape[3] * factor)
    stride = X.shape[2]
    offset = X.shape[3]
    in_dim = stride * offset
    out_dim = in_dim * factor * factor

    upsamp_matrix = T.zeros((in_dim, out_dim))
    rows = T.arange(in_dim)
    cols = rows * factor + (rows / stride * factor * offset)
    upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)

    flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))

    up_flat = T.dot(flat, upsamp_matrix)
    upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0],
                                 output_shape[1], output_shape[2]))
    return upsamp


def normalization_layer(input_variable, layer_shape):
    if len(layer_shape) == 4:
        # conv bc01 but layer_shape is (new_c, old_c, w, h)
        np_G = np.ones(layer_shape[0]).astype(theano.config.floatX)
        np_B = np.zeros(layer_shape[0]).astype(theano.config.floatX)
        G = theano.shared(np_G)
        B = theano.shared(np_B)
        normed = (input_variable - input_variable.mean(
            axis=(0, 2, 3), keepdims=True)) / (input_variable.std(
                axis=(0, 2, 3), keepdims=True) + 1E-6)
        out = G.dimshuffle('x', 0, 'x', 'x') * normed + B.dimshuffle(
            'x', 0, 'x', 'x')
    else:
        np_G = np.ones(layer_shape[1]).astype(theano.config.floatX)
        np_B = np.zeros(layer_shape[1]).astype(theano.config.floatX)
        G = theano.shared(np_G)
        B = theano.shared(np_B)
        normed = (input_variable - input_variable.mean(
            axis=0, keepdims=True)) / (input_variable.std(
                axis=0, keepdims=True) + 1E-6)
        out = G * normed + B
    params = [G, B]
    return out, params


def linear_layer(input_variable, layer_shape, random_state):
    np_W = 0.1 * (random_state.rand(
        *layer_shape).astype(theano.config.floatX) - 0.5)
    W = theano.shared(np_W)
    np_b = np.zeros(layer_shape[1]).astype(theano.config.floatX)
    b = theano.shared(np_b)
    params = [W, b]
    l = T.dot(input_variable, W) + b

    # batch_normalization
    out, n_params = normalization_layer(l, layer_shape)
    params += n_params
    return out, params


def relu_layer(input_variable, layer_shape, random_state):
    out, params = linear_layer(input_variable, layer_shape, random_state)
    return relu(out), params


def tanh_layer(input_variable, layer_shape, random_state):
    out, params = linear_layer(input_variable, layer_shape, random_state)
    return T.tanh(out), params


def bw_grid_vis(X, show=True, save=False, transform=False):
    ngrid = int(np.ceil(np.sqrt(len(X))))
    sqrt_shp = int(np.sqrt(X.shape[1]))
    npxs = np.sqrt(X[0].size)
    img = np.zeros((npxs * ngrid + ngrid - 1,
                    npxs * ngrid + ngrid - 1))
    for i, x in enumerate(X):
        j = i % ngrid
        i = i / ngrid
        if len(x.shape) < 3:
            x = x.reshape((sqrt_shp, sqrt_shp))
        img[i*npxs+i:(i*npxs)+npxs+i, j*npxs+j:(j*npxs)+npxs+j] = x
    if show:
        plt.imshow(img, interpolation='nearest')
        plt.show()
    if save:
        imsave(save, img)
    return img


def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d


def mnist(datasets_dir='/Tmp/kastner'):
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    random_state = np.random.RandomState(1000)
    pwr = 0.0
    test_x, test_y = test_set
    test_x = test_x.astype('float32') + pwr * random_state.randn(*test_x.shape)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32') + pwr * random_state.randn(
        *valid_x.shape)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32') + pwr * random_state.randn(
        *train_x.shape)
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def make_paths(n_code, n_paths, n_steps=480):
    """
    create a random path through code space by interpolating between points
    """
    paths = []
    p_starts = np.random.randn(n_paths, n_code)
    for i in range(n_steps / 48):
        p_ends = np.random.randn(n_paths, n_code)
        for weight in np.linspace(0., 1., 48):
            paths.append(p_starts*(1-weight) + p_ends*weight)
        p_starts = np.copy(p_ends)

    paths = np.asarray(paths)
    return paths


# TODO: FIX THIS WHOLE THING
class PickleMixin(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class ConvVAE(PickleMixin):
    def __init__(self, image_save_root=None, snapshot_file="snapshot.pkl",
                 enc_sizes=[256, 128],
                 dec_sizes=[256, 128], n_code=64, learning_rate=0.1,
                 momentum=0.99, batch_size=20, n_epoch=100):
        self.srng = RandomStreams()
        self.enc_sizes = enc_sizes
        self.dec_sizes = dec_sizes
        self.n_code = n_code
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = theano.shared(np.cast['float32'](learning_rate))
        self.momentum = momentum
        self.costs_ = []
        self.epoch_ = 0
        self.snapshot_file = snapshot_file
        self.image_save_root = image_save_root
        """
        if os.path.exists(self.snapshot_file):
            print("Loading from saved snapshot " + self.snapshot_file)
            f = open(self.snapshot_file, 'rb')
            classifier = cPickle.load(f)
            self.__setstate__(classifier.__dict__)
            f.close()
        """

    def _setup_functions(self, X, random_state):
        X_sym = T.tensor4()
        e_sym = T.matrix()
        X_sym.tag.test_value = X[:self.batch_size]
        e_sym.tag.test_value = random_state.randn(
            self.batch_size, self.n_code).astype(theano.config.floatX)

        """
        Z_sym = T.matrix()
        Z_sym.tag.test_value = random_state.randn(
            self.n_batch, self.n_code).astype(theano.config.floatX)
        """

        enc_tuples = []
        dec_tuples = []
        if len(X.shape) != 4:
            raise ValueError("Batch should be 4D in b, c, h, w format")

        # Number of channels in below layer
        prev_size = X.shape[1]
        downpool_factor = 1
        for s in self.enc_sizes:
            if isinstance(s, (tuple, list)):
                enc_t = (s[0], prev_size, s[1], s[2], s[3])
            else:
                raise ValueError("ConvVAE only takes tuples of encoder size")
                enc_t = (prev_size, s)
            downpool_factor = downpool_factor * s[3]
            enc_tuples.append(enc_t)
            prev_size = s[0]

        downpool_size = X.shape[2] / downpool_factor
        if X.shape[2] % downpool_factor != 0:
            raise ValueError("Pool shapes must match image size"
                             "and layer depth!")

        prev_size = X.shape[1]
        for s in self.dec_sizes[::-1]:
            if isinstance(s, (tuple, list)):
                dec_t = (prev_size, s[0], s[1], s[2], s[3])
            else:
                raise ValueError("ConvVAE only takes tuples of encoder size")
                dec_t = (prev_size, s)
            dec_tuples.append(dec_t)
            prev_size = s[0]

        print(enc_tuples)
        print(dec_tuples[::-1])

        if not hasattr(self, "params"):
            print('generating weights')
            enc_params = []
            in_sym = X_sym
            for n in range(len(enc_tuples)):
                filter_shape = (enc_tuples[n][0], enc_tuples[n][1],
                                enc_tuples[n][2], enc_tuples[n][3])
                pool_shape = (enc_tuples[n][4], enc_tuples[n][4])
                print("Encode filters")
                print(filter_shape)
                print(pool_shape)
                out_sym, params = conv_layer(in_sym, filter_shape,
                                             pool_shape, random_state)
                enc_params.extend(params)
                in_sym = out_sym

            # linear tuple
            # assume square
            in_sym = in_sym.reshape((in_sym.shape[0], -1))
            latent_size = (enc_tuples[-1][0] * downpool_size * downpool_size,
                           self.n_code)
            in_sym, hidden_params = tanh_layer(in_sym, latent_size,
                                               random_state)
            enc_params.extend(hidden_params)

            translation_size = (self.n_code, self.n_code)
            mu_sym, mu_params = linear_layer(in_sym,
                                             translation_size,
                                             random_state)
            enc_params.extend(mu_params)
            sigma_sym, sigma_params = linear_layer(in_sym,
                                                   translation_size,
                                                   random_state)
            # Constrain to be > 0
            sigma_sym = T.nnet.softplus(sigma_sym + 1E-15)
            enc_params.extend(sigma_params)
            self.enc_params = enc_params

            # Code layer calculations
            log_sigma_sym = T.log(sigma_sym)
            code_sym = mu_sym + T.exp(log_sigma_sym) * e_sym

            # Decoding from the code layer
            decode_size = (self.n_code,
                           dec_tuples[-1][1] * downpool_size * downpool_size)
            dec_sym, dec_params = relu_layer(code_sym,
                                             decode_size,
                                             random_state)

            reshape_size = (-1, dec_tuples[-1][1], downpool_size,
                            downpool_size)
            print("Reshape size")
            print(reshape_size)
            dec_sym = dec_sym.reshape(reshape_size)
            # stop = -1 to include 0
            in_sym = dec_sym
            for n in range(len(dec_tuples) - 1, -1, -1):
                # Reverse order due to end reversal
                if len(enc_tuples[n]) < 4:
                    out_sym, params = relu_layer(in_sym, dec_tuples[n],
                                                 random_state)
                else:
                    filter_shape = (dec_tuples[n][0], dec_tuples[n][1],
                                    dec_tuples[n][2], dec_tuples[n][3])
                    pool_shape = (enc_tuples[n][4], enc_tuples[n][4])
                    print("Decode filters")
                    print(filter_shape)
                    print(pool_shape)
                    if n == 0:
                        out_sym, params = deconv_layer(in_sym, filter_shape,
                                                       pool_shape,
                                                       random_state,
                                                       activation="hard_tanh")

                    else:
                        out_sym, params = deconv_layer(in_sym, filter_shape,
                                                       pool_shape, random_state)
                dec_params.extend(params)
                in_sym = out_sym

            y_sym = out_sym
            self.dec_params = dec_params
            self.params = self.enc_params + self.dec_params

        # Derived from
        # http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # with \sigma_2 = 1 and \mu_2 = 0
        # Key identity:
        # x = exp(log(x))
        # exp(log(sigma ** 2)) = exp(2 log(sigma))
        kl_cost = -0.5 * T.sum(2 * log_sigma_sym - T.exp(2 * log_sigma_sym) -
                               mu_sym ** 2 + 1)
        # see https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf
        # page 3
        likelihood_cost = T.sum(T.sqr(X_sym - y_sym))
        # from Autoencoding Variational Bayes
        # http://arxiv.org/abs/1312.6114
        cost = kl_cost + likelihood_cost

        learning_rate = self.learning_rate
        momentum = self.momentum
        grads = T.grad(cost, self.params)
        opt = rmsprop(self.params)
        updates = opt.updates(self.params, grads,
                              learning_rate / np.cast['float32'](
                                  self.batch_size),
                              momentum)

        print('compiling')
        self._fit_function = theano.function([X_sym, e_sym], cost,
                                             updates=updates)
        self._reconstruct = theano.function([X_sym, e_sym], y_sym)
        self._x_given_z = theano.function([code_sym], y_sym)
        self._z_given_x = theano.function([X_sym], (mu_sym, log_sigma_sym))

    def fit(self, X):
        random_state = np.random.RandomState(1999)

        if not hasattr(self, "_fit_function"):
            self._setup_functions(X, random_state)

        xs = random_state.randn(self.batch_size, self.n_code).astype(
            theano.config.floatX)
        idx = random_state.randint(0, len(X), self.batch_size)
        x_rec = X[idx].astype(theano.config.floatX)
        n = 0.
        for e in range(self.n_epoch):
            t = time()
            for n, (i, j) in enumerate(minibatch_indices(X, self.batch_size)):
                xmb = X[i:j]
                cost = self._fit_function(xmb, random_state.randn(
                    xmb.shape[0], self.n_code).astype(theano.config.floatX))
                self.costs_.append(cost)
                n += xmb.shape[0]
            print("Train iter", e)
            print("Total iters run", self.epoch_)
            print("Total Cost", cost)
            print("Mean Cost per Example", cost / len(xmb))
            print("Time", time() - t)
            self.epoch_ += 1

            if e % (self.n_epoch // 10) == 0 or e == (self.n_epoch - 1):
                print("Saving model snapshot")
                f = open(self.snapshot_file, 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            def tf(x):
                return ((x + 1.) / 2.).transpose(1, 2, 0)

            if e == (self.n_epoch - 1) or e % (self.n_epoch // 10) == 0:
                if self.image_save_root is None:
                    image_save_root = os.path.split(__file__)[0]
                else:
                    image_save_root = self.image_save_root
                samples_path = os.path.join(
                    image_save_root, "sample_images_epoch_%d" % self.epoch_)
                if not os.path.exists(samples_path):
                    os.makedirs(samples_path)

                samples = self._x_given_z(xs)
                samples = samples[:100]
                recs = self._reconstruct(x_rec, random_state.randn(
                    len(x_rec), self.n_code).astype(theano.config.floatX))
                recs = recs[:100]
                x_rec = x_rec[:100]

                img1 = bw_grid_vis(x_rec, show=False)
                img2 = bw_grid_vis(recs, show=False)
                img3 = bw_grid_vis(samples, show=False)

                imsave(os.path.join(samples_path, 'source.png'), img1)
                imsave(os.path.join(samples_path, 'source_recs.png'), img2)
                imsave(os.path.join(samples_path, 'random_samples.png'), img3)

                paths = make_paths(self.n_code, 3)
                for i in range(paths.shape[1]):
                    path_samples = self._x_given_z(paths[:, i, :].astype(
                        theano.config.floatX))
                    for j, sample in enumerate(path_samples):
                        imsave(os.path.join(samples_path,
                                            'paths_%d_%d.png' % (i, j)),
                               sample.squeeze())

    def transform(self, x_rec):
                recs = self._reconstruct(
                    x_rec, np.random.randn(x_rec.shape[0], self.n_code).astype(
                        theano.config.floatX))
                return recs

    def encode(self, X, e=None):
        if e is None:
            e = np.ones((X.shape[0], self.n_code)).astype(
                theano.config.floatX)
        return self._z_given_x(X, e)

    def decode(self, Z):
        return self._z_given_x(Z)

if __name__ == "__main__":
    tr, _, _, = mnist()
    trX, trY = tr
    tf = ConvVAE(image_save_root="/Tmp/kastner/conv_vae",
                 snapshot_file="/Tmp/kastner/conv_mnist_snapshot.pkl",
                 enc_sizes=[(64, 3, 3, 2), (128, 3, 3, 2)],
                 dec_sizes=[(128, 3, 3, 2), (64, 3, 3, 2)],
                 n_code=512,
                 learning_rate=.01, momentum=0.9, n_epoch=1000, batch_size=128)
    trX = trX.astype(theano.config.floatX)
    trX = trX.reshape(len(trX), 1, 28, 28)
    tf.fit(trX)
    recs = tf.transform(trX[:100])
