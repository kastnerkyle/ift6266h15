# (C) Kyle Kastner, June 2014
# License: BSD 3 clause

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_batches
from scipy.linalg import eigh
from scipy.linalg import svd
import numpy as np


class EigZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, scale_by=1., copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy
        self.scale_by = float(scale_by)

    def fit(self, X, y=None):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        eigs, eigv = eigh(np.dot(X.T, X) / n_samples +
                          self.bias * np.identity(n_features))
        components = np.dot(eigv * np.sqrt(1.0 / eigs), eigv.T)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed


class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, scale_by=1., copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy
        self.scale_by = float(scale_by)

    def fit(self, X, y=None):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = svd(np.dot(X.T, X) / n_samples, full_matrices=False)
        components = np.dot(VT.T * np.sqrt(1.0 / (np.diag(S) + self.bias)), VT)
        self.covar_ = np.dot(X.T, X)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed


class IncrementalZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, batch_size=None, bias=.1,
                 scale_by=1., copy=True):
        self.n_components = n_components
        self.batch_size = batch_size
        self.bias = bias
        self.scale_by = scale_by
        self.copy = copy
        self.scale_by = float(scale_by)
        self.mean_ = None
        self.covar_ = None
        self.n_samples_seen_ = 0.

    def fit(self, X, y=None):
        self.mean_ = None
        self.covar_ = None
        self.n_samples_seen_ = 0.
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size
        for batch in gen_batches(n_samples, self.batch_size_):
            self.partial_fit(X[batch])
        return self

    def partial_fit(self, X):
        self.components_ = None
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        X /= self.scale_by
        n_samples, n_features = X.shape
        batch_mean = np.mean(X, axis=0)
        # Doing this without subtracting mean results in numerical instability
        # will have to play some games to work around this
        if self.mean_ is None:
            X -= batch_mean
            batch_covar = np.dot(X.T, X)
            self.mean_ = batch_mean
            self.covar_ = batch_covar
            self.n_samples_seen_ += float(n_samples)
        else:
            prev_mean = self.mean_
            prev_sample_count = self.n_samples_seen_
            prev_scale = self.n_samples_seen_ / (self.n_samples_seen_
                                                 + n_samples)
            update_scale = n_samples / (self.n_samples_seen_ + n_samples)
            self.mean_ = self.mean_ * prev_scale + batch_mean * update_scale

            X -= batch_mean
            # All of this correction is to minimize numerical instability in
            # the dot product
            batch_covar = np.dot(X.T, X)
            batch_offset = (self.mean_ - batch_mean)
            batch_adjustment = np.dot(batch_offset[None].T, batch_offset[None])
            batch_covar += batch_adjustment * n_samples

            mean_offset = (self.mean_ - prev_mean)
            mean_adjustment = np.dot(mean_offset[None].T, mean_offset[None])
            self.covar_ += mean_adjustment * prev_sample_count

            self.covar_ += batch_covar
            self.n_samples_seen_ += n_samples

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        if self.components_ is None:
            U, S, VT = svd(self.covar_ / self.n_samples_seen_,
                           full_matrices=False)
            components = np.dot(VT.T * np.sqrt(1.0 / (S + self.bias)), VT)
            self.components_ = components[:self.n_components]
        X /= self.scale_by
        X -= self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed


if __name__ == "__main__":
    from numpy.testing import assert_almost_equal
    import matplotlib.pyplot as plt
    from scipy.misc import lena
    # scale_by is necessary otherwise float32 results are numerically unstable
    # scale_by is still not enough to totally eliminate the error in float32
    # for many, many iterations but it is very close
    X = lena().astype('float32')
    X_orig = np.copy(X)
    random_state = np.random.RandomState(1999)
    #X = random_state.rand(10000, 512).astype('float64') * 255.
    #X_orig = np.copy(X)
    scale_by = 255.
    zca = ZCA(scale_by=scale_by)
    for batch_size in [512, 511, 249, 128, 12, 2, 1]:
        print("Testing batch size %i" % batch_size)
        izca = IncrementalZCA(batch_size=batch_size, scale_by=scale_by)
        # Test that partial fit over subset has the same mean!
        zca.fit(X[:batch_size])
        izca.partial_fit(X[:batch_size])
        # Make sure data was not modified
        assert_almost_equal(X[:batch_size], X_orig[:batch_size])
        # Make sure single batch results match
        assert_almost_equal(zca.mean_, izca.mean_, decimal=3)
        assert_almost_equal(zca.covar_, izca.covar_, decimal=3)

        izca.fit(X[:100])
        izca.partial_fit(X[100:200])
        zca.fit(X[:200])
        # Make sure 2 batch results match
        assert_almost_equal(zca.mean_, izca.mean_, decimal=3)
        assert_almost_equal(zca.covar_, izca.covar_, decimal=3)
        # Make sure the input array is not modified
        assert_almost_equal(X, X_orig, decimal=3)
        X_zca = zca.fit_transform(X)
        X_izca = izca.fit_transform(X)
        # Make sure the input array is not modified
        assert_almost_equal(X, X_orig, decimal=3)
        # Make sure the means are equal
        assert_almost_equal(zca.mean_, izca.mean_, decimal=3)
        # Make sure the covariances are equal
        assert_almost_equal(zca.covar_, izca.covar_, decimal=3)
        # Make sure the components are equal
        assert_almost_equal(X_zca, X_izca, decimal=3)
    plt.imshow(X, cmap="gray")
    plt.title("Original")
    plt.figure()
    plt.imshow(X_zca, cmap="gray")
    plt.title("ZCA")
    plt.figure()
    plt.imshow(X_izca, cmap="gray")
    plt.title("IZCA")
    plt.figure()
    plt.matshow(zca.components_)
    plt.title("ZCA")
    plt.figure()
    plt.matshow(izca.components_)
    plt.title("IZCA")
    plt.show()
