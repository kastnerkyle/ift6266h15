import os
import zipfile
import fnmatch
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt


def load_color(reshaped=True, random_seed=123522):
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'train.zip'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'https://dl.dropboxusercontent.com/u/15378192/train.zip'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    data_dir = os.path.join(data_path, 'cvd')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        zf = zipfile.ZipFile(data_file)
        zf.extractall(data_dir)

    data_file = os.path.join(data_path, 'cvd_color.npy')
    label_file = os.path.join(data_path, 'cvd_color_labels.npy')
    if not os.path.exists(data_file):
        print('... loading data')
        cat_matches = []
        dog_matches = []
        for root, dirname, filenames in os.walk(data_dir):
            for filename in fnmatch.filter(filenames, 'cat*'):
                cat_matches.append(os.path.join(root, filename))
            for filename in fnmatch.filter(filenames, 'dog*'):
                dog_matches.append(os.path.join(root, filename))

        sort_key = lambda x: int(x.split('.')[-2])
        cat_matches = sorted(cat_matches, key=sort_key)
        dog_matches = sorted(dog_matches, key=sort_key)

        def square(X):
            return imresize(X, (48, 48))

        X_cat = np.asarray([square(mpimg.imread(f))
                            for f in cat_matches])
        y_cat = np.zeros((len(X_cat),))
        X_dog = np.asarray([square(mpimg.imread(f))
                            for f in dog_matches])
        y_dog = np.ones((len(X_dog),))
        X = np.concatenate((X_cat, X_dog), axis=0).astype('float32')
        y = np.concatenate((y_cat, y_dog), axis=0).astype('int32')
        np.save(data_file, X)
        np.save(label_file, y)
    else:
        X = np.load(data_file)
        y = np.load(label_file)

    random_state = np.random.RandomState(random_seed)
    idx = random_state.permutation(len(X))
    if reshaped:
        X_s = X[idx].reshape(len(X), -1)
        y_s = y[idx]
    else:
        X_s = X[idx]
        X_s = X.transpose(0, 3, 1, 2)
        y_s = y[idx]

    train_x = X_s[:20000]
    valid_x = X_s[20000:22500]
    test_x = X_s[22500:]
    train_y = y_s[:20000]
    valid_y = y_s[20000:22500]
    test_y = y_s[22500:]
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    valid_x = valid_x.astype('float32')
    valid_y = valid_y.astype('int32')
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_gray(random_seed=123522):
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'train.zip'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'https://dl.dropboxusercontent.com/u/15378192/train.zip'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    data_dir = os.path.join(data_path, 'cvd')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        zf = zipfile.ZipFile(data_file)
        zf.extractall(data_dir)

    data_file = os.path.join(data_path, 'cvd_gray.npy')
    label_file = os.path.join(data_path, 'cvd_gray_labels.npy')
    if not os.path.exists(data_file):
        print('... loading data')
        cat_matches = []
        dog_matches = []
        for root, dirname, filenames in os.walk(data_dir):
            for filename in fnmatch.filter(filenames, 'cat*'):
                cat_matches.append(os.path.join(root, filename))
            for filename in fnmatch.filter(filenames, 'dog*'):
                dog_matches.append(os.path.join(root, filename))

        sort_key = lambda x: int(x.split('.')[-2])
        cat_matches = sorted(cat_matches, key=sort_key)
        dog_matches = sorted(dog_matches, key=sort_key)

        def square_and_gray(X):
            # From Roland
            gray_consts = np.array([[0.299], [0.587], [0.144]])
            return imresize(X, (48, 48)).dot(gray_consts).squeeze()

        X_cat = np.asarray([square_and_gray(mpimg.imread(f))
                            for f in cat_matches])
        y_cat = np.zeros((len(X_cat),))
        X_dog = np.asarray([square_and_gray(mpimg.imread(f))
                            for f in dog_matches])
        y_dog = np.ones((len(X_dog),))
        X = np.concatenate((X_cat, X_dog), axis=0).astype('float32')
        y = np.concatenate((y_cat, y_dog), axis=0).astype('int32')
        np.save(data_file, X)
        np.save(label_file, y)
    else:
        X = np.load(data_file)
        y = np.load(label_file)

    random_state = np.random.RandomState(random_seed)
    idx = random_state.permutation(len(X))
    X_s = X[idx].reshape(len(X), -1)
    y_s = y[idx]

    train_x = X_s[:20000]
    valid_x = X_s[20000:22500]
    test_x = X_s[22500:]
    train_y = y_s[:20000]
    valid_y = y_s[20000:22500]
    test_y = y_s[22500:]
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    valid_x = valid_x.astype('float32')
    valid_y = valid_y.astype('int32')
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval

if __name__ == "__main__":
    train, valid, test = load_gray()
