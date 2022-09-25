import gzip
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn import Knn
from urllib.request import urlretrieve


def load_mnist(root='./mnist'):
    # TODO Load the MNIST dataset

    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
    if not os.path.exists(root + "\\t10k-labels-idx1-ubyte"):
        # Why this : The load process can be terminated if an exception is generated.
        if os.path.exists(root):
            shutil.rmtree(root)
        os.mkdir(root)
        # 1. Download the MNIST dataset from
        #    http://yann.lecun.com/exdb/mnist/
        if not os.path.exists("download\\t10k-labels-idx1-ubyte.gz"):
            if os.path.exists("download"):
                shutil.rmtree("download")
            os.mkdir("download")
            urlretrieve("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                        "download\\train-images-idx3-ubyte.gz")
            urlretrieve("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                        "download\\train-labels-idx1-ubyte.gz")
            urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                        "download\\t10k-images-idx3-ubyte.gz")
            urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                        "download\\t10k-labels-idx1-ubyte.gz")
        open(root + "\\train-images-idx3-ubyte", "wb+").write(
            gzip.GzipFile("download\\train-images-idx3-ubyte.gz").read())
        open(root + "\\train-labels-idx1-ubyte", "wb+").write(
            gzip.GzipFile("download\\train-labels-idx1-ubyte.gz").read())
        open(root + "\\t10k-images-idx3-ubyte", "wb+").write(
            gzip.GzipFile("download\\t10k-images-idx3-ubyte.gz").read())
        open(root + "\\t10k-labels-idx1-ubyte", "wb+").write(
            gzip.GzipFile("download\\t10k-labels-idx1-ubyte.gz").read())
        shutil.rmtree("download")
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.
    return np.fromfile(root + "\\train-images-idx3-ubyte", np.array(6e4, 28, 28)), \
           np.fromfile(root + "\\train-labels-idx1-ubyte", np.array(6e4, )), \
           np.fromfile(root + "\\t10k-images-idx3-ubyte", np.array(1e4, 28, 28)), \
           np.fromfile(root + "\\t10k-labels-idx1-ubyte", np.array(1e4, ))


# Input:
# root: str, the directory of mnist

# Output:
# X_train: np.array, shape (6e4, 28, 28)
# y_train: np.array, shape (6e4,)
# X_test: np.array, shape (1e4, 28, 28)
# y_test: np.array, shape (1e4,)

# Hint:
# 1. Use np.fromfile to load the MNIST dataset(notice offset).
# 2. Use np.reshape to reshape the MNIST dataset.

# YOUR CODE HERE
# raise NotImplementedError
...


# End of todo


def main():
    X_train, y_train, X_test, y_test = load_mnist()
    knn = Knn()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
