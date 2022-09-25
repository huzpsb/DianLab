import gzip
import os
import shutil
import time
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np

from knn import Knn


def to_bin(img, threshold):
    a = img - np.array([[threshold]])
    b = np.floor(a / np.array([[256]]))
    c = b + np.array([[1]], dtype=np.int16)
    bfilter = c.astype('uint8')
    result = bfilter * np.array([[255]], dtype=np.uint8)
    return result


def load_mnist(root='./mnist'):
    # Load the MNIST dataset
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
    train_labels = np.fromfile(root + "\\train-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    train_length = len(train_labels)
    train_data = to_bin(np.fromfile(root + "\\train-images-idx3-ubyte", dtype=np.uint8, offset=16) \
                        .reshape(train_length, 784), 100)

    test_labels = np.fromfile(root + "\\t10k-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    test_length = len(test_labels)
    test_data = to_bin(np.fromfile(root + "\\t10k-images-idx3-ubyte", dtype=np.uint8, offset=16) \
                       .reshape(test_length, 784), 100)
    return train_data, train_labels, test_data, test_labels


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

def display_image(image, title):
    image = image.reshape(28, 28, 1).squeeze()
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=plt.cm.gray_r)


def main():
    t1 = time.time()
    X_train, y_train, X_test, y_test = load_mnist()
    # display_image(X_train[2], y_train[2])
    knn = Knn()
    knn.fit(X_train, y_train)
    correct = 0
    wrong = 0
    #
    for index in range(0, len(y_test)):
        t = time.time()
        predicted = knn.predict(X_test[index])
        print('Train :%s ms' % ((time.time() - t) * 1000))
        expected = y_test[index]
        if predicted == expected:
            correct += 1
        else:
            wrong += 1
        # print("predicted: ", predicted, " ,expected: ", expected)
        # print("rate: ", correct / (correct + wrong))
        if correct % 10 == 0:
            print("rate: ", correct / (correct + wrong))
    #
    print("final-rate: ", correct / (correct + wrong))
    #    y_pred = knn.predict(X_test)
    #    correct = sum((y_test - y_pred) == 0)

    #    print('==> correct:', correct)
    #    print('==> total:', len(X_test))
    #    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(knn.predict(X_test[i]))
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
