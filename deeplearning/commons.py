import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn import datasets
import matplotlib.pyplot as plt
from keras.datasets import cifar10


# 正規化
def normalize(X):
    norm = np.linalg.norm(X, ord=2, axis=1)
    return X / norm[:, np.newaxis]


# 0 <= N <= 70000
def load_mnist(N=10000):
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')
    indices = np.random.permutation(range(70000))[:N]
    X = mnist.data[indices]
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]  # 1-of-K表現
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

    X_train, X_test = normalize(X_train), normalize(X_test)

    return X_train, X_test, Y_train, Y_test


def load_cifar(test_size=1000):
    (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()

    cifar_X = np.r_[cifar_X_1, cifar_X_2]
    cifar_y = np.r_[cifar_y_1, cifar_y_2]

    cifar_X = cifar_X.astype('float32') / 255
    cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

    train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y,
                                                        test_size=test_size,
                                                        random_state=42)

    return (train_X, test_X, train_y, test_y)
