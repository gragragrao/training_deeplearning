import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn import datasets
import matplotlib.pyplot as plt


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
