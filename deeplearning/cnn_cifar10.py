# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn import datasets
import matplotlib.pyplot as plt
from keras.datasets import cifar10

from load_cifar10 import *


# load cifar10
maybe_download_and_extract()
