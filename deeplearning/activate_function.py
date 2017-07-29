import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


def score_prediction(true_array, predicted_array):
    return np.sum(true_array == predicted_array) * 100 / len(true_array)


# 正規化する（これあるのとないのでスコアが違いすぎる笑）
def normalize(X):
    norm = np.linalg.norm(X, ord=2, axis=1)
    return X / norm[:, np.newaxis]


# MNISTのデータを読み込んでおく
from sklearn import datasets
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

rng = np.random.RandomState(1234)
random_state = 42

# 数が多いので制限する。dataとtargetはそれぞれ70000ずつある。
N = 70000

indices = np.random.permutation(range(70000))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]  # 1-of-K表現
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

X_train, X_test = normalize(X_train), normalize(X_test)
y_test = np.argmax(Y_test, axis=1)

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])

hidden_units = 200
W1 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(784, hidden_units)).astype('float32'))
W2 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(hidden_units, 10)).astype('float32'))
b1 = tf.Variable(np.zeros(hidden_units).astype('float32'))
b2 = tf.Variable(np.zeros(10).astype('float32'))
params = [W1, b1, W2, b2]

for activation_function in [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu]:
    u1 = tf.matmul(x, W1) + b1
    z1 = activation_function(u1)
    u2 = tf.matmul(z1, W2) + b2
    y = tf.nn.softmax(u2)

    # tf.log(0) によるNanを防ぐ
    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0))))

    gW1, gb1, gW2, gb2 = tf.gradients(cost, params)

    esp = 0.02
    updates = [
        W1.assign_add(-esp * gW1),
        b1.assign_add(-esp * gb1),
        W2.assign_add(-esp * gW2),
        b2.assign_add(-esp * gb2),
    ]
    train = tf.group(*updates)

    valid = tf.argmax(y, axis=1)

    n_epochs = 30
    batch_size = 10
    n_batches = X_train.shape[0] // batch_size

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            X_train, Y_train = shuffle(X_train, Y_train, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: X_train[start:end], t: Y_train[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: X_test, t: Y_test})
        during = time.time() - start_time
        print('function: ', activation_function.__name__, ', f1 score: ', score_prediction(y_test, pred_y), ', time: ', during, 's')
