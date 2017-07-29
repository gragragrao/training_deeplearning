# -*- coding: utf-8 -*-
from commons import *

rng = np.random.RandomState(1234)
random_state = 42

# 学習パラメーター
HIDDEN_UNITS = 200
LEARNING_RATE = 0.02
N_EPOCHS = 30
N_BATCHES = 10

X_train, X_test, Y_train, Y_test = load_mnist(N=70000)

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])

hidden_units = HIDDEN_UNITS
W1 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(784, hidden_units)).astype('float32'))
W2 = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(hidden_units, 10)).astype('float32'))
b1 = tf.Variable(np.zeros(hidden_units).astype('float32'))
b2 = tf.Variable(np.zeros(10).astype('float32'))
params = [W1, b1, W2, b2]


def main(X_train, X_test, Y_train, Y_test, activation_function):
    u1 = tf.matmul(x, W1) + b1
    z1 = activation_function(u1)
    u2 = tf.matmul(z1, W2) + b2
    y = tf.nn.softmax(u2)

    # tf.log(0) によるNanを防ぐ
    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0))))

    gW1, gb1, gW2, gb2 = tf.gradients(cost, params)

    eps = LEARNING_RATE
    updates = [
        W1.assign_add(-eps * gW1),
        b1.assign_add(-eps * gb1),
        W2.assign_add(-eps * gW2),
        b2.assign_add(-eps * gb2),
    ]
    train = tf.group(*updates)

    valid = tf.argmax(y, axis=1)
    y_test = np.argmax(Y_test, axis=1)

    n_epochs = N_EPOCHS
    batch_size = N_BATCHES
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
        print('function: ', activation_function.__name__, ', f1 score: ',
              f1_score(y_test, pred_y, average='macro'), ', time: ', during, 's')


if __name__ == '__main__':
    main(X_train, X_test, Y_train, Y_test, tf.nn.sigmoid)
    main(X_train, X_test, Y_train, Y_test, tf.nn.tanh)
    main(X_train, X_test, Y_train, Y_test, tf.nn.relu)
