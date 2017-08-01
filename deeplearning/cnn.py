# -*- coding: utf-8 -*-
from commons import *

rng = np.random.RandomState(1234)
random_state = 42

# 学習パラメーター
X_train, X_test, Y_train, Y_test = load_mnist(N=70000)
y_test = np.argmax(Y_test, axis=1)

N_EPOCHS = 70
BATCHE_SIZE = 100
VIS_UNITS = X_train.shape[0]


# 縦横28, チャネル数1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1, 1, 1, 1], padding='VALID'):
        # Xavier Initializer
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        h = np.sqrt(6 / (fan_in + fan_out))
        self.W = tf.Variable(rng.uniform(low=-h, high=h, size=filter_shape).astype('float32'))
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'))  # バイアスはフィルターごと
        self.function = function
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)


class Pooling:
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)


class Flatten:
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))


class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        h = np.sqrt(6 / (in_dim + out_dim))
        self.W = tf.Variable(rng.uniform(low=-h, high=h, size=(in_dim, out_dim)).astype('float32'))
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)


class Dropout:
    def __init__(self, keep_prob=0.6):
        self.keep_prob = keep_prob

    def f_prop(self, x):
        return tf.nn.dropout(x, self.keep_prob)


layers = [
    Conv((5, 5, 1, 40), tf.nn.relu),
    Pooling(),
    Conv((5, 5, 40, 80), tf.nn.relu),
    Pooling(strides=(1, 1, 1, 1)),
    Conv((4, 4, 80, 120), tf.nn.relu),
    Pooling(strides=(1, 1, 1, 1)),
    Conv((2, 2, 120, 150), tf.nn.relu),
    Pooling(ksize=(1, 1, 1, 1), strides=(1, 1, 1, 1)),
    Flatten(),
    Dense(2 * 2 * 150, 10, tf.nn.softmax)
]

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])


def f_props(layers, x):
    for layer in layers:
        x = layer.f_prop(x)
    return x


y = f_props(layers, x)

cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

valid = tf.argmax(y, 1)

n_epochs = N_EPOCHS
batch_size = BATCHE_SIZE
n_batches = VIS_UNITS // batch_size

start_time = time.time()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  # これを忘れないように！！
    for epoch in range(n_epochs):
        X_train, Y_train = shuffle(X_train, Y_train, random_state=random_state)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train, feed_dict={x: X_train[start:end], t: Y_train[start:end]})
        pred_y = sess.run(valid, feed_dict={x: X_test})

        print("今、{}回目が終わった。{}s経過".format(epoch + 1, time.time() - start_time))

print("f1_score: {}, time: {}s".format(f1_score(y_test, pred_y, average='macro'), time.time() - start_time))
