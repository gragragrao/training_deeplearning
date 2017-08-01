# -*- coding: utf-8 -*-
from commons import *

rng = np.random.RandomState(1234)
random_state = 42

# 学習パラメーター
X_train, X_test, Y_train, Y_test = load_mnist(N=70000)
y_test = np.argmax(Y_test, axis=1)

VIS_UNITS = X_train.shape[1]  # = 784
HIDDEN_UNITS = 500
LEARNING_RATE = 0.1
CORRUPTION_LEVEL = np.float32(0.3)  # ノイズ
N_EPOCHS = 30
N_BATCHES = 10


class Autoencoder:
    def __init__(self, vis_dim, hid_dim, W, function=lambda x: x):
        self.W = W
        self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name='a')
        self.b = tf.Variable(np.zeros(hid_dim).astype('float32'), name='b')
        self.function = function
        self.params = [self.W, self.a, self.b]

    def encode(self, x):
        u = tf.matmul(x, self.W) + self.b
        return self.function(u)

    def decode(self, x):
        u = tf.matmul(x, tf.transpose(self.W)) + self.a
        return self.function(u)

    def f_prop(self, x):
        y = self.encode(x)
        return self.decode(y)

    def reconst_error(self, x, noise):
        tilde_x = x * noise
        reconst_x = self.f_prop(tilde_x)
        error = -tf.reduce_mean(tf.reduce_sum(x * tf.log(reconst_x) + (1. - x) * tf.log(1. - reconst_x), axis=1))
        return error, reconst_x


class Dense:
    def __init__(self, in_dim, out_dim, function):
        self.W = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function
        self.params = [self.W, self.b]

        self.ae = Autoencoder(in_dim, out_dim, self.W, self.function)

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.function(u)
        return self.z

    def pretrain(self, x, noise):
        cost, reconst_x = self.ae.reconst_error(x, noise)
        return cost, reconst_x


layers = [
    Dense(VIS_UNITS, HIDDEN_UNITS, tf.nn.sigmoid),
    Dense(HIDDEN_UNITS, HIDDEN_UNITS, tf.nn.sigmoid),
    Dense(HIDDEN_UNITS, HIDDEN_UNITS, tf.nn.sigmoid),
    Dense(HIDDEN_UNITS, 10, tf.nn.softmax)
]


def sgd(cost, params, eps=np.float32(0.1)):
    g_params = tf.gradients(cost, params)

    updates = []
    for param, g_param in zip(params, g_params):
        if g_param is not None:
            updates.append(param.assign_add(-eps * g_param))
    return updates


X = np.copy(X_train)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

start_time = time.time()
for l, layer in enumerate(layers[:-1]):
    corruption_level = np.float(0.3)
    batch_size = 100
    n_batches = X.shape[0] // batch_size
    n_epochs = 20

    x = tf.placeholder(tf.float32)
    noise = tf.placeholder(tf.float32)

    cost, reconst_x = layer.pretrain(x, noise)
    params = layer.params
    train = sgd(cost, params)
    encode = layer.f_prop(x)

    for epoch in range(n_epochs):
        X = shuffle(X, random_state=random_state)
        err_all = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            _noise = rng.binomial(size=X[start:end].shape, n=1, p=1 - corruption_level)
            _, err = sess.run([train, cost], feed_dict={x: X[start:end], noise: _noise})
            err_all.append(err)

    X = sess.run(encode, feed_dict={x: X})

'''
自己符号化器を使って学習をする前に、重みの可視化をしておく。

    weight = sess.run(tf.transpose(model.W))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(weight[i].reshape((28, 28)), cmap='gray')
    fig.savefig('./images/autoencoding_weight.png')
'''

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])


def f_props(layers, x):
    params = []
    for layer in layers:
        x = layer.f_prop(x)
        params += layer.params
    return x, params


y, params = f_props(layers, x)

cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(y), axis=1))
updates = sgd(cost, params)
train = tf.group(*updates)
valid = tf.argmax(y, axis=1)

for epoch in range(n_epochs):
    X_train, Y_train = shuffle(X_train, Y_train, random_state=random_state)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train, feed_dict={x: X_train[start:end], t: Y_train[start:end]})
    pred_y = sess.run(valid, feed_dict={x: X_test})

during = time.time() - start_time
print('f1 score: ', f1_score(y_test, pred_y, average='macro'), ', time: ', during, 's')

sess.close()
