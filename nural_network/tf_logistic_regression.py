#!/usr/bin/env python
# -*- coding: utf-8 -*-
from commons.libraries import *

# define parameters
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))


# create model
# when you set 0 on shape[0], you can use the number you like of datasets.
x = tf.placeholder(tf.float32, shape=[None, 2])  # input
t = tf.placeholder(tf.float32, shape=[None, 1])  # label

y = tf.nn.sigmoid(tf.matmul(x, w) + b)

# error function
# tf.reduce_sum == np.sum
cross_entoropy = -tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))

# train
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entoropy)

# evaluate the outcome of test
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)


# practice
# OR-gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# you have to do calicurate in tensorflow session.
# following code is initializing variables and sequences defined above.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(200):
    sess.run(train_step, feed_dict={x: X, t: Y})

# you can know the result with eval method.
classified = correct_prediction.eval(session=sess, feed_dict={x: X, t: Y})
print(classified)

prob = y.eval(session=sess, feed_dict={x: X, t: Y})
print(prob)
