#!/usr/bin/env python
# -*- coding: utf-8 -*-
from commons.libraries import *

# kr.models.Sequentials: to define the model of layers

model = kr.models.Sequential([
    kr.layers.Dense(input_dim=2, units=1),
    kr.layers.Activation('sigmoid')
])

# another way
'''
model = kr.models.Sequential()
model.add(kr.layers.Dense(input_dims=2, units=1))
model.add(kr.layers.Activation('sigmoid'))
'''

# create model
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

model.fit(X, Y, epochs=200, batch_size=1)

classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print(classes)
print(prob)
