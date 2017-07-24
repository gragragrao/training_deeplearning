import numpy as np
import tensorflow as tf
import keras as kr

# kr.models.Sequentials: to define the model of layers

model = kr.models.Sequential([
    kr.layers.Dense(input_dims=2, units=1),
    kr.layers.Activation('sigmoid')
])

# another way
'''
model = kr.models.Sequential()
model.add(kr.layers.Dense(input_dims=2, units=1))
model.add(kr.layers.Activation('sigmoid'))
'''
