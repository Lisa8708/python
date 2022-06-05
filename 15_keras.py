from asyncio.proactor_events import constants
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import datetime, io
from matplotlib import pyplot as plt
# 只打印错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf2.0 以下版本需要启动动态视图
tf.enable_eager_execution()

# datasets, layers, losses, metrics

# 1. build a meter
acc_meter = metrics.accuracy()

# 2. update
acc_meter.update_state(y, pred)

# get average data

# clear buffer
acc_meter.reset_state()

# Compile, Fit,
network.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss = tf.losses.categorical_crossentropy(from_logits=True),
    metrics = ['accuracy']
)
network.fit(database, epochs=10)


## Sequential, layers.Layer, Model
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=(None, 28*28))
model.summary()

model.trainable_variables
model.call()

class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
        self.bias = self.add_variable('b', [output_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1=MyDense(28*28, 256)
        self.fc2=MyDense(256, 128)
        self.fc3=MyDense(128, 64)
        self.fc4=MyDense(64, 32)
        self.fc5=MyDense(32, 10)
    def call(self, inputs, training=None):
        x = self.fc1(inputs, activation=tf.nn.relu)
        x = self.fc2(x, activation=tf.nn.relu)
        x = self.fc3(x, activation=tf.nn.relu)
        x = self.fc4(x, activation=tf.nn.relu)
        x = self.fc5(x)
        return x

# save/load weights/entire model/saved_model
model.save_weights('./checkpoints/my_checkpoint')

model = create_model() # Sequential()
model.load_weights('./checkpoints/my_checkpoint')

# 保存网络全部结构，不需要重新构建
model.save('model.h5')
network = tf.keras.models.load_model('model.h5')

# saved_model  不同语言之间共同的协议，可以互相调用
tf.saved_model.save(model, 'saved_model/')
imported = tf.saved_model.load('saved_model/')
f = imported.signatures['serving_default']
print(f(x=tf.ones([1, 28, 28, 3])))