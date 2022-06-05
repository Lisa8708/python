from asyncio.proactor_events import constants
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses
import datetime, io
from matplotlib import pyplot as plt
# 只打印错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf2.0 以下版本需要启动动态视图
tf.enable_eager_execution()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

batch_size = 128

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(10000).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(batch_size)
db_iter = iter(db)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])

#model.build(input_shape=[None, 28*28])
#model.summary()
#optimizer = optimizers.Adam(lr=1e-3)

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
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x

network = MyModel()
network.compile(
    optimizer = optimizers.Adam(lr=0.01),
    loss = losses.CategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy'])

print('---------------- network ------------------')
network.build(input_shape=[None, 28*28])
network.summary()
network.load_weights('weights.ckpt')

print('---------------- fit ------------------')
#network.fit(db, epochs=5, validation_data=db_test, validation_freq=2, verbose=1)

print('---------------- evaluate ------------------')
network.evaluate(db_test)

print('---------------- predict ------------------')
sample=next(iter(db_test))
x = sample[0]
y = tf.argmax(sample[1], axis=1)
pred=tf.argmax(network.predict(x), axis=1)
print(f'y:{y}\npred:{pred}')

#network.save_weights('weights.ckpt')
