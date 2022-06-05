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

#sota acc=80%
def preprocess(x, y):
    x = 2*tf.cast(x, dtype=tf.float32)/255. - 1
    #x = tf.reshape(x, [-1, 32*32*3])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

batch_size = 128
(x, y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y)
y_test = tf.squeeze(y_test)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).shuffle(10000).batch(batch_size)

sample = next(iter(train_db))
print(f'batch: {sample[0].shape, sample[1].shape}')

class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
        self.bias = self.add_variable('b', [output_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel + self.bias
        return x

class MyNetWork(keras.Model):
    def __init__(self):
        super(MyNetWork, self).__init__()
        self.fc1 = MyDense(32*32*3, 256)
        self.dp1 = layers.Dropout(0.1)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [-1, 32*32*3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x

model = MyNetWork()
model.compile(
    optimizer = optimizers.Adam(lr=0.001),
    loss = losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_db, epochs=5, validation_data=test_db, validation_freq=1)

model.evaluate(test_db)
model.save_weights('output_model/cifar10/weights.ckpt')

#model.load_weight('cifar10/weights.ckpt')