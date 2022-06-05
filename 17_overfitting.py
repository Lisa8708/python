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

#tf.split()
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val = tf.split(x, num_or_size_splits=[50000, 10000])
y_train, y_val = tf.split(y, num_or_size_splits=[50000, 10000])

train_db = tf.data.Dataset.from_tensor_slices((x,y))

# k-cross validation 略微有提升
network.fit(db_train_val, epochs=6, validation_split=0.1, validation_freq=2)

# 过拟合的判断：
# train_acc高， test_acc低
# 过拟合的处理方法：regularization l1, l2

# method 1:
l2_model = keras.models.Sequential([
    keras.layers.Dense(16,
        kernel_regularization=keras.regularizers.l2(0.001),
        activation =tf.nn.relu,
        input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16,
        kernel_regularization=keras.regularizers.l2(0.001),
        activation =tf.nn.relu),
    keras.layers.Dense(1, activation =tf.nn.sigmoid)
])

# method 2:
for step, (x, y) in enumerate(train_db):
    y_onehot = tf.one_hot(y, depth=10)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(losses.CategoricalCrossentropy(y_onehot, out, from_logits=True))
        loss_regularization = []
        for p in network.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
        loss = loss + 0.001*loss_regularization
    grads = tape.gradient(loss, network.trainable_variables)
    optimizers.apply_gradients(zip(grads, network.trainable_variables))

## Momentum 动量

## learning rate decay
optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)
optimizer = optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
optimizer = optimizers.Adam(learning_rate=0.01, beta1=0.1, beta2=0.1)

# 动态学习率
optimizer = SGD(learning_rate=0.2)
for epoch in range(100):
    optimizer.learning_rate = 0.2*(100-epoch)/100

# early stop

# dropout: train, test 过程不一致， train有dropout，test没有
model = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

for step, (x, y) in enumerate(train_db):
    with tf.GradientTape() as tape:
        x = tf.reshape(x, [-1, 28*28])
        out = model(x, training=True)

# test
out = model(x, training=False)

# Stochastic 符合某一分布
