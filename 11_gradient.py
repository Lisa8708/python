from asyncio.proactor_events import constants
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import datasets

# 只打印错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf2.0 以下版本需要启动动态视图
tf.enable_eager_execution()

# 梯度的方向代表函数增大的方向，梯度的模代表函数增大的速率。
#with tf.GradientTape() as tape:
w = tf.Variable(1.)
b = tf.Variable(2.0)
x = tf.Variable(3.)
y = x*w

with tf.GradientTape(persistent=True) as tape:
    y2 = x*w + b
    tape.watch([w])
grad = tape.gradient(y2, [w])
print(f'grad:{grad}')

with tf.GradientTape() as t1:
    t1.watch(w)
    with tf.GradientTape() as t2:
        t2.watch([w, b])
        y = x*w + b
    dy_dw, dy_db = t2.gradient(y, [w,b])
d2y_dw2 = t1.gradient(dy_dw, [w])
print(f'dy_dw:{dy_dw}, dy_db:{dy_db}, d2y_dw2:{d2y_dw2}')

# sigmoid
a = tf.linspace(-5., 5, 10)
with tf.GradientTape() as tape:
    tape.watch([a])
    y = tf.sigmoid(a)

grads = tape.gradient(y, [a])
print(f"sigmoid grads:{grads}")

# tf.tanh
# relu: tf.nn.relu(x)
a = tf.linspace(-5., 5, 10)
with tf.GradientTape() as tape:
    tape.watch([a])
    y = tf.nn.relu(a)
grads = tape.gradient(y, [a])
print(f"relu grads:{grads}")

with tf.GradientTape() as tape:
    tape.watch([a])
    y = tf.nn.leaky_relu(a)
grads = tape.gradient(y, [a])
print(f"leaky_relu grads:{grads}")

## MSE gradient
x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.zeros([3])
y = tf.constant([2,0])

with tf.GradientTape() as tape:
    tape.watch([w,b])
    prob = tf.nn.softmax(x@w+b, axis=1)
    loss = tf.reduce_mean(tf.keras.losses.MSE(tf.one_hot(y, depth=3), prob))
grads = tape.gradient(loss, [w, b])
#print(grads)
print(f'mse: w:{grads[0]}\nb:{grads[1]}')

## softmax:exp(logit)/sum(exp(logit))
with tf.GradientTape() as tape:
    tape.watch([w,b])
    logits = x@w+b
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(y, depth=3), logits, from_logits=True))
    loss2 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(y, depth=3), tf.nn.softmax(logits), from_logits=False))

grads = tape.gradient(loss, [w, b])
print(f'cross_entropy: w:{grads[0]}\nb:{grads[1]}')

with tf.GradientTape() as tape:
    tape.watch([w,b])
    prob = tf.nn.softmax(x@w+b)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.one_hot(y, depth=3), prob, from_logits=False))

grads = tape.gradient(loss, [w, b])
print(f'cross_entropy: w:{grads[0]}\nb:{grads[1]}')
