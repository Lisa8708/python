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

x = tf.random.normal([1,3])
w = tf.ones([3,1])
b = tf.ones([1])
y = tf.constant([1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    logits = tf.sigmoid(x@w + b)
    loss = tf.reduce_mean(tf.keras.losses.MSE(y, logits))

grads = tape.gradient(loss, [w,b])
print(f'sigmoid mse: w:{grads[0]}\nb:{grads[1]}')

x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.zeros([3])
y=tf.constant([2,0])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x@w+b, axis=1)
    loss = tf.reduce_mean(tf.keras.losses.MSE(tf.one_hot(y, depth=3), prob))

grads = tape.gradient(loss, [w,b])
print(f'softmax mse: w:{grads[0]}\nb:{grads[1]}')

## 链式法则 dy_dx = dy_du*du_dx
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1, b1, w2, b2])
    y1 = x*w1 + b1
    y2 = y1*w2 + b2

dy2_dy1 = tape.gradient(y2, [y1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]
dy2_dw2 = tape.gradient(y2, [w2])[0]

def himmeblau(x):
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)

X, Y = np.meshgrid(x,y)
Z = himmeblau([X,Y])

from matplotlib import pyplot as plt
fig = plt.figure('himmeblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#plt.show()


x = tf.constant([-4., 0.])
lr = 1e-2
# gradient
for step in range(50):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = himmeblau(x)
    grads = tape.gradient(y, [x])[0]
    x -= lr*grads
    print(f'step:{step}, x:{x}, f(x):{y}')
