import tensorflow as tf
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import datetime
#
tf.enable_eager_execution()

x = tf.random.normal([2,4,4,3], mean=1, stddev=0.5)

net = layers.BatchNormalization(axis=3)
out = net(x, training=True)

print(net.variables)

for i in range(100):
    out = net(x, training=True)

print(net.variables)
optimizer = optimizers.Adam(lr=1e-4)

for i in range(10):
    with tf.GradientTape() as tape:
        out = net(x, training=True)
        loss = tf.reduce_mean(tf.pow(out, 2)) -1
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
