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

x = tf.random.normal([4, 784])
net = tf.keras.layers.Dense(512)
out = net(x)
print("out:", out.shape)

##  net.build(input_shape(None,20))

x = tf.random.normal([100,10])
model = keras.Sequential(
    [keras.layers.Dense(5, activation='relu'),  # params: 10*5+5  w=shape(10,5), b=shape(5)
    keras.layers.Dense(2, activation='relu'),   # w:5*2+2, b:2
    keras.layers.Dense(1)]                      # w:2*1, b:1
)
model.build(input_shape=[None, 10])
model.summary()

for p in model.trainable_variables:
    print(p.name, p.shape)

# float tf.linspace(-2,2,5) 会报错
a = tf.linspace(-2.,2,5)
print("a:", a)
print('sigmoid:', tf.sigmoid(a))
print('softmax:', tf.nn.softmax(a))

# loss function
# mse:reduce_mean((y-out)^2),  loss/batch/cls_num
# cross entropy:,

# mse:
y = tf.constant([1,2,3,0,2])
y = tf.one_hot(y, depth=4)
y = tf.cast(y, dtype=tf.float32)

out = tf.random.normal([5,4])
loss1 = tf.reduce_mean(tf.square(y-out))
loss2 = tf.square(tf.norm(y-out))/(5*4)
loss3 = tf.reduce_mean(tf.keras.losses.MSE(y, out))
# 三种方法等价
print(loss1, loss2, loss3)

# entropy
# 熵：形容单个分布的信息量大小
# 熵越小，信息量越大，越不稳定。 entropy = -sum(p*logp), tf.math.log默认以e为底，
# 交叉熵:两个分布之间的信息的交流 。entropy = -sum(p*logq), q=p时，交叉熵loss最小
a = tf.fill([4], 0.25)
entropy = -a*tf.math.log(a)/tf.math.log(2.)
print(f'a entropy: {tf.reduce_sum(entropy)}')

# a:2.0, b:1.0
b = tf.constant([0.1, 0.05, 0.05, 0.8])
entropy2 = -b*tf.math.log(b)/tf.math.log(2.)
print(f'b entropy: {tf.reduce_sum(entropy2)}')

loss1 = tf.keras.losses.categorical_crossentropy([0,1,0,0], [0.25,0.25,0.25,0.25])
loss2 = tf.keras.losses.categorical_crossentropy([0,1,0,0], [0.05,0.75,0.05,0.15])
print(f"loss1:{loss1}, loss2:{loss2}")

loss3 = tf.keras.losses.BinaryCrossentropy()([1],[0.1])
loss4 = tf.keras.losses.binary_crossentropy([1],[0.1])
loss5 = tf.keras.losses.categorical_crossentropy([0,1], [0.9,0.1])
print(f"loss3:{loss3}, loss4:{loss4}, loss5:{loss5}")

#
loss1 = tf.keras.losses.categorical_crossentropy([0,1], [0.1, 0.9], from_logits=True)
loss2 = tf.losses.softmax_cross_entropy([0,1], [0.1, 0.9])
print(f"loss1:{loss1}, loss2:{loss2}")