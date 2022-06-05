import base64
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import datasets

tf.enable_eager_execution()

# pad, tile
a = tf.reshape(tf.range(9), [3,3])

# [[0,0],[0,0]] -> 上下左右
print(tf.pad(a, [[0,0],[0,0]]))
print(tf.pad(a, [[1,0],[0,0]]))
print(tf.pad(a, [[1,1],[1,1]]))

#
b = tf.random.normal([4,28,28,3])
bb = tf.pad(b, [[0,0], [2,2], [2,2], [0,0]])
print(f'b pad:{bb.shape}')

# tile [1,2]--对应维度复制的次数 性能低于tf.broadcast_to
print('a tile:', tf.tile(a, [1,2]))
aa = tf.expand_dims(a, axis=0)
print('aa tile:', tf.tile(aa, [2,1,1]))

print('aa broadcast:', tf.broadcast_to(aa, [2,3,3]))
#print('broadcast:', tf.broadcast_to(a, [2,3,3]))

# 限幅 -裁剪
# clip_by_value,
a = tf.range(10)
# max(x, 2)
print("maximum:", tf.maximum(a, 2))
print("minimum:", tf.minimum(a, 2))

# min_value, max_value tf.minimum(tf.maximum(a, 2), 8)
print('clip:', tf.clip_by_value(a, 2, 8))

#relu: maximum(x,0)
a = a-5
print('relu:', tf.nn.relu(a))

#
a = tf.random.normal([2,2], mean=10)
print('a norm:', tf.norm(a))

# 限定norm=15, 保证梯度不变
aa = tf.clip_by_norm(a, 15)
print('aa norm:', tf.norm(aa))

# tf.clip_by_global_norm(grads, norm) 所有的w缩放的程度相同,norm为所有w的norm

(x,y),_ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32)/50.
