import numpy as np
import os
import tensorflow as tf

# concat, stack, unstack, split, norm-L1,L2
# reduce_mean, argmax, tf.unique, tf.gather

tf.enable_eager_execution()

print('------------- concat: 除拼接维度外的其他维度相同 -------------')
a = tf.ones([4,35,8])
b = tf.ones([2,35,8])
print(tf.concat([a,b], axis=0).shape)

a = tf.ones([4,32,8])
b = tf.ones([4,3,8])
print(tf.concat([a,b], axis=1).shape)

print('------------- stack: 维度完全相同, 可逆 -------------')
# stack 增加新的维度
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.stack([a,b], axis=0)
print(c.shape) # [2,4,35,8]
print(tf.stack([a,b], axis=3).shape) # [4,35,8,2]

# unstack return list
aa, bb = tf.unstack(c, axis=0)
print(aa.shape, bb.shape)

# split 拆分为两个
res = tf.split(c, axis=3, num_or_size_splits=2)
print(len(res), res[0].shape)

res = tf.split(c, axis=3, num_or_size_splits=[2,2,4])
print(len(res), [r.shape for r in res])

print('------------- norm ---------------')
## statistics
# Norm L1, L2
a = tf.ones([2,2])
print(f'a:{a}')
print(f'L2 norm:{tf.norm(a)}')
norm2 = tf.sqrt(tf.reduce_sum(tf.square(a)))
print(f'L2 norm2: {norm2}')

# ord范数，axis:在哪个维度上进行范数计算
print(f'L2 Norm: {tf.norm(a, ord=2, axis=1)}')
print(f'L1 Norm: {tf.norm(a, ord=1)}')
print(f'L1 Norm: {tf.norm(a, ord=1, axis=0)}')

b = tf.ones([4,28,28,3])
print(f'norm:{tf.norm(b)}')
print(f'norm2:{tf.sqrt(tf.reduce_sum(tf.square(b)))}')

print('-------------reduce_-----------')
a = tf.random.normal([4,10])
print(tf.reduce_mean(a), tf.reduce_min(a), tf.reduce_max(a))

print(tf.reduce_mean(a, axis=1),
    tf.reduce_min(a, axis=1),
    tf.reduce_max(a, axis=1))

print('------------ argmax, argmin ------------')
# 默认axis=0
print(tf.argmax(a)) #[10]
print(tf.argmax(a, axis = 1)) # [4]

print('--------------tf.accuracy------------')
# tf.equal
a = tf.constant([1,2,3,2,5])
b = tf.range(1,6)
print(tf.equal(a,b))

a = tf.convert_to_tensor([[0.1, 0.2, 0.7],[0.9, 0.05,0.05]])
pred = tf.cast(tf.argmax(a, axis=1), dtype=tf.int32)
y = tf.convert_to_tensor([2,1])
correct = tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.int32))

acc = correct/y.shape
print('acc:', acc)

print('-------------unique------------')
a = tf.constant([4,2,2,4,3])
print(tf.unique(a))

# 使用gather进行还原
unique, idx = tf.unique(a).y, tf.unique(a).idx
print(tf.gather(unique, idx))
