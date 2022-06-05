import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

# reshape, transpose, expand_dims, squeeze, @=matmul

print('------------reshape, transpose ----------------')
# reshape
a = tf.random.normal([4,28,28,3])
print(a.ndim, a.shape)

# 一个view只能有一个-1
print(tf.reshape(a, [4, 784, 3]).shape)
print(tf.reshape(a, [4,-1,3]).shape)

print(tf.reshape(a, [4, 784*3]).shape)
print(tf.reshape(a, [4, -1]).shape)

# transpose
#a = tf.random.normal((4,3,2,1))
print(a.shape)

# permernate
print(tf.transpose(a).shape)
print(tf.transpose(a, perm = [0,3,1,2]).shape) # 图片的pytorch格式
print(tf.transpose(a, perm = [0,1,3,2]).shape)

print('------------expand_dims, squeeze ----------------')
# expand dim
a = tf.random.normal([4,35,8])
print(tf.expand_dims(a, axis=0).shape) # [1,4,35,8]
# axis >=0 正序， <0 逆序
print(tf.expand_dims(a, axis=-2).shape) # [4,35,1,8]

# squeeze 减少维度 axis只能去掉shape=1的维度,
a = tf.zeros([1,2,1,1,3])
print(tf.squeeze(a).shape) #[2,3]
print(tf.squeeze(a, axis=0).shape)
print(tf.squeeze(a, axis=2).shape)

print('------------broadcast ----------------')
# broadcasting  增加维度，使得a，b的shape相同,节省内存空间 +已经包含了broadcast
a = tf.random.normal([4,3])
b = tf.random.normal([3])
print(a+[1,2,3])
print(a+[5])

# broadcastable
# False [4,32,32,3] [2,32,32,3]

# broadcast_to == expand_dims + tile
print(tf.broadcast_to(a, [2,4,3]).shape)
aa = tf.expand_dims(a, axis=0) # [1,4,3]
#print(aa.shape)
# tile 平铺用于同一维度上的复制，第一个维度复制2次，其他维度不复制
aaa = tf.tile(aa, [2,1,1])
print(aaa.shape)

# calculate tensor 计算  element_wise:对应维度值运算 \ matrix_wise \ dim_wise
b = tf.fill([2,2], 2.)
a = tf.ones([2,2])

print(a+b)
print(b%a)

print(tf.math.log(a))
print(tf.exp(a))

# log(2,8) = log(10,8)/log(10,2)
print(tf.math.log(8.)/tf.math.log(2.))

print('------------matmul:@ ----------------')
# @ = matmul  [4] [2,3]*[3,5] = [4,2,5]
a = tf.ones([4,2,3])
b = tf.fill([4,3,5], 2.0)

print((a@b).shape)
print(tf.matmul(a,b))

# y = W*x+b ==X@W+b
x = tf.ones([4,2])
W = tf.ones([2,1])
print(x@W+0.2)