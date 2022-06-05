# ! -*-encoding:utf-8 -*-
import numpy as np
import tensorflow as tf
# random.normal, random.truncated_normal, gather, gather_nd, boolean_mask

tf.enable_eager_execution()
print(tf.__version__)

a = np.arange(5)
a1 = tf.convert_to_tensor(a)

# 可以优化的参数
b = tf.Variable(a)
b.trainable # True

print(tf.is_tensor(b)) # True
b.dtype

a1 = tf.ones([])
# <==>
print(a1.numpy())

#
tf.fill([2,3], 0) # equal tf.zeros([2,2])
tf.random.normal([2,2], mean=1, stddev=1)

# 性能优于normal
tf.random.truncated_normal([2,2], mean=0, stddev=1)
tf.random.uniform([2,2], minval=0, maxval=100)

tf.random.shuffle(a)

a = tf.random.normal([10, 784])

idx = tf.range(10)
idx = tf.random.shuffle(idx)
tf.gather(a, idx)


# 索引 与 切片
a = tf.random.normal([4,28,28,3])
a[1].shape # [28,28,3]
a[1][2][3][2] # equal to next rows
a[1,2,3,2]

a[0:2,:,:,:]
# 隔行采样
a[:,0:28:2,0:28:2,:] # [4,14,14,3]
a[:,::2,::2,:] # [4,14,14,3]
# ... 等价于所有的冒号
a[0, ...]
a[...,0].shape # [4,28,28]
a[0,...,2].shape # [28,28]
a[0,1,...,2].shape # [28]

# 逆序
a = tf.range(4)
a[::-1]  # [3,2,1,0]
# 逆序+隔行
a[::-2]  # [3,1]
a[2::-2] # [2,0]

#a = tf.range(10)
print(a[-2:]) # [8,9]

# gather 采样 4个班级的35个学生的8门课程得分
a = tf.random.normal([4,35,8], mean=80, stddev=10)

# axis取的维度，indices：取的维度的索引号
print(tf.gather(a, axis=0, indices=[2,3]).shape) #[2,35,8]
print(a[2:4].shape) #[2,25,8]

tf.gather(a, axis=0, indices=[2,1,3,0]) # [4,35,8]
tf.gather(a, axis=1, indices=[2,3,9,7,16]) # [4,5,8]

print('---------------多维度采样:gather, gather_nd ------------')
## 学生xx的yy课程
aa = tf.gather(a, axis=1, indices=[2,3,9,7,16])
aaa = tf.gather(aa, axis=2, indices=[3,5,7])

# 班级0的所有学生的所有课程
tf.gather_nd(a, [0]) # [35,8] a[0]
# 班级0学生1的所有课程
tf.gather_nd(a, [0,1]) # [8] a[0][1]
#
tf.gather_nd(a, [0,1,2]) # [] a[0][1][2]
tf.gather_nd(a, [[0,1,2]])  # [a[0][1][2]]

## 班级0学生0，班级1学生1 的8门成绩
print(tf.gather_nd(a, [[0,0], [1,1]])) # [2,8]
print(tf.gather_nd(a, [[0,0], [1,1], [2,2]])) # [3,8]

# 班级0学生0的课程0，班级1学生1的课程1，。。。
print(tf.gather_nd(a, [[0,0,0], [1,1,1], [2,2,2]])) # [3]
print(tf.gather_nd(a, [[[0,0,0], [1,1,1], [2,2,2]]])) # [1,3]

print("----------------------boolean_mask ------------")
# mask的维度要与向量对应的维度保持一致
print(tf.boolean_mask(a, mask=[True, True, False, False]).shape)
print(a.shape)
print(tf.boolean_mask(a, mask=[True, True, False, True, True, False, True, True], axis=2).shape)

a = tf.ones([2,3,4])
# 对应维度 (2,3)
print(tf.boolean_mask(a, mask=[[True, False, False], [False, True, True]]))
