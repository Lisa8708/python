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

# where：查询索引, scatter_nd：更新, meshgrid：生成网格数据

a = tf.random.normal([3,3])
mask = a>0
print(f'mask:{mask}')
print('boolean_mask:', tf.boolean_mask(a, mask))

# 返回满足条件的索引
indices = tf.where(mask)
print('indices:', indices) # 返回mask=True的索引
print('gather_nd:', tf.gather_nd(a, indices))

# where(cond, A, B)
a = tf.ones([3,3])
b = tf.zeros([3,3])
# True 返回a的值，False返回b的值
print('where:', tf.where(mask, a, b))

# update value  生成结果满足shape，不足用0填充
indices = tf.constant([[4],[3],[1],[7]])
updates = tf.constant([9,10,11,12])
shape = tf.constant([8])
print('scatter_nd:', tf.scatter_nd(indices, updates, shape))
# tf.Tensor([ 0 11  0 10  9  0  0 12], shape=(8,), dtype=int32)

indices = tf.constant([[0],[2]])
updates = tf.constant([
        [[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]],
        [[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]]])
shape = tf.constant([4,4,4])

print(tf.scatter_nd(indices, updates, shape))

print('--------------meshgrid:生成网格数据，生成坐标轴----------')
y = tf.linspace(-2.,2.,5) # [-2. -1.  0.  1.  2.]
x = tf.linspace(-2.,2.,5)
print(f"x:{x}\ny:{y}")

points_x, points_y = tf.meshgrid(x,y)
print(f"points_x:{points_x}\npoins_y:{points_y}") # shape=[5,5]
points = tf.stack([points_x, points_y], axis=2)
print(f'points:{points.shape}\n{points}')

print('----------------------- 等高线图 ---------------------')
def func(x):
    return tf.math.sin(x[...,0]) + tf.math.sin(x[...,1])

x = tf.linspace(0.,2*3.14, 500)
y = tf.linspace(0.,2*3.14, 500)
points_x, points_y = tf.meshgrid(x,y)
points = tf.stack([points_x, points_y], axis=2)

z = func(points)
print(f'points:{points.shape},z:{z.shape}')

from matplotlib import pyplot as plt
plt.figure('plot 2d func value')
# heat_map 热力图
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

# 等高线
plt.figure('plot 2d func contour')
plt.contour(points_x, points_y, z)
plt.colorbar()
plt.show()