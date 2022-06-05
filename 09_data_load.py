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

# data: boston housing, mnist, cifar10/100-image:10大类，100小类，data相同 label不同, imdb-review sentiment
# shuffle, repeat, batch, map

(x,y), (x_test, y_test) = keras.datasets.mnist.load_data()  # keras.datasets.cifar10.load_data()
print(f"x:{x.shape}, y:{y.shape}, x_test:{x_test.shape}, y_test:{y_test.shape}")
print(x.min(), x.max(), x.mean())
y_onehot = tf.one_hot(y, depth=10)

(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(f"x:{x.shape}, y:{y.shape}, x_test:{x_test.shape}, y_test:{y_test.shape}")
data_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))
data_set=data_set.shuffle(10000)

def preprocess(x, y):
    x=tf.cast(x, dtype=tf.float32)/255
    y=tf.cast(y,dtype=tf.int32)
    y=tf.one_hot(y, depth=10)
    return x, y

data_set = data_set.map(preprocess)
data_set2 = data_set.batch(32)
res = next(iter(data_set2))
print(f"res:{res[0].shape, res[1].shape}")

# 数据迭代5次
data_set4 = data_set2.repeat(5)

def prepare_mnist_features_and_labels(x, y):
    x=tf.cast(x, dtype=tf.float32)/255
    y=tf.cast(y,dtype=tf.int32)
    #y=tf.one_hot(y, depth=10)
    return x, y

def mnist_dataset():
    (x,y), (x_val, y_val) = keras.datasets.mnist.load_data()
    y=tf.one_hot(y, depth=10)
    y_val=tf.one_hot(y,depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x,y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.shuffle(60000).batch(64)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(prepare_mnist_features_and_labels)
    ds_val = ds_val.shuffle(10000).batch(64)

    return ds, ds_val
