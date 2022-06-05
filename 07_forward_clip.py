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

# load_data
(x,y), _ = datasets.mnist.load_data()

# x:0~255 -> 0~1
x = tf.convert_to_tensor(x, dtype=tf.float32)/50.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)

# 数据创建batch
train_data = tf.compat.v1.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_data)
sample = next(train_iter)

def accuray(target, output):
    output = tf.argmax(output, axis=1)
    target = tf.argmax(target, axis=1)
    #print('output:', output)
    #print('target:', target)
    correct = tf.reduce_sum(tf.cast(tf.equal(target, output), dtype=tf.int32))
    return correct/target.shape[0]


def main():
    w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
    b1 = tf.Variable(tf.zeros([512]))
    w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
    b2 = tf.Variable(tf.zeros([256]))
    w3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))

    optimizer = optimizers.SGD(lr=0.1)
    for epoch in range(10):
        for step, (x, y) in enumerate(train_data):
            x = tf.reshape(x, (-1, 784))
            with tf.GradientTape() as tape:
                h1 = x@w1 + b1
                h1 = tf.nn.relu(h1)
                h2 = h1@w2 + b2
                h2 = tf.nn.relu(h2)
                output = h2@w3 + b3

                loss = tf.square(y-output)
                loss = tf.reduce_mean(loss)

                # compute gradient
                grads = tape.gradient(loss, [w1,b1,w2,b2,w3,b3])
                #print('========== before: =========')
                #for g in grads:
                #    print(tf.norm(g))

                # clip gradients 15=裁剪后的范数
                grads, _ = tf.clip_by_global_norm(grads, 15)
                #print('========== after: =========')
                #for g in grads:
                #    print(tf.norm(g))

                # upate w, b: w'=w-lr*grad
                optimizer.apply_gradients(zip(grads, [w1,b1,w2,b2,w3,b3]))

                acc = accuray(y, output)

            if step%100==0:
                print(f"epoch:{epoch}, step:{step}, loss:{loss}, acc:{acc}")

if __name__ == '__main__':
    main()
