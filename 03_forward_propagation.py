import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# from_tensor_slices, iter, next, tf.Variable 可更新,
# tf.GradientTape, w1.assign_sub()

print(tf.__version__)
# 只打印错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf2.0 以下版本需要启动动态视图
tf.enable_eager_execution()

#print(datasets.mnist.load_data())
(x,y), (x_test, y_test) = datasets.mnist.load_data()
# x:0~255 -> 0~1
x = tf.convert_to_tensor(x, dtype=tf.float32)/255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)/255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

# 数据创建batch
train_data = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_data)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

# (b,784) -> (b,256) -> (b,128) -> (b,10)， #stddev 初始值会影响收敛的速度
w1= tf.Variable(tf.random.truncated_normal([784,256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-2

for epoch in range(50):
    for step, (x,y) in enumerate(train_data):
        x = tf.reshape(x, [-1, 28*28])
        # comput gradient
        with tf.GradientTape() as tape:
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

            # compute loss
            y_onehot = tf.one_hot(y, depth=10)
            # mse=sum(mean(y-out)^2)
            loss = tf.square(y_onehot-out)
            loss = tf.reduce_mean(loss)
        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        #print(f"grads:{len(grads)}")
        # w1 = w1 - lr*wl_grad
        # 原地更新，数据类型不变  ---等价于 ==> w1 = w1 - lr*grads[0]，但数据类型变化
        # grads = optimizer.apply
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
        w2.assign_sub(lr*grads[2])
        b2.assign_sub(lr*grads[3])
        w3.assign_sub(lr*grads[4])
        b3.assign_sub(lr*grads[5])

        # loss:nan 梯度爆炸或弥散, 设置参数的stddev
        if step%100==0:
            print(f"epoch:{epoch}, step:{step}, loss:{loss}")

    # test/ eval
    total_num, total_correct = 0, 0
    for step, (x,y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 28*28])
        h1 = tf.nn.relu(x@w1+b1)
        h2 = tf.nn.relu(h1@w2+b2)
        logit = h2@w3+b3

        #print("logit:", logit)
        prob = tf.nn.softmax(logit, axis=1)
        pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
        #print("prob:", prob)
        #print("pred:", pred)
        #print("y:", y)
        correct = tf.cast(tf.equal(y, pred), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_num += int(x.shape[0])
    print(total_correct, total_num)
    accuracy = total_correct / total_num
    print(f"epoch:{epoch}, train loss:{loss} test acc:{accuracy}")
