from asyncio.proactor_events import constants
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import datetime
# 只打印错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf2.0 以下版本需要启动动态视图
tf.enable_eager_execution()


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

batch_size = 128

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(10000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(batch_size)

db_iter = iter(db)
sample = next(db_iter)

print(f'batch:{batch_size}, shape:{sample[0].shape, sample[1].shape}')

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])

model.build(input_shape=[None, 28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)

acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

def main():
    for epoch in range(10):
        for step, (x,y) in enumerate(db):
            x = tf.reshape(x, [-1, 28*28])
            with tf.GradientTape() as tape:
                logits = model(x)
                #probs = tf.nn.softmax(logits)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.keras.losses.MSE(y_onehot, logits))
                loss_crs = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
                loss_meter.update_state(loss_crs)
            grads = tape.gradient(loss_crs, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step%100 == 0:
                print(f'epoch:{epoch}, step:{step}, loss_crs:{loss_crs}, loss_mse:{loss_mse}, loss_meter:{loss_meter}')
                loss_meter.reset_state()

        total, total_correct = 0., 0.
        for x,y in db_test:
            x = tf.reshape(x, [-1, 28*28])
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis = 1), dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total += int(x.shape[0])

            acc_meter.update_state(y, pred)
        acc = total_correct/total

        print(f'test acc: {acc}, acc_meter:{acc_meter}')

    with summary_writer.as_default():
        tf.summary.scalar('loss', float(loss), step=epoch)
        tf.summary.scalar('accuracy', float(train_accuracy), step=epoch)

if __name__ == '__main__':
    main()