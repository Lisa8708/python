# -*- encoding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import numpy as np
from tqdm import tqdm
tf.enable_eager_execution()
tf.random.set_random_seed(22)
np.random.seed(22)

# mlp : multi-layer perceptron
# model: mlp
class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=100, activation='relu')
        self.fc = layers.Dense(units=10)
    def call(self, x):
        # [b, 784] -> [b, 784]
        x = self.flatten(x)
        #print(x.shape)
        # [b, 784] -> [b, 100]
        x = self.dense(x)
        # [b, 100] -> [b, 10]
        logits = self.fc(x)
        return logits

def main():
    # param:
    batch_size = 128
    #compute current learning rate
    learning_rate = 1e-4
    start_learning_rate = 1e-4
    decay_steps = 1000
    decay_size = 0.95
    epochs = 20

    # data
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
    train_x, test_x = train_x.astype(np.float32)/255., test_x.astype(np.float32)/255.
    train_y, test_y = tf.squeeze(train_y), tf.squeeze(test_y)
    train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_db = train_db.shuffle(10000).batch(batch_size)
    test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_db = test_db.shuffle(10000).batch(batch_size)

    sample = next(iter(train_db))
    print(f'train: {train_x.shape}, {train_y.shape}, sample: {sample[0].shape, sample[1].shape}')

    model = MLP()
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs): #
        global_step, loss = 0, 0.
        for step, (x, y) in tqdm(enumerate(train_db)):
            #global_step += step
            #learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, decay_steps, decay_size)
            #optimizer = optimizers.Adam(learning_rate=learning_rate)
            with tf.GradientTape() as tape:
                # x: [b, 28*28]
                logits = model(x)
                loss = losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            # 增加gradient clip mnist val acc: 0.9687 -> 0.9687# 耗时增加
            #for grad in grads:
            #    tf.clip_by_norm(grad, 15.)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            #if step%100 == 0:
            #    print(f'epoch:{epoch}, step:{step}, loss:{loss}')

        total_num, total_correct = 0., 0.
        for (x, y) in test_db:
            logits = model.predict(x)
            pred = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=1), dtype=tf.uint8)
            #print('y:', y)
            #print('pred:', pred)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
            #print('correct:', tf.cast(pred == y, dtype=tf.int32), correct)
            total_correct += correct.numpy()
            total_num += int(x.shape[0])
        print(f'eval epoch: {epoch}, loss:{loss}, acc: {round(total_correct/total_num, 4)}')
        #totoal_num:{total_num}, total_correct:{total_correct}')

if __name__ == '__main__':
    main()
