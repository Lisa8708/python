# -*- encoding: utf-8 -*-

import sys
print(sys.version)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import datetime
import random
import numpy as np
import time
from PIL import Image
#
tf.enable_eager_execution()

tf.random.set_random_seed(22)
np.random.seed(22)

# is generative model

# denoising autoEncoder
# dropout autoEncoder
# adversarial autoEncoder

# VAE: variational AutoEncoder 求微分：学习到一个分布，进行sample，生成新的case
# KL散度divergence: 形容两个分布之间的差异
# KL(p/q) = sum(p(x)*log(p(x)/q(x)))

# 把多张image保存到一张image上
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))
    index = 0 #
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index+=1
    new_im.save(name)

h_dim = 20 # 降维后的维度
batch_size = 512
lr = 1e-3 #

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(10000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(x_test) # (x_test, y_test)
test_db = test_db.shuffle(10000).batch(batch_size)

print(x_train.shape, y_train.shape)

class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        # encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoders:
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        # [b, 784] -> [b, 10]
        h = self.encoder(inputs)

        # [b, 10] -> b[784]
        x_hat = self.decoder(h)
        return x_hat

model = AE()
model.build(input_shape=(None, 784))
model.summary()
optimizer = optimizers.Adam(lr)

for epoch in range(1000):
    for step, x in enumerate(train_db):
        # [b, 28, 28] -> [b, 784]
        #print(x)
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)
        grads = tape.gradient(rec_loss, model.trainable_variables)
        #for g in grads:
        #    tf.clip_by_norm(g, 15)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step%100 == 0:
            print(epoch, step, float(rec_loss))

    # evaluation
    x = next(iter(test_db))
    logits = model(tf.reshape(x, [-1, 784]))
    x_hat = tf.sigmoid(logits)
    # [b, 784] -> [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])

    # [b, 28, 28] -> [2b, 28, 28]
    #print(x, x_hat)
    x_concat = tf.concat([x, x_hat], axis=0)
    x_concat = x_hat
    x_concat = x_concat.numpy()*255
    x_concat = x_concat.astype(np.uint8)
    save_images(x_concat, name=f'ae_images/rec_epoch_{epoch}.png')