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

image_size = 28*28
h_dim = 256 # 降维后的维度
z_dim = 10
batch_size = 512
lr = 1e-3 #

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
# 不归一 会导致loss=nan
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(10000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(x_test) # (x_test, y_test)
test_db = test_db.shuffle(10000).batch(batch_size)

print(x_train.shape, y_train.shape)

class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = layers.Dense(h_dim)
        self.fc2 = layers.Dense(z_dim) # get mean prediction
        self.fc3 = layers.Dense(z_dim)

        # Decoder
        self.fc4 = layers.Dense(h_dim)
        self.fc5 = layers.Dense(image_size)

    def encoder(self, x): #
        h = tf.nn.relu(self.fc1(x))
        # get mean prediction
        mu = self.fc2(h)
        # get variance prediction
        log_var = self.fc3(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = tf.exp(log_var)**0.5
        eps = tf.random.normal(std.shape)
        return mu + std * eps

    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        return out

    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)

    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None):
        # [b, 784] -> [b, z_dim], [b, z_dim]
        mu, log_var = self.encoder(inputs)
        # reparameterize
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


model = VAE()
model.build(input_shape=(batch_size, 784))  # 不能指定(None, 784), 需要明确指定大小
model.summary()
optimizer = optimizers.Adam(lr=1e-3)

for epoch in range(1000):
    for step, x in enumerate(train_db):
        # [b, 28, 28] -> [b, 784]
        #print(x.shape)
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)
            # rec_loss.shape (512, 784)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss)/batch_size
            ## 注意不能直接使用tf.reduce_mean()

            # kl-divergence N(mu, log_var) -> N(0, 1) please refer to
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            # kl_div.shape: (512, 10)
            kl_div = - 0.5 * (1. + log_var - tf.square(mu) - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div)/batch_size
            loss = rec_loss + 1. * kl_div

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step%100 == 0:
            print(epoch, step, 'loss:', float(loss), 'rec_loss:', float(rec_loss), 'kl_div:', float(kl_div))

    # evaluation: sample data
    z = tf.random.normal((batch_size, z_dim))
    logits = model.decoder(z)
    x_hat = tf.nn.sigmoid(logits)
    # [b, 784] -> [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy()*255.

    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, name=f'vae_images/sample_rec_epoch_{epoch}.png')

    # evaluation test_db
    x = next(iter(test_db))
    logits, mu, log_var = model(tf.reshape(x, [-1, 784]))
    x_hat = tf.nn.sigmoid(logits)
    # [b, 784] -> [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])

    # [b, 28, 28] -> [2b, 28, 28]
    #print(x, x_hat)
    x_concat = tf.concat([x, x_hat], axis=0)
    x_concat = x_hat
    x_concat = x_concat.numpy()*255.
    x_concat = x_concat.astype(np.uint8)
    save_images(x_concat, name=f'vae_images/test_rec_epoch_{epoch}.png')