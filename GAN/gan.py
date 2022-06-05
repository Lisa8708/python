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
import glob
import os
#from scipy.misc.pilutil import toimage

from dataset import make_anime_dataset
#
tf.enable_eager_execution()

tf.random.set_random_seed(22)
np.random.seed(22)

# GAN  ：Generator Discriminator
# KL divergence vs JS divergence = (KL(p|q) + KL(q/p))/2
# JS 散度的缺陷：当P、Q完全不重叠时，KL->无穷，JS=log2，无法优化，gradient vanish

# DCGAN: deconvolution gan == Transposed GAN 从小图片变大图片 4*4 -> 512*512, 效果比WGAN略优，但需要精心设计参数和结构，
# W-GAN：不需要精心设计参数和结构
    ### JS divergence -> EM distance (Earth Mover distance == Wasserstein Distance，仅解决training稳定的问题)
    ### bad: weight clipping;  good: gradient penalty
    ### WGAN-GP: gradient penalty

# GAN学习指南：从原理入门到制作生成Demo ：https://zhuanlan.zhihu.com/p/24767059

# Anime 日本动漫图片 96*96
#精灵宝可梦数据集
#链接：https://pan.baidu.com/s/1O-YBLBeqDpui_FhspnwY3g
#提取码：r1ze

#动漫头像数据集
#链接：https://pan.baidu.com/s/11u7k9O56hwsZv4iB2lqgCQ
#提取码：mqmp

def save_result(val_out, val_block_size, image_fn, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_fn)  # , mode=color_mode

def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                    labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                    labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, batch_z, batch_x, training):
    # 1. treat real image as real
    # 2. treat generated image as fake image
    fake_image = generator(batch_z, training)
    #print(fake_image.shape)
    d_fake_logits = discriminator(fake_image, training)
    d_real_logits = discriminator(batch_x, training)

    # 真的照片全部判定为真， 假的照片全部判定为假
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_real + d_loss_fake
    return loss

def g_loss_fn(generator, discriminator, batch_z, training):
    fake_image = generator(batch_z, training)
    d_fake_logits = discriminator(fake_image, training)
    loss = celoss_ones(d_fake_logits) # 虽然生成的照片为假，但希望判定为真
    return loss

class Genarator(keras.Model):
    def __init__(self):
        super(Genarator, self).__init__()
        ## 注意参数的设置
        # z:[b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]
        self.fc = layers.Dense(3*3*512)
        # Transposed convolution 将图片size变大
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        # [z, 100] -> [z, 3*3*512]
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x = tf.nn.tanh(x)
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b, 64, 64, 3] -> [b, 1]
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')

        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # [b, h, w, 3] -> [b, -1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        # [b, h, w, c] -> [b, -1]
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

def main():
    # hyper parameters
    z_dim = 100
    epochs=10000
    batch_size = 512
    learning_rate = 0.002
    training = True

    img_path = glob.glob(r'../data/faces/*.jpg')
    assert len(img_path) > 0

    # datasets: (512, 64, 64, 3), img_shape:(64, 64, 3)
    datasets, img_shape, _ = make_anime_dataset(img_path, batch_size)
    #print(datasets, img_shape)
    sample = next(iter(datasets))
    # (512, 64, 64, 3) 1.0 -1.0
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())

    datasets = datasets.repeat()
    db_iter = iter(datasets)

    generator = Genarator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        batch_z = tf.random.normal([batch_size, z_dim])
        batch_x = next(db_iter)

        # train
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch%100 == 0:
            print(f'epoch:{epoch}, d_loss={round(float(d_loss), 5)}, g_loss={round(float(g_loss), 5)}')

            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('images', f'gan_fake_image_{epoch}.png')
            #print(tf.reduce_max(fake_image), tf.reduce_min(fake_image))
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

if __name__ == "__main__":
    main()
