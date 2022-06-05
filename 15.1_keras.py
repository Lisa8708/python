from asyncio.proactor_events import constants
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses
import datetime
# 只打印错误信息
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf2.0 以下版本需要启动动态视图
tf.enable_eager_execution()

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
x, x_test = x/255.0, x_test/255.
print(x.shape, y.shape)

model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dropout(0.2),
    layers.Dense(10, activation = tf.nn.softmax)  #  == from_logits=False
])

model.compile(
    optimizer = optimizers.Adam(lr=0.01),
    loss = 'sparse_categorical_crossentropy',  # losses.CategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy']
)
model.fit(x, y, epochs=10)

model.evaluate(x_test, y_test)

pred=model.predict(x_test)
pred=tf.argmax(pred, axis=1)

pred[:5]