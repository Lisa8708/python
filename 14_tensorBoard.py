from asyncio.proactor_events import constants
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import datetime, io
from matplotlib import pyplot as plt
# 只打印错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf2.0 以下版本需要启动动态视图
tf.enable_eager_execution()


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def image_grid(images):
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  for i in range(25):
    # Start next subplot.
    plt.subplot(5, 5, i + 1, title='name')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
  return figure

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

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.contrib.summary.create_file_writer(log_dir)

# get x from (x, y)
sample_img = next(iter(db))[0]
sample_img = tf.reshape(sample_img[0], [1,28,28,1])

with summary_writer.as_default():
    tf.contrib.summary.image("Training sample:", sample_img, step=0)

for step, (x,y) in enumerate(db):
    x = tf.reshape(x, [-1, 28*28])
    with tf.GradientTape() as tape:
        logits = model(x)
        #probs = tf.nn.softmax(logits)
        y_onehot = tf.one_hot(y, depth=10)
        loss_mse = tf.reduce_mean(tf.keras.losses.MSE(y_onehot, logits))
        loss_crs = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
    grads = tape.gradient(loss_crs, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step%100 == 0:
        print(f'epoch:{1}, step:{step}, loss_crs:{loss_crs}, loss_mse:{loss_mse}')
        with summary_writer.as_default():
            tf.contrib.summary.scalar("Training Loss:", float(loss_crs), step=step)

    if step%500 == 0:
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
        acc = total_correct/total
        print(f'step:{step}, test acc: {acc}')

        # print(x.shape)
        test_images = x[:25]
        test_images = tf.reshape(test_images, [-1, 28, 28, 1])
        with summary_writer.as_default():
            tf.contrib.summary.scalar('test acc', float(acc), step=step)
            tf.contrib.summary.image('test-onebyone-images', test_images, max_images=25, step=step) # tf2.0 max_outputs

            #test_images = tf.reshape(test_images, [-1, 28, 28])
            #figure = image_grid(test_images)
            #tf.contrib.summary.image('test images:', plot_to_image(figure), step=step)
