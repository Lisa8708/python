import tensorflow as tf
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import datetime
#
tf.enable_eager_execution()

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    # y = tf.one_hot(y, depth=100)
    return x, y

# Conv2D 与 Conv2DTranspose 刚好相反
# Conv2D: new_rows = math.floor((rows-kernel)/stride) + 1
# Conv2DTranspose: new_rows = (rows-1)*stride + kernel

def main():
    model = Sequential([ # 5 units of conv + max pooling
        # unit 1
        layers.Conv2D(64, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        # unit 2
        layers.Conv2D(128, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        # unit 3
        layers.Conv2D(256, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        # unit 4
        layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        # unit 5
        layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

        layers.Flatten(), # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dropout(0.5),
        layers.Dense(100, activation=None)
    ])

    model.build(input_shape=[None, 32, 32, 3])

    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True), #targets是one_hot编码 用categorical_crossentropy ，tagets是数字编码，用sparse_categorical_crossentropy
        metrics=['accuracy']
    )

    log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    (x, y), (x_test, y_test) = datasets.cifar100.load_data()
    print(x.shape, y.shape, x_test.shape, y_test.shape, int(tf.reduce_max(x)), int(tf.reduce_min(x)))
    # train_db = tf.data.Dataset.from_tensor_slices((x, y))
    # train_db = train_db.shuffle(10000).map(preprocess).batch(16)
    #
    # test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # test_db = test_db.shuffle(10000).map(preprocess).batch(16)
    #
    # # model.fit(train_db, epochs=20, validation_data=test_db, validation_freq=1, callbacks=[tensorboard_callback])

    x, x_test = x/255., x_test/255.
    model.fit(x, y, batch_size=16, epochs=5)
    model.evaluate(x_test, y_test, callbacks=[tensorboard_callback])

if __name__ == '__main__':
    main()