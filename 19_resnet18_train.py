import tensorflow as tf
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import datetime
from resnet import resnet18
#
tf.enable_eager_execution()

def preprocess(x, y):
    x = 2*tf.cast(x, dtype=tf.float32)/255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    # y = tf.one_hot(y, depth=100)
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(10000).map(preprocess).batch(16)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(10000).map(preprocess).batch(16)


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    optimizer = optimizers.Adam(lr=1e-4)
    model.summary()

    for epoch in range(5):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x) #(16, 1, 1, 512)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.keras.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step%100 == 0:
                print(epoch, step, 'loss:', float(loss))


        total_num, correct_num = 0, 0
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.int32))
            correct_num += int(correct)
            total_num += int(x.shape[0])
        print('test acc:', correct_num/total_num)

if __name__ == '__main__':
    main()