# -*- encoding: utf-8 -*-

import sys
print(sys.version)

import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras import datasets, metrics, layers, Sequential, losses, optimizers
import datetime
import random
import numpy as np
import time

#
tf.enable_eager_execution()

tf.random.set_random_seed(22)
np.random.seed(22)

# the most frequent words
total_word = 10000
max_seq_len = 80
embedding_len = 100
batch_size = 32

# data
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_word)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_seq_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_seq_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size=batch_size, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.shuffle(1000).batch(batch_size=batch_size, drop_remainder=True)

print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batch_size, units]), tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units]), tf.zeros([batch_size, units])]
        # embedding
        self.embedding = layers.Embedding(total_word, embedding_len, input_length=max_seq_len)

        # simple_rnn
        #self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        #self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.2)

        # LSTM
        #self.rnn_cell0 = layers.LSTMCell(units, dropout=0.5)
        #self.rnn_cell1 = layers.LSTMCell(units, dropout=0.5)

        # GRU
        self.rnn_cell0 = layers.GRUCell(units, dropout=0.5)
        self.rnn_cell1 = layers.GRUCell(units, dropout=0.5)

        # fc
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        :param inputs: [b, 80]
        :param training: None ==> training=True
        :return:
        """
        x = inputs
        x = self.embedding(x)
        # [b, 80, 100] ==> [b, 64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1): # word:[b, 100] b个句子的第i个单词
            # h1 = x*wxh + h0*whh
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        # out: [b, 64]
        return prob

def main():
    units = 32
    epochs = 4
    start = time.time()
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)
    end = time.time()
    print('total time:', end-start)
if __name__ == '__main__':
    main()