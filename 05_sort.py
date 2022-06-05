import base64
import numpy as np
import os
import tensorflow as tf

tf.enable_eager_execution()
# sort, argsort, topK, Top-5 ACC

a = tf.random.shuffle(tf.range(5))
print('raw:', a)
# 返回排序后的值
print('sort:', tf.sort(a, direction='DESCENDING'))
# 返回排序值的索引
print('argsort:', tf.argsort(a, direction='DESCENDING'))

b = tf.random.uniform([3,3], maxval=10, dtype=tf.int32)
print('sort:', tf.sort(b))
print('argsort:', tf.argsort(b))

values, indices = tf.math.top_k(b, 2)
#print('top-k', tf.math.top_k(b, 2))
print(f'top-k values:{values}, indices:{indices}')

# top-k acc
prob = tf.convert_to_tensor([[0.1, 0.2, 0.7],[0.2, 0.7,0.1]])
target = tf.constant([2,0])

#k_b = tf.math.top_k(prob, 3).indices
#k_b = tf.transpose(k_b)
#print(f'k_b:{k_b}')
#target = tf.broadcast_to(target, [3,2])

#
def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.shape[0]
    print(f'batch_size:{batch_size}')

    pred = tf.math.top_k(output, max_k).indices
    pred = tf.transpose(pred)
    print(pred.shape)
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    res =[]
    for k in top_k:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.int32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k/batch_size)
        print(f'k:{k}, acc:{acc}')
        res.append(acc)
    return res

print(f'output:{prob}\ntarget:{target}')
print("accuracy:", accuracy(prob, target, top_k=(1,2,3)))

output = tf.random.normal([10,6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
acc = accuracy(output, target, top_k=range(1,7))
print(f'acc:{acc}')
