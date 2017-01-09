# !/usr/bin/python2
# Original from:
# https://github.com/Yugnaynehc/tensorlayer/blob/master/example/tutorial_mnist.py

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import tensorlayer as tl
import time

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(
    shape=(-1, 28, 28, 1))

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)
X_val = np.asarray(X_val, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.int64)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)

print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_val.shape', X_val.shape)
print('y_val.shape', y_val.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)
print('X %s\t y %s' % (X_test.dtype, y_test.dtype))

sess = tf.InteractiveSession()

batch_size = 128

# Read data
X = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1], name='X')
y_ = tf.placeholder(tf.int64, shape=[batch_size, ], name='y_')


# Build the network
net = tl.layers.InputLayer(X, name='input_layer')
net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=[5, 5, 1, 32], strides=[
                            1, 1, 1, 1], padding='SAME', name='conv1')
net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[
                          1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool1')
net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=[5, 5, 32, 64], strides=[
                            1, 1, 1, 1], padding='SAME', name='conv2')
net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[
                          1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool2')

net = tl.layers.FlattenLayer(net, name='flat')
net = tl.layers.DropoutLayer(net, keep=0.5, name='drop3')
net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu, name='fc3')
net = tl.layers.DropoutLayer(net, keep=0.5, name='drop4')
net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name='fc4')

y = net.outputs

# Compute loss
ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
cost = ce

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training settings
n_epoch = 20
lr = 1e-4
print_freq = 1

train_params = net.all_params

train_op = tf.train.AdamOptimizer(
    lr, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False).minimize(cost, var_list=train_params)

sess.run(tf.initialize_all_variables())
net.print_params()
net.print_layers()

writer = tf.train.SummaryWriter('log/test_logs', sess.graph)

writer.flush()

print('learning_rate: %f' % lr)
print('batch_size: %d' % batch_size)

for epoch in range(1, n_epoch + 1):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(
            X_train, y_train, batch_size, shuffle=True):
        feed_dict = {X: X_train_a, y_: y_train_a}
        feed_dict.update(net.all_drop)
        sess.run(train_op, feed_dict=feed_dict)

    if epoch == 1 or epoch % print_freq == 0:
        print('Epoch %d of %d took %fs' % (epoch, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(net.all_drop)
            feed_dict = {X: X_train_a, y_: y_train_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            train_loss += err
            train_acc += ac
            n_batch += 1
        print('train loss: %f' % (train_loss / n_batch))
        print('train acc: %f' % (train_acc / n_batch))

        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(
                X_val, y_val, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(net.all_drop)
            feed_dict = {X: X_val_a, y_: y_val_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            val_loss += err
            val_acc += ac
            n_batch += 1
        print('val loss: %f' % (val_loss / n_batch))
        print('val acc: %f' % (val_acc / n_batch))

print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(
        X_test, y_test, batch_size, shuffle=True):
    dp_dict = tl.utils.dict_to_one(net.all_drop)
    feed_dict = {X: X_val_a, y_: y_val_a}
    feed_dict.update(dp_dict)
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    test_loss += err
    test_acc += ac
    n_batch += 1
print('test loss: %f' % (test_loss / n_batch))
print('test acc: %f' % (test_acc / n_batch))
