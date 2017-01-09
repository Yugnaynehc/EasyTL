# !/usr/bin/python2
# Preliminary experiments on MNIST
# Reference: Distilling the Knowledge in a Neural Network

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import tensorlayer as tl
import time
import os


def big_net(X_placeholder, temperature):
    net = tl.layers.InputLayer(X_placeholder, name='big_input')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop1')
    net = tl.layers.DenseLayer(net, n_units=1200, act=tf.nn.relu, name='big_fc1')
    net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    net = tl.layers.DenseLayer(net, n_units=1200, act=tf.nn.relu, name='big_fc2')
    net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name='big_fc3')
    net = tl.layers.LambdaLayer(net, lambda x: x / temperature, name='big_scale')
    return net


def distil_net(X_placeholder, temperature):
    net = tl.layers.InputLayer(X_placeholder, name='distil_input')
    net = tl.layers.DenseLayer(net, n_units=1200, act=tf.nn.relu, name='distil_fc1')
    net = tl.layers.DenseLayer(net, n_units=1200, act=tf.nn.relu, name='distil_fc2')
    net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name='distil_fc3')
    net = tl.layers.LambdaLayer(net, lambda x: x / temperature, name='distil_scale')
    return net


X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(
    shape=(-1, 784))

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)
X_val = np.asarray(X_val, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.int64)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

batch_size = 128

# Read data
X = tf.placeholder(tf.float32, shape=[batch_size, 784], name='X')
y_ = tf.placeholder(tf.int64, shape=[batch_size, ], name='y_')


# Build the network
net_big = big_net(X, 10)
# output logits with temperature of big net
y_big = net_big.outputs

net_distil = distil_net(X, 10)
# output logits with temperature of distil net
y_distil = net_distil.outputs


# Compute loss of distil net
ce = tf.reduce_mean(tl.cost.cross_entropy(y_distil, y_big))
cost = ce

# Compute accuracy
correct_prediction = tf.equal(tf.argmax(y_distil, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Dropout switcher
dp_dict_big = tl.utils.dict_to_one(net_big.all_drop)

# Visualizing network
writer = tf.train.SummaryWriter('log/test_logs', sess.graph)
writer.flush()

# Define big model saving name (in the current path)
big_model_save_name = r'model_big.npz'

# Try to load big model
if os.path.exists(big_model_save_name):
    print('load big model')
    load_big_params = tl.files.load_npz(path='', name=big_model_save_name)
    tl.files.assign_params(sess, load_big_params, net_big)
else:
    print('Need to train big model first!')
    exit(1)

# Define distilling model saving name (in the current path)
distil_model_save_name = r'model_distil.npz'

# If saved distilling model esited, load it
if os.path.exists(distil_model_save_name):
    print('load existed distilling model')
    load_distil_params = tl.files.load_npz(path='', name=distil_model_save_name)
    tl.files.assign_params(sess, load_distil_params, net_distil)

# else train a model
else:
    print('train the distilling model')

    # Training settings
    n_epoch = 20
    lr = 1e-4
    print_freq = 5

    print('learning_rate: %f' % lr)
    print('batch_size: %d' % batch_size)

    train_params = net_distil.all_params
    train_op = tf.train.AdamOptimizer(
        lr, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False).minimize(cost, var_list=train_params)

    sess.run(tf.initialize_all_variables())

    # Print network params
    net_distil.print_params()
    net_distil.print_layers()

    print('strat training')
    for epoch in range(1, n_epoch + 1):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {X: X_train_a, y_: y_train_a}
            sess.run(train_op, feed_dict=feed_dict)

        if epoch == 1 or epoch % print_freq == 0:
            print('Epoch %d of %d took %fs' % (epoch, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                    X_train, y_train, batch_size, shuffle=False):
                feed_dict = {X: X_train_a, y_: y_train_a}
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print('train loss: %f' % (train_loss / n_batch))
            print('train acc: %f' % (train_acc / n_batch))

            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                    X_val, y_val, batch_size, shuffle=False):
                feed_dict = {X: X_val_a, y_: y_val_a}
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print('val loss: %f' % (val_loss / n_batch))
            print('val acc: %f' % (val_acc / n_batch))
    # Save model
    tl.files.save_npz(net_distil.all_params, name=distil_model_save_name)

print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(
        X_test, y_test, batch_size, shuffle=False):
    feed_dict = {X: X_test_a, y_: y_test_a}
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    test_loss += err
    test_acc += ac
    n_batch += 1
print('test loss: %f' % (test_loss / n_batch))
print('test acc: %f' % (test_acc / n_batch))
