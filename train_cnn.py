"""
CNN (encoder performing unsupervised feature learning by mapping an input image to a learned fixed-length vector):

Image (625x1250)
  --> 3x3 receptive fields convolved with stride 1 as in VGGNet (this done twice)
    --> fixed-size output vector
      --> CNN(width=32)
        --> CNN(width=64)
          --> CNN(width=128)
            --> 2 fully-connected layers (size=1024) applying the rectified linear unit activation
              --> output vector (to be concatenated with 1st LSTM output vector)

"""

import tensorflow as tf
import numpy as np
from helpers.utils import load_pickle
from definitions import data_dir


def extract_data(path):
  print 'Unpickling {}'.format(path.split('/').pop())
  
  data = load_pickle(path)
  
  return data['data'], data['labels']


def conv_2d(x, dims=(3, 3), filters=32, strides=(1, 1),
           std=1e-3, padding='SAME', activation=tf.identity, scope='conv2d', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    s = x.get_shape().as_list()
    
    shape = list(dims) + [s[3], filters]
  
    W = tf.Variable(tf.random_normal(shape=shape, stddev=std), name='W')
    b = tf.Variable(tf.ones([filters]) * std, name='b')
    o = tf.nn.convolution(x, W, padding, strides=strides)
    o = o + b
    
    return activation(o)


def fully_connected(x, output_units=100, activation=tf.identity, std=1e-3, scope='fc', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    s = x.get_shape().as_list()
    
    shape = [s[1], output_units]
    
    W = tf.Variable(tf.random_normal(shape, stddev=std))
    b = tf.Variable(tf.ones([shape[1]]) * std)
    h = tf.matmul(x, W) + b

    return activation(h)


def get_batch(X, Y, N, n=32):
  inds = np.random.choice(range(N), size=n, replace=False)
  return X[inds], Y[inds]


if __name__ == '__main__':
  # Extract pickled train, validation, and test data
  X_train, Y_train = extract_data('{}/train.pkl'.format(data_dir))
  X_val, Y_val = extract_data('{}/validation.pkl'.format(data_dir))
  X_test, Y_test = extract_data('{}/test.pkl'.format(data_dir))

  print 'X_train.shape = {}, Y_train.shape = {}'.format(X_train.shape, Y_train.shape)
  # => X_train.shape = (6, 1250, 640, 3), Y_train.shape = (6,)
  
  # Define our params
  num_train_images = X_train.shape[0]  # => 6
  train_steps = num_train_images
  N = num_train_images
  Nbatch = 1
  print_step = 6
  W = X_train.shape[1]  # => 1250
  H = W
  C = 3  # What is C, and why 3?
  filters = 4  # Why?
  dims = [3, 3]  # is this the size of sliding window?
  strides = [1, 1]
  act = tf.nn.relu
  std = 1e-2
  num_classes = num_train_images  # each image should have its own unique label, so num_classes should equal num images?
  lr = 1e-2
  momentum = 0.9
  l2_coeff = 1.0
  
  x = tf.placeholder(shape=[None, W, H, C], dtype=tf.float32)
  y = tf.placeholder(shape=[None], dtype=tf.int32)  # wrong datatype --> currently dealing with dtype='|S8749'

  # 3 CNNs
  o = conv_2d(x, dims=dims, filters=filters, strides=strides, std=std, activation=act, scope='conv_1')  # width=32?
  o = conv_2d(o, dims=dims, filters=filters, strides=strides, std=std, activation=act, scope='conv_2')  # width=64??
  o = conv_2d(o, dims=dims, filters=filters, strides=srtrides, std=std, activation=act, scope='conv_3')  # width=128?

  shape = o.get_shape().as_list()
  o = tf.reshape(o, [-1, shape[1] * shape[2] * shape[3]])
  
  # 2 FCLs
  o = fully_connected(o, num_classes, activation=act, scope='fcl_1')
  o = fully_connected(o, num_classes, activation=act, scope='fcl_2')

  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=o, labels=y))

  var_list = tf.trainable_variables()

  l2_reg = 0
  for v in var_list:
    l2_reg += tf.reduce_mean(tf.square(v))

  l2_reg = (1.0 / len(var_list)) * l2_coeff * l2_reg
  loss = loss + l2_reg

  opt = tf.train.MomentumOptimizer(lr, momentum)
  train = opt.minimize(loss)
  init = tf.global_variables_initializer()
  
  train_hist = []
  val_hist = []
  
  sess = tf.Session()
  sess.run(init)
  
  # Why do this here before going into for-loop?
  xb, yb = get_batch(X_train, Y_train, N, n=Nbatch)

  for i in range(train_steps):
    xb, yb = get_batch(X_train, Y_train, N, n=Nbatch)
    l, _ = sess.run([loss, train], {x: xb, y: yb})
  
    if i % print_step == 0:
      xb, yb = get_batch(X_val, Y_val, N, n=Nbatch)
      lval = sess.run(loss, {x: xb, y: yb})
      
      print "iter: {} Train: {}, Val: {}".format(i, l, lval)
      
      train_hist.append(l)
      val_hist.append(lval)

  a = sess.run(o, {x: xb, y: yb})

  # What are these?
  print a[:2]
  print a.shape
  print a[0] / np.sum(a[0])