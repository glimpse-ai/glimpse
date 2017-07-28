import tensorflow as tf
import numpy as np


def conv_2d(x, dims=[3, 3], filters=32, strides=[1, 1],
           std=1e-3, padding='SAME', activation=tf.identity, scope='conv2d', reuse=False):
  """
  args:
      x, (tf tensor), tensor with shape (batch,width,height,channels)
      dims, (list), size of convolution filters
      filters, (int), number of filters used
      strides, (list), number of steps convolutions slide
      std, (float/string), std of weight initialization, 'xavier' for xavier
          initialization
      padding, (string), 'SAME' or 'VALID' determines if input should be padded
          to keep output dimensions the same or not
      activation, (tf function), tensorflow activation function, e.g. tf.nn.relu
      scope, (string), scope under which to store variables
      reuse, (boolean), whether we want to reuse variables that have already
          been created (i.e. reuse an earilier layer)

  returns:
      a, (tf tensor), the output of the convolution layer, has size
          (batch, new_width , new_height , filters)
  """
  with tf.variable_scope(scope, reuse=reuse):
    s = x.get_shape().as_list()

    shape = dims + [s[3], filters]

    if std == 'xavier':
      std = np.sqrt(2.0 / (s[1] * s[2] * s[3]))

    W = tf.Variable(tf.random_normal(shape=shape, stddev=std), name='W')

    b = tf.Variable(tf.ones([filters]) * std, name='b')

    o = tf.nn.convolution(x, W, padding, strides=strides)

    o = o + b

    a = activation(o, name='a')

    return a


def fully_connected(x, output_units=100, activation=tf.identity, std=1e-3,
                   scope='fc', reuse=False):
  """
  args:
      x, (tf tensor), tensor with shape (batch,width,height,channels)
      std, (float/string), std of weight initialization, 'xavier' for xavier
          initialization
      output_units,(int), number of output units for the layer
      activation, (tf function), tensorflow activation function, e.g. tf.nn.relu
      scope, (string), scope under which to store variables
      reuse, (boolean), whether we want to reuse variables that have already
          been created (i.e. reuse an earilier layer)

  returns:
      a, (tf tensor), the output of the fully_connected layer, has size
          (batch, output_units)
  """
  with tf.variable_scope(scope, reuse=reuse):
    s = x.get_shape().as_list()

    shape = [s[1], output_units]

    if std == 'xavier':
      std = np.sqrt(2.0 / shape[0])

    W = tf.get_variable('W', shape=shape, initializer=tf.random_normal_initializer(0.0, std))
    b = tf.get_variable("b", shape=shape[1], initializer=tf.random_normal_initializer(0.0, std))

    h = tf.matmul(x, W) + b
    a = activation(h, name='a')
    return a
