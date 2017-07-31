from layers import conv_2d, conv_1d, fully_connected
import tensorflow as tf

def conv_text(x, num_filters=32, filter_dims=5, fc_size=1024,
               scope='conv_text', batch_size=4):
  s = x.get_shape().as_list()

  with tf.variable_scope(scope):
    # downsample image with stride [3, 3]
    a = conv_1d(x, dims=filter_dims, filters=num_filters, strides=3, std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv1')

    # no downsampling with stride [1, 1]
    a = conv_1d(a, filter_dims, filters=num_filters, strides=1, std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv2')

    num_filters = 2 * num_filters
    # downsample image with stride [2, 2]
    a = conv_1d(a, filter_dims, filters=num_filters, strides=1, std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv3')

    # no downsampling with stride [1, 1]
    a = conv_1d(a, filter_dims, filters=num_filters, strides=1, std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv4')

    num_filters = 2 * num_filters
    # downsample image with stride [2, 2]
    a = conv_1d(a, filter_dims, filters=num_filters, strides=2, std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv5')

    # no downsampling with stride [1, 1]
    a = conv_1d(a, filter_dims, filters=num_filters, strides=1, std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv6')

    # downsample image with stride [2, 2]
    num_filters = 32
    a = conv_1d(a, filter_dims, filters=num_filters, strides=2, std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv7')

    # Convert to vector with fullyconnected layer
    a = tf.reshape(a, shape=[batch_size, -1])

    a = fully_connected(a, output_units=fc_size, activation=tf.nn.relu,
                       std='xavier', scope='fc')

    print "output vector of conv_text is {}".format(a)
    return a

def conv_block(x, num_filters=32, filter_dims=[5, 5], fc_size=1024,
               scope='conv_block', batch_size=4):
  s = x.get_shape().as_list()

  with tf.variable_scope(scope):
    # downsample image with stride [3, 3]
    a = conv_2d(x, dims=[7, 7], filters=num_filters, strides=[3, 3], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv1')

    # no downsampling with stride [1, 1]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[1, 1], std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv2')

    num_filters = 2 * num_filters
    # downsample image with stride [2, 2]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[2, 2], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv3')

    # no downsampling with stride [1, 1]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[1, 1], std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv4')

    num_filters = 2 * num_filters
    # downsample image with stride [2, 2]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[2, 2], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv5')

    # no downsampling with stride [1, 1]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[1, 1], std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv6')

    # downsample image with stride [2, 2]
    num_filters = 32
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[2, 2], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv7')

    # Convert to vector with fullyconnected layer
    a = tf.reshape(a, shape=[batch_size, -1])

    a = fully_connected(a, output_units=fc_size, activation=tf.nn.relu,
                       std='xavier', scope='fc')

    print "output vector of conv_block is: {}".format(a)
    return a


def lstm_block(x, v, t, lstm_size=512, vocab_size=52, num_words=30, feed_previous=False,
               scope='lstm_block', reuse=False, batch_size=4):
  with tf.variable_scope(scope, reuse=reuse):
    with tf.variable_scope('lstm_1', reuse=reuse):
      lstm_first = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=reuse)
      state_first = lstm_first.zero_state(batch_size, tf.float32)

      o_1, state_first = lstm_first(x[:, 0, :], state_first)

      r = tf.concat([o_1, v, t], axis=1)

    with tf.variable_scope('lstm_2', reuse=reuse):
      lstm_second = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=reuse)
      state_second = lstm_second.zero_state(batch_size, tf.float32)

      o_2, state_second = lstm_second(r, state_second)

    o = fully_connected(o_2, output_units=vocab_size, std='xavier', activation=tf.identity, reuse=False, scope='lstm_fc')

  with tf.variable_scope(scope, reuse=True):
    # Teacher training, we feed in a list of words so dont need to feed back in
    # the output of the lstm
    outputs = []
    outputs.append(o)
    for i in range(num_words - 1):
      if not feed_previous:
        word = x[:, i + 1, :]
      else:
        word = tf.softmax(o)

      with tf.variable_scope('lstm_1', reuse=True):
        o, state_first = lstm_first(word, state_first)

      o = tf.concat([o, v,t],axis=1)

      with tf.variable_scope('lstm_2', reuse=True):
        o, state_second = lstm_second(o, state_second)

      o = fully_connected(o, output_units=vocab_size, std='xavier', activation=tf.identity, reuse=True, scope='lstm_fc')

      outputs.append(o)

  return outputs
