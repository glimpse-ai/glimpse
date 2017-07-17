from glimpse.neural_net import NeuralNet
import tensorflow as tf
import numpy as np
from helpers.utils import load_pickle
from helpers.definitions import data_dir


class CNN(NeuralNet):
  
  def __init__(self, conv_layers=0, fcl_layers=0):
    NeuralNet.__init__(self)
    
    self.conv_layers = conv_layers
    self.fcl_layers = fcl_layers
    
    # Init empty attrs
    self.X_train = None
    self.Y_train = None
    self.X_val = None
    self.Y_val = None
    self.X_test = None
    self.Y_test = None

    # Extract pickled train, validation, and test data
    self.extract_data()
    
    # Define non-static params left out of params.json
    self.define_dynamic_params()
    
    # Establish weight placeholders
    self.x = tf.placeholder(shape=[None, self.params.W, self.params.H, self.params.C], dtype=tf.float32)
    self.y = tf.placeholder(shape=[None], dtype=tf.int32)  # wrong datatype --> currently dealing with dtype='|S8749'
    self.o = self.x
    
    # Construct the network
    self.build_nn()
    
  def extract_data(self):
    for set in ['train', 'val', 'test']:
      print 'Unpickling {} set'.format(set)
      data = load_pickle('{}/{}.pkl'.format(data_dir, set))
      
      x, y = data['data'], data['labels']
      
      setattr(self, 'X_{}'.format(set), x)
      setattr(self, 'Y_{}'.format(set), y)
    
    print 'X_train.shape = {}, Y_train.shape = {}'.format(self.X_train.shape, self.Y_train.shape)
    
  def define_dynamic_params(self):
    num_train_images = self.X_train.shape[0]  # => 6 right now
    W = self.X_train.shape[1]  # => 1250
    
    dyn_params = {
      'num_train_images': num_train_images,
      'train_steps': num_train_images,
      'N': num_train_images,
      'W': W,
      'H': W,
      'act': tf.nn.relu,
      'num_classes': num_train_images  # each image should have its own unique label... so this should equal num_images?
    }
    
    [setattr(self.params, k, v) for k, v in dyn_params.iteritems()]
    
  def build_nn(self):
    # Add 3 Conv2D layers
    self.add_conv(n_layers=self.conv_layers)
    
    # Reshape o for some goddamn reason
    shape = self.o.get_shape().as_list()
    self.o = tf.reshape(self.o, [-1, shape[1] * shape[2] * shape[3]])

    # Add 2 FCL layers
    self.add_fcl(n_layers=self.fcl_layers)
  
  def train(self):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.o, labels=self.y))
  
    var_list = tf.trainable_variables()
  
    l2_reg = 0
    for v in var_list:
      l2_reg += tf.reduce_mean(tf.square(v))
  
    l2_reg = (1.0 / len(var_list)) * self.params.l2_coeff * l2_reg
    loss += l2_reg
  
    opt = tf.train.MomentumOptimizer(self.params.lr, self.params.momentum)
    train = opt.minimize(loss)
    init = tf.global_variables_initializer()
  
    train_hist = []
    val_hist = []
  
    sess = tf.Session()
    sess.run(init)
  
    for i in range(self.params.train_steps):
      xb, yb = self.get_batch(self.X_train, self.Y_train, self.params.N, n=self.params.Nbatch)
      l, _ = sess.run([loss, train], {self.x: xb, self.y: yb})
  
      if i % self.params.print_step == 0:
        xb, yb = self.get_batch(self.X_val, self.Y_val, self.params.N, n=self.params.Nbatch)
        lval = sess.run(loss, {self.x: xb, self.y: yb})
    
        print "iter: {} Train: {}, Val: {}".format(i, l, lval)
    
        train_hist.append(l)
        val_hist.append(lval)

    a = sess.run(self.o, {self.x: xb, self.y: yb})

    # What are these?
    print a[:2]
    print a.shape
    print a[0] / np.sum(a[0])

  def add_conv(self, n_layers=0):
    for i in range(n_layers):
      self.o = self.conv_2d(self.o, dims=self.params.dims, filters=self.params.filters, strides=self.params.strides,
                            std=self.params.std, activation=self.params.act, scope='conv_{}'.format(i + 1))
      
  def add_fcl(self, n_layers=0):
    for i in range(n_layers):
      self.o = self.fully_connected(self.o, self.params.num_classes,
                                    activation=self.params.act, scope='fcl_{}'.format(i + 1))

  @staticmethod
  def get_batch(X, Y, N, n=32):
    inds = np.random.choice(range(N), size=n, replace=False)
    return X[inds], Y[inds]

  @staticmethod
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

  @staticmethod
  def fully_connected(x, output_units=100, activation=tf.identity, std=1e-3, scope='fc', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
      s = x.get_shape().as_list()
      shape = [s[1], output_units]
    
      W = tf.Variable(tf.random_normal(shape, stddev=std))
      b = tf.Variable(tf.ones([shape[1]]) * std)
      h = tf.matmul(x, W) + b
    
      return activation(h)
