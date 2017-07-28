import os
import h5py
import tensorflow as tf
import numpy as np
from glimpse.helpers.definitions import dataset_path, model_dir, model_path, model_name
from glimpse.utils.params import Params
from glimpse.utils import pixnet


class Trainer:
  params = Params('trainer')

  def __init__(self):
    # Initialize empty attrs
    self.X_train, self.Y_train = None, None
    self.X_val, self.Y_val = None, None
    self.X_test, self.Y_test = None, None

    self.x_image = None
    self.x_words = None
    self.y_words = None

    self.train_label_lens = None
    self.val_label_lens = None
    self.test_label_lens = None
    
    self.loss = 0.0
    self.output_words = None
    self.opt = None
    self.minimize_loss = None
    self.sess = None

    self.saver = None
  
    # Read hdf5 dataset from disk
    self.extract_data()

    # Construct our network
    self.build_network()

  def extract_data(self):
    print 'Extracting data from dataset...'
    dataset = h5py.File(dataset_path, 'r')

    train_set = dataset.get('train')
    val_set = dataset.get('val')
    test_set = dataset.get('test')

    self.X_train, self.Y_train = train_set.get('images'), train_set.get('labels')
    self.X_val, self.Y_val = val_set.get('images'), val_set.get('labels')
    self.X_test, self.Y_test = test_set.get('images'), test_set.get('labels')
    
    # Non-padded label lengths
    self.train_label_lens = train_set.get('label_lens')
    self.val_label_lens = val_set.get('label_lens')
    self.test_label_lens = test_set.get('label_lens')

  def build_network(self):
    print 'Building network...'
    batch_size = self.params.batch_size
    num_words = self.params.num_words
    vocab_size = self.params.vocab_size

    self.x_image = tf.placeholder(shape=[batch_size] + self.params.image_size, dtype=tf.float32)
    self.x_words = tf.placeholder(shape=[batch_size, num_words, vocab_size], dtype=tf.float32)
    self.y_words = tf.placeholder(shape=[batch_size, num_words, vocab_size], dtype=tf.float32)

    v = pixnet.conv_block(self.x_image, batch_size=batch_size)

    self.output_words = pixnet.lstm_block(self.x_words, v, vocab_size=vocab_size,
                                     num_words=num_words, batch_size=batch_size)

    for i in range(len(self.output_words)):
      self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_words[:, i, :],
                                                                          logits=self.output_words[i]))

    self.loss = self.loss / num_words

    self.opt = tf.train.AdamOptimizer(self.params.learning_rate)
    self.minimize_loss = self.opt.minimize(self.loss)

  def get_batch(self):
    N = self.X_train.shape[0]

    inds = list(np.random.choice(range(N), size=self.params.batch_size, replace=False))
    inds.sort()
    
    x = self.X_train[inds]
    y = self.Y_train[inds]

    # Get random starting point in the dml string to use as input
    # The labels are then the starting point shifted one up
    max_dml_length = y.shape[1] - self.params.num_words - 1
    starts = np.random.choice(range(max_dml_length), size=self.params.batch_size, replace=False)
    
    y_in = np.zeros((self.params.batch_size, self.params.num_words, self.params.vocab_size))
    y_out = np.zeros((self.params.batch_size, self.params.num_words, self.params.vocab_size))

    for i in range(self.params.batch_size):
        start = starts[i]
        end = start + self.params.num_words
        shifted_start = start + 1
        shifted_end = end + 1
        
        y_in[i] = y[i, start:end, :]
        y_out[i] = y[i, shifted_start:shifted_end, :]

    return x, y_in, y_out

  def train(self):
    # Create a new session and initialize vars
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    # Create our model saver
    self.saver = tf.train.Saver(max_to_keep=200)  # Why 200? No clue
    
    # Restore the previous model if it exists
    self.manage_previous_model()

    print 'Starting to train. Press Ctrl+C to save and exit.'

    try:
      for it in range(self.params.train_steps)[self.params.gstep:]:
        print '{}/{}'.format(it, self.params.train_steps)
        
        x_in, y_in, y_lab = self.get_batch()
        
        self.sess.run(self.minimize_loss, {self.x_image: x_in, self.x_words: y_in, self.y_words: y_lab})
        
        self.params.gstep += 1
        
        if self.params.gstep % self.params.print_every == 0:
          l = self.sess.run(self.loss, {self.x_image: x_in, self.x_words: y_in, self.y_words: y_lab})
          print "iteration {}: training loss = {}".format(it, l)
  
        # Reached a checkpoint
        if self.params.gstep % self.params.save_every == 0:
          self.save_session()
        
    except (KeyboardInterrupt, SystemExit):
      print('Interruption detected, exiting the program...')
    
    self.save_session()

  def manage_previous_model(self):
    # Create model dir if not already there
    if not os.path.exists(model_dir):
      os.mkdir(model_dir)
    
    for f in os.listdir(model_dir):
      if f.startswith(model_name):
        print 'Previous model found. Restoring...'
        self.saver.restore(self.sess, model_path)
        break

  def save_session(self):
    print 'Saving model and its params...'
    # Save model params
    self.params.save()
    
    # Save session
    self.saver.save(self.sess, model_path)