import os
import h5py
import tensorflow as tf
import numpy as np
from glimpse.helpers.definitions import dataset_path, model_dir, model_path, model_name
from glimpse.utils.params import Params
from glimpse.utils import pixnet
from glimpse.model import build_network

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
    batch_size = self.params.batch_size
    num_words = self.params.num_words
    vocab_size = self.params.vocab_size
    max_length = self.params.max_length
    learning_rate = self.params.learning_rate
    image_size = self.params.image_size

    inputs,outputs,train = build_network(batch_size,num_words,vocab_size,max_length,
          image_size,learning_rate)
          
    self.x_image,self.x_words,self.y_words,self.y_past = inputs
    self.output_words = outputs
    self.loss,self.minimize_loss = train

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

    #get max string length
    self.params.max_length = self.Y_train.shape[1]

  def get_batch(self):
    N = self.X_train.shape[0]

    inds = list(np.random.choice(range(N), size=self.params.batch_size, replace=False))
    inds.sort()

    x = self.X_train[inds]
    y = self.Y_train[inds]

    # Get random starting point in the dml string to use as input
    # The labels are then the starting point shifted one up

    y_in = np.zeros((self.params.batch_size, self.params.num_words, self.params.vocab_size))
    y_out = np.zeros((self.params.batch_size, self.params.num_words, self.params.vocab_size))
    y_past = np.zeros((self.params.batch_size, self.params.max_length, self.params.vocab_size))

    for i in range(self.params.batch_size):
      lab_len = self.train_label_lens[inds[i]] - self.params.num_words - 1

      if lab_len <= 0:
        print 'Label length <= 0...skipping index {}'.format(inds[i])
        return None

      start = np.random.randint(lab_len)
      end = start + self.params.num_words
      shifted_start = start + 1
      shifted_end = end + 1

      y_in[i] = y[i, start:end, :]
      y_out[i] = y[i, shifted_start:shifted_end, :]

      y_past[i,:start,:] = y[i,:start,:]
      y_past[i,start:,self.params.vocab_size-1] = 1.0

    return x, y_in, y_out, y_past

  def train(self):
    # Create a new session and initialize vars
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    # Create our model saver
    self.saver = tf.train.Saver()

    # Restore the previous model if it exists
    self.manage_previous_model()

    print 'Starting to train. Press Ctrl+C to save and exit.'

    try:
      for it in range(self.params.train_steps)[self.params.gstep:]:
        print '{}/{}'.format(it, self.params.train_steps)

        batch_info = self.get_batch()

        if not batch_info:
          self.params.gstep += 1
          continue

        x_in, y_in, y_lab, y_past = batch_info

        self.sess.run(self.minimize_loss, {self.x_image: x_in, self.x_words: y_in, self.y_words: y_lab,
            self.y_past : y_past})

        self.params.gstep += 1

        if self.params.gstep % self.params.print_every == 0:
          l = self.sess.run(self.loss, {self.x_image: x_in, self.x_words: y_in, self.y_words: y_lab,
            self.y_past : y_past})
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
