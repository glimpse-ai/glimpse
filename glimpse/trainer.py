import json
import os
import tensorflow as tf
import numpy as np
from glimpse.model import Model
from glimpse.helpers import dataset
from glimpse.helpers.definitions import global_step_path
from glimpse.utils.vocab import vec2dml


class Trainer:
  batch_size = 4
  print_every = 10
  save_every = 100
  train_steps = 20000
  num_words = 30

  def __init__(self, feed_previous=False):
    # Establish train, val, and test data refs
    print 'Extracting data...'
    self.X_train, self.Y_train, self.train_label_lens = dataset.train()
    self.X_val, self.Y_val, self.val_label_lens = dataset.val()
    self.X_test, self.Y_test, self.test_label_lens = dataset.test()

    # Get our model instance
    self.model = Model()

    # Build network
    inputs, output, loss_info = self.model.build_network(batch_size=self.batch_size,
                                                         num_words=self.num_words,
                                                         feed_previous=feed_previous)

    # Establish network inputs, final output, loss & minimize_loss functions
    self.x_image, self.x_words, self.y_words, self.y_past = inputs
    self.output_words = output
    self.loss, self.minimize_loss = loss_info

    # Create a new session and initialize globals
    print 'Initializing session...'
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    # Create our saver
    self.saver = tf.train.Saver(max_to_keep=200)

    # Restore prev model if exists
    if self.model.exists():
      print 'Previous model found. Restoring...'
      self.saver.restore(self.sess, self.model.path)

    # Get stored global step value
    self.global_step = self.get_gstep()

  def get_batch(self):
    N = self.X_train.shape[0]

    inds = list(np.random.choice(range(N), size=self.batch_size, replace=False))
    inds.sort()

    x = self.X_train[inds]
    y = self.Y_train[inds]

    # Get random starting point in the dml string to use as input
    # The labels are then the starting point shifted one up

    y_in = np.zeros((self.batch_size, self.num_words, self.model.vocab_size))
    y_out = np.zeros((self.batch_size, self.num_words, self.model.vocab_size))
    y_past = np.zeros((self.batch_size, self.model.max_length, self.model.vocab_size))

    for i in range(self.batch_size):
      lab_len = self.train_label_lens[inds[i]] - self.num_words - 1

      start = np.random.randint(lab_len)
      end = start + self.num_words
      shifted_start = start + 1
      shifted_end = end + 1

      y_in[i] = y[i, start:end, :]
      y_out[i] = y[i, shifted_start:shifted_end, :]

      y_past[i, :start, :] = y[i, :start, :]
      y_past[i, start:, self.model.vocab_size - 1] = 1.0

    return x, y_in, y_out, y_past

  def train(self):
    print 'Starting to train. Press Ctrl+C to save and exit.'
    
    try:
      for i in range(self.train_steps)[self.global_step:]:
        print '{}/{}'.format(i, self.train_steps)

        x_in, y_in, y_lab, y_past = self.get_batch()

        feed_dict = {
          self.x_image: x_in,
          self.x_words: y_in,
          self.y_words: y_lab,
          self.y_past: y_past
        }

        self.sess.run(self.minimize_loss, feed_dict)

        self.global_step += 1

        if self.global_step % self.print_every == 0:
          l = self.sess.run(self.loss, feed_dict)
          print "iteration {}: training loss = {}".format(self.global_step, l)

        # Reached a checkpoint
        if self.global_step % self.save_every == 0:
          self.save_session()

    except (KeyboardInterrupt, SystemExit):
      print('Interruption detected, exiting the program...')

    self.save_session()

  def predict(self, images):
    N = images.shape[0]
    predicted_words = np.zeros((N, self.model.max_length, self.model.vocab_size))
    predicted_words[:, :, self.model.vocab_size - 1] = 1.0

    for i in range(0, self.model.max_length, self.num_words - 10):
      start_ind = i
      end_ind = i + self.num_words

      # Don't shift indexes on first loop -- otherwise, first char will always be pad character
      if start_ind == 0:
        shifted_start = start_ind
        shifted_end = end_ind
      else:
        shifted_start = start_ind + 1
        shifted_end = end_ind + 1

      input_words = predicted_words[:, start_ind:end_ind, :]

      try:
        outputs = self.sess.run(self.output_words, {
                                  self.x_image: images,
                                  self.x_words: input_words,
                                  self.y_past: predicted_words
                                })

        outputs = np.asarray(outputs)
        outputs = np.transpose(outputs, axes=(1, 0, 2))

        predicted_words[:, shifted_start:shifted_end, :] = outputs

        # Log 1st batch's predicted DML
        print vec2dml(predicted_words[0]) + '\n'

      except KeyboardInterrupt:
        return predicted_words
      except BaseException, e:
        print 'Prediction error: {}'.format(e)

    return predicted_words

  def save_session(self):
    print 'Saving session...'
    self.set_gstep()
    self.saver.save(self.sess, self.model.path)

  def get_gstep(self):
    # Create global_step.json if not there yet
    if not os.path.exists(global_step_path):
      self.set_gstep()
      return 0

    with open(global_step_path) as f:
      return json.load(f).get('val') or 0

  def set_gstep(self):
    with open(global_step_path, 'w+') as f:
      f.write(json.dumps({'val': self.global_step or 0}, indent=2))