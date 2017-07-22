import tensorflow as tf
import numpy as np
import h5py
import os
import json
from helpers.vocab import vocab
from helpers.definitions import data_dir, params_dir
from glimpse.params import Params
from deeplearning.util import pixnet

params_fname = 'trainer'


class Trainer:
  params = Params(params_fname)
  
  def __init__(self):
    self.X_train, self.Y_train = None, None
    self.X_val, self.Y_val = None, None
    self.X_test, self.Y_test = None, None
    
    self.x_image = None
    self.x_words = None
    self.y_words = None
    
    self.loss = 0.0
    
    setattr(self.params, 'model_path', '{}/{}'.format(data_dir, self.params.model_name))
    setattr(self.params, 'vocab_size', len(vocab))
    
    self.writer = tf.train.SummaryWriter(data_dir)
    self.saver = tf.train.Saver(max_to_keep=self.params.keep_max_checkpoints)
    
    with open('{}/{}.json'.format(data_dir, params_fname)) as f:
      self.json_params = json.load(f) or {}
      self.global_step = self.json_params.get('global_step')
      
    self.sess = None
    
    self.build_network()

  def build_network(self):
    bs = self.params.batch_size
    img_sz = self.params.image_size
    nw = self.params.num_words
    vs = self.params.vocab_size
    
    self.x_image = tf.placeholder(shape=[bs] + img_sz, dtype=tf.float32)
    self.x_words = tf.placeholder(shape=[bs, nw, vs], dtype=tf.float32)
    self.y_words = tf.placeholder(shape=[bs, nw, vs], dtype=tf.float32)
    
    cnn_output_vec = pixnet.conv_block(self.x_image, batch_size=bs)
    output_words = pixnet.lstm_block(self.x_words, cnn_output_vec, vocab_size=vs, num_words=nw, batch_size=bs)

    for i in range(len(output_words)):
      self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=self.y_words[:, i, :], logits=output_words[i]))
      
    self.loss //= nw

  def train(self):
    train = tf.train.AdamOptimizer(self.params.lr).minimize(self.loss)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    self.manage_previous_model()

    merged_summaries = tf.merge_all_summaries()

    for self.global_step in range(self.params.train_steps)[self.global_step:]:
      # Example of what I've done in the past:
      # _, loss, summary = sess.run(ops + (merged_summaries,), feed_dict)
      # self.writer.add_summary(summary, i)
  
      # TODO: X, Y_in, and Y_out need to be changed to reference ffffX_train, sometfhing, and something else
      # sess.run(train, {x_image: X[:4], x_words: Y_in[:4], y_words: Y_out[:4]})
      # yhat = sess.run(output_words, {x_image: X[:4], x_words: Y_in[:4], y_words: Y_out[:4]})
      # l = sess.run(loss, {x_image: X[:4], x_words: Y_in[:4], y_words: Y_out[:4]})

      self.global_step += 1

      if not self.global_step % self.save_interval:
        self.save_session()

  def manage_previous_model(self):
    if os.path.exists(self.params.model_name):
      print('Restoring previous model from {}'.format(self.params.model_name))
      self.saver.restore(self.sess, self.params.model_path)
      print('Model restored.')

  def save_session(self):
    self.save_global_step()
    self.saver.save(self.sess, self.params.model_path)
    print 'Model saved.'

  def save_global_step(self):
    self.json_params['global_step'] = self.global_step
    
    with open('{}/{}.json'.format(params_dir, params_fname), 'w+') as f:
      f.write(json.dumps(self.json_params, indent=2, sort_keys=True))