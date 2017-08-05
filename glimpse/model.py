import os
from glimpse.helpers.definitions import model_path
from glimpse.utils.params import Params
from glimpse.utils import pixnet
import tensorflow as tf
import numpy as np

def build_network(batch_size,num_words,vocab_size,max_length,image_size,learning_rate,feed_previous=False):
    print 'Building network...'
    
    x_image = tf.placeholder(shape=[batch_size] + image_size, dtype=tf.float32)
    x_words = tf.placeholder(shape=[batch_size, num_words, vocab_size], dtype=tf.float32)
    y_words = tf.placeholder(shape=[batch_size, num_words, vocab_size], dtype=tf.float32)
    y_past = tf.placeholder(shape=[batch_size, max_length, vocab_size], dtype=tf.float32)
  
    v = pixnet.conv_block(x_image, batch_size=batch_size)

    t = pixnet.conv_text(y_past, batch_size=batch_size)

    output_words = pixnet.lstm_block(x_words, v, t, vocab_size=vocab_size,
                                     num_words=num_words, batch_size=batch_size,
                                     feed_previous=feed_previous)
  
    loss = 0.0
    for i in range(len(output_words)):
      loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_words[:, i, :],
                                                                          logits=output_words[i]))

    loss = loss / num_words

    if not feed_previous:
        opt = tf.train.AdamOptimizer(learning_rate)
        minimize_loss = opt.minimize(loss)
    else:
        opt = None
        minimize_loss = None

    return (x_image,x_words,y_words,y_past),output_words,(loss,minimize_loss)


class Model:
  params = Params('trainer')

  def __init__(self, path=model_path, feed_previous=False):
    self.sess = tf.Session()
    # self.sess.run(tf.global_variables_initializer())

    # Construct our network
    self.batch_size = self.params.batch_size
    self.num_words = self.params.num_words
    self.vocab_size = self.params.vocab_size
    self.max_length = self.params.max_length
    self.learning_rate = self.params.learning_rate
    self.image_size = self.params.image_size

    inputs,outputs,train = build_network(self.batch_size,self.num_words,
    self.vocab_size,self.max_length,self.image_size,self.learning_rate, feed_previous)

    self.x_image,self.x_words,self.y_words,self.y_past = inputs
    self.output_words = outputs
    self.loss,self.minimize_loss = train

    self.sess.run(tf.global_variables_initializer())

    # if not os.path.exists(path):
    #   raise BaseException('Model not found at path: {}'.format(path))
    self.saver = tf.train.Saver(max_to_keep=200)

    print 'Restoring model...'

    self.saver.restore(self.sess, path)

  def batch_predict(self,images):
    N = images.shape[0]
    predicted_words = np.zeros((N, self.max_length ,self.vocab_size))
    predicted_words[:,:,self.vocab_size-1] = 1.0

    for i in range(0,self.max_length,self.num_words-10):
      print "predicting words starting at {}".format(i)
      start_ind = i
      end_ind = i+self.num_words
      shifted_start = start_ind+1
      shifted_end = end_ind+1
      input_words = predicted_words[:,start_ind:end_ind,:]
      
      try:
        outputs = self.sess.run(self.output_words,{self.x_image:images,
            self.x_words:input_words,self.y_past:predicted_words})

        outputs = np.asarray(outputs)
        outputs = np.transpose(outputs,axes=(1,0,2))
        print np.argmax(outputs,axis=2)
        predicted_words[:,shifted_start:shifted_end,:] = outputs
      except BaseException, e:
        print 'FUCKING ERROR: {}'.format(e)
          
    return predicted_words