import os
from glimpse.helpers.definitions import model_path
import tensorflow as tf


class Model:
  
  def __init__(self, path=model_path):
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    
    if not os.path.exists(path):
      raise BaseException('Model not found at path: {}'.format(path))

    self.saver = tf.train.Saver(max_to_keep=200)
    
    print 'Restoring model...'
    
    self.saver.restore(self.sess, path)
    
  def batch_predict(self):
    return
  
  def single_predict(self):
    return