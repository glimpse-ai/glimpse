import h5py
import tensorflow as tf
from glimpse.helpers.definitions import dataset_path, image_width, image_height, image_color_repr
from glimpse.utils.vocab import vocab
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
    
    self.loss = 0.0
    self.sess = None
    self.output_words = None
    
    # Read hdf5 dataset from disk
    self.extract_data()
    
    # Set dynamic params
    setattr(self.params, 'vocab_size', len(vocab))
    setattr(self.params, 'image_size', [image_width, image_height, len(image_color_repr)])
    setattr(self.params, 'train_size', self.X_train.shape[0])
    
    # Construct our network
    self.build_network()
    
  def extract_data(self):
    dataset = h5py.File(dataset_path, 'r')
    
    train_set = dataset.get('train')
    val_set = dataset.get('val')
    test_set = dataset.get('test')
    
    self.X_train, self.Y_train = train_set.get('images'), train_set.get('labels')
    self.X_val, self.Y_val = val_set.get('images'), val_set.get('labels')
    self.X_test, self.Y_test = test_set.get('images'), test_set.get('labels')

  def build_network(self):
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

  def train(self):
    opt = tf.train.AdamOptimizer(self.params.learning_rate)
    train = opt.minimize(self.loss)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    # @gmaher: these 3 vars look correct?
    X = self.X_train
    Y_in = self.Y_train[:, :self.params.num_words, :]
    Y_out = self.Y_train[:, 1:self.params.num_words + 1, :]
    
    # Note: we have access to self.params.train_size for looping

    self.sess.run(train, {self.x_image: X[:4], self.x_words: Y_in[:4], self.y_words: Y_out[:4]})
    
    yhat = self.sess.run(self.output_words, {self.x_image: X[:4], self.x_words: Y_in[:4], self.y_words: Y_out[:4]})
    
    l = self.sess.run(self.loss, {self.x_image: X[:4], self.x_words: Y_in[:4], self.y_words: Y_out[:4]})