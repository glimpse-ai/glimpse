import os
import tensorflow as tf
from glimpse.helpers.definitions import model_path, model_dir, model_name, image_width, image_height, image_color_repr
from glimpse.utils.vocab import vocab_size
from glimpse.utils import pixnet
from glimpse.helpers.dataset import train

Y_train = train()[1]


class Model:
  path = model_path
  learning_rate = 0.0001
  image_size = [image_height, image_width, len(image_color_repr)]
  vocab_size = vocab_size
  max_length = Y_train.shape[1]

  def exists(self):
    return os.path.exists(model_dir) and len([f for f in os.listdir(model_dir) if f.startswith(model_name)]) > 0

  def build_network(self, batch_size=4, num_words=30, feed_previous=False):
    print 'Building network...'
    x_image = tf.placeholder(shape=[batch_size] + self.image_size, dtype=tf.float32)
    x_words = tf.placeholder(shape=[batch_size, num_words, self.vocab_size], dtype=tf.float32)
    y_words = tf.placeholder(shape=[batch_size, num_words, self.vocab_size], dtype=tf.float32)
    y_past = tf.placeholder(shape=[batch_size, self.max_length, self.vocab_size], dtype=tf.float32)

    v = pixnet.conv_block(x_image, batch_size=batch_size)

    t = pixnet.conv_text(y_past, batch_size=batch_size)

    output_words = pixnet.lstm_block(x_words, v, t, vocab_size=self.vocab_size,
                                     num_words=num_words, batch_size=batch_size,
                                     feed_previous=feed_previous)

    loss = 0.0
    for i in range(len(output_words)):
      loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_words[:, i, :],
                                                                     logits=output_words[i]))

    loss = loss / num_words

    minimize_loss = None
    if not feed_previous:
      opt = tf.train.AdamOptimizer(self.learning_rate)
      minimize_loss = opt.minimize(loss)

    return (x_image, x_words, y_words, y_past), output_words, (loss, minimize_loss)