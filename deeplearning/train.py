import tensorflow as tf
import numpy as np
import util.pixnet as pixnet
from helpers.vocab import vocab
from helpers.definitions import image_width, image_height
from deeplearning.util import extract_data, get_batch
 
################################
# Define parameters
################################
image_size = [image_width, image_height, 3]
vocab_size = len(vocab)
num_words = 30  # attention
batch_size = 4  # Reason this is 4?
learning_rate = 1e-3

################################
# Extract data
################################
X_train, Y_train = extract_data('train')
X_val, Y_val = extract_data('val')
X_test, Y_test = extract_data('test')

print 'X_train.shape = {}, Y_train.shape = {}'.format(X_train.shape, Y_train.shape)

# Set more params
# Nbatch = 64 ?
N = X_train.shape[0] # => 6
train_steps = N

# Do we still need this?
# X = np.random.randn(train_steps, image_size[0], image_size[1], image_size[2])
# Y = np.zeros((train_steps, num_words + 1, vocab_size))
# inds = np.random.randint(vocab_size, size=(train_steps, num_words + 1, 1))
#
# for i in range(train_steps):
#   for j in range(num_words + 1):
#     Y[i, j, inds[i, j]] = 1
#
# # Do I need to set Y_train_in and Y_train_out?
# Y_in = Y[:, :num_words, :]
# Y_out = Y[:, 1:num_words + 1, :]

################################
# Build network
################################
x_image = tf.placeholder(shape=[batch_size] + image_size, dtype=tf.float32)
x_words = tf.placeholder(shape=[batch_size, num_words, vocab_size], dtype=tf.float32)
y_words = tf.placeholder(shape=[batch_size, num_words, vocab_size], dtype=tf.float32)

v = pixnet.conv_block(x_image, batch_size=batch_size)

output_words = pixnet.lstm_block(x_words, v, vocab_size=vocab_size,
                                 num_words=num_words, batch_size=batch_size)

loss = 0.0
for i in range(len(output_words)):
  loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_words[:, i, :],
                                                                 logits=output_words[i]))
loss = loss / num_words

################################=
# Train network
################################
opt = tf.train.AdamOptimizer(learning_rate)
train = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# TODO: X, Y_in, and Y_out need to be changed to reference X_train, something, and something else

sess.run(train, {x_image: X[:4], x_words: Y_in[:4], y_words: Y_out[:4]})

yhat = sess.run(output_words, {x_image: X[:4], x_words: Y_in[:4], y_words: Y_out[:4]})

l = sess.run(loss, {x_image: X[:4], x_words: Y_in[:4], y_words: Y_out[:4]})