import json
import numpy as np
from definitions import data_dir

# Load JSON array of vocab words from file
with open('{}/vocab.json'.format(data_dir)) as f:
  vocab = json.load(f)

num_words = len(vocab)

pad_char = vocab[-1]

# Create a num_words_x_num_words placeholder matrix
word_vectors = np.zeros([num_words, num_words])

word_to_index = {}

i = 0
for c in vocab:
  word_to_index[c] = i
  word_vectors[i][i] = 1
  i += 1


def word2vec(word):
  return word_vectors[word_to_index.get(word)]


def dml2vec(dml):
  return [list(word2vec(c)) for c in dml]