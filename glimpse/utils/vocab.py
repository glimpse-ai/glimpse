import json
import numpy as np
from glimpse.helpers.definitions import vocab_path
from glimpse.helpers import invert_map

# Load JSON array of vocab words from file
with open(vocab_path) as f:
  vocab = json.load(f)

vocab_size = len(vocab)

# pad_char will be last char in vocab
pad_char = vocab[-1]

# Create a vocab_size_x_vocab_size placeholder matrix
word_vectors = np.zeros([vocab_size, vocab_size])

word_to_index = {}

i = 0
for c in vocab:
  word_to_index[c] = i
  word_vectors[i][i] = 1
  i += 1

index_to_word = invert_map(word_to_index)


def word2vec(word):
  return word_vectors[word_to_index.get(word)]


def vec2word(vec):
  index = list(vec).index(1.0)
  return index_to_word[index]
  

def dml2vec(dml):
  return [list(word2vec(c)) for c in dml]


def vec2dml(vec):
  return ''.join([vec2word(v) for v in vec]).rstrip(pad_char)