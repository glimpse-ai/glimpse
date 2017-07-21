import json
from definitions import data_dir


with open('{}/vocab.json'.format(data_dir)) as f:
  vocab = json.load(f)

word_to_index = {}

i = 0
for c in vocab:
  word_to_index[c] = i
  i += 1

index_to_word = {v: k for k, v in word_to_index.iteritems()}


def dml_to_ints(dml):
  return [word_to_index.get(c) for c in dml]