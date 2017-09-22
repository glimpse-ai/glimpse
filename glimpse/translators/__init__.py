import json
from glimpse.helpers.definitions import translators_dir
import numpy as np


with open('{}/lorem_ipsum.txt'.format(translators_dir)) as f:
  lorem_ipsum = f.read()


def json_map(name):
  with open('{}/{}.json'.format(translators_dir, name)) as f:
    return json.load(f)


def get_attrs_map():
  return json_map('attrs_map')


def get_attr_vals_map():
  return json_map('attr_values_map')


def get_tags_map():
  return json_map('tags_map')


tags_map = get_tags_map()
num_tags = len(tags_map)

word_to_index = {k: int(v) for k, v in tags_map.iteritems()}
index_to_word = {v: k for k, v in word_to_index.iteritems()}

word_vectors = np.zeros([num_tags, num_tags])

for i in range(num_tags):
  word_vectors[i][i] = 1


def one_hot_for_tag(tag):
  if tag not in word_to_index:
    return None

  return word_vectors[word_to_index.get(tag)]


def one_hot_for_index(i):
  return word_vectors[i]