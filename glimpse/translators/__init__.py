import json
from glimpse.helpers.definitions import translators_dir

with open('{}/lorem_ipsum.txt'.format(translators_dir)) as f:
  lorem_ipsum = f.read()


def get_attrs_map():
  return json_map('attrs_map')


def get_attr_vals_map():
  return json_map('attr_values_map')


def get_tags_map():
  return json_map('tags_map')


def json_map(name):
  with open('{}/{}.json'.format(translators_dir, name)) as f:
    return json.load(f)