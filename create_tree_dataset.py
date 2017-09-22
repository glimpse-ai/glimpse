"""
Script to create Image + Tree Structure dataset
"""
import os
import sys
import h5py
from glimpse.helpers.definitions import data_dir, image_dir, html_dir, image_color_repr, image_height, image_width
import numpy as np
from glimpse.translators import get_tags_map
from bs4 import BeautifulSoup, Tag
from math import ceil
from scipy import misc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from argparse import ArgumentParser
from random import shuffle

# Max number of child nodes any node can have
MAX_CONNECTIONS = 10

tags_map = get_tags_map()


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--max_els', type=int, default=201)
  return parser.parse_args(sys.argv[1:])


def get_soup(n):
  with open('{}/{}.html'.format(html_dir, n)) as f:
    return BeautifulSoup(f.read(), 'html.parser')


def normalize(arr):
  return (1.0 * arr) / 255


def get_tree_specs():
  args = parse_args()
  max_depth = (args.max_els - 1) / MAX_CONNECTIONS
  max_nodes = ((args.max_els - 1) * MAX_CONNECTIONS) + 1

  return args.max_els, max_nodes, max_depth


def get_num_els(el):
  return len([el for el in el.recursiveChildGenerator() if type(el) == Tag]) + 1


def find_el_depth(el, d=1):
  if el.parent.name == 'body':
    return d
  else:
    return find_el_depth(el.parent, (d + 1))


def get_split_data(max_els, max_depth, train_split=0.7):
  # Get filenames where both html and images exist
  names = [n[:-5] for n in os.listdir(html_dir) if
           n.endswith('.html') and os.path.exists('{}/{}.png'.format(image_dir, n[:-5]))]

  names_len = len(names)

  filtered_names = []

  print 'Filtering through {} HTML files...'.format(names_len)

  i = 0
  for n in names:
    i += 1

    if i % 100 == 0:
      print '{} / {}'.format(i, names_len)

    soup = get_soup(n)

    body_children = [el for el in soup.body.recursiveChildGenerator() if type(el) == Tag]

    if not body_children:
      continue

    deepest_depth = max([find_el_depth(el) for el in body_children])
    num_els = len(body_children) + 1

    if deepest_depth > max_depth or num_els > max_els or too_many_connections([soup.body] + body_children):
      continue

    filtered_names.append(n)

  filtered_names_len = len(filtered_names)

  print '{} resulting filtered files.'.format(filtered_names_len)

  shuffle(filtered_names)

  val_split = (1 - train_split) / 2
  train_split_index = int(ceil(train_split * filtered_names_len))
  val_split_index = train_split_index + int(ceil(val_split * filtered_names_len))

  split = {}
  split['train'] = filtered_names[:train_split_index]
  split['val'] = filtered_names[train_split_index:val_split_index]
  split['test'] = filtered_names[val_split_index:]

  return split


def too_many_connections(els):
  for el in els:
    if len(children(el)) > MAX_CONNECTIONS:
      return True

  return False


def children(el):
  return [c for c in el.children if type(c) == Tag]


def fresh_depth_map(d):
  return {str(i): [] for i in range(1, d + 1)}


def classes_and_connections(n, max_depth, max_nodes):
  soup = get_soup(n)
  depth_map = fresh_depth_map(max_depth)
  els = [el for el in soup.body.recursiveChildGenerator() if type(el) == Tag]

  for el in els:
    depth = find_el_depth(el)
    depth_map[str(depth)].append(el)

  node_index = 1
  for i in range(1, max_depth + 1):
    els = depth_map.get(str(i))  # get els at depth i

    for el in els:
      el.attrs['index'] = node_index

      # Get this el's set of siblings (including itself)
      parent_children = children(el.parent)

      node_index += 1

      # If this is the last el in it's sibling group, extend the node_index
      # with null indexes up to MAX_CONNECTIONS
      if parent_children.index(el) == len(parent_children) - 1:
        node_index += (MAX_CONNECTIONS - len(parent_children))

  classes = []
  connections = []

  for i in range(max_nodes):
    if i == 0:
      el = soup.body
    else:
      el = soup.body.find(attrs={'index': str(i)})

    if el:
      class_num = int(tags_map.get(el.name))
      conns = []

      for c in children(el):
        conns.append(int(c.attrs['index']))

      if len(conns) > 0:
        while len(conns) < MAX_CONNECTIONS:
          conns.append(conns[-1] + 1)
      else:
        conns = [-1 for j in range(MAX_CONNECTIONS)]
    else:
      class_num = -1
      conns = [-1 for j in range(MAX_CONNECTIONS)]

    classes.append(class_num)
    connections.append(conns)

  return classes, connections


def create_grouped_datasets(f, set_name, filenames, max_nodes, max_depth):
  # Create group
  g = f.create_group(set_name)

  num_records = len(filenames)

  print 'Formatting {} dataset with {} records...'.format(set_name, num_records)

  # Create datasets for group
  images = g.create_dataset('images', shape=(num_records, image_height, image_width, len(image_color_repr)),
                            dtype=np.float32)

  classes = g.create_dataset('classes', shape=(num_records, max_nodes), dtype=np.float32)

  connections = g.create_dataset('connections', shape=(num_records, max_nodes, MAX_CONNECTIONS), dtype=np.float32)

  # Populate datasets
  i = 0
  for n in filenames:
    if i % 100 == 0 and i > 0:
      print '{} / {}.'.format(i, num_records)

    images[i] = normalize(misc.imread('{}/{}.png'.format(image_dir, n), mode=image_color_repr))

    classes[i], connections[i] = classes_and_connections(n, max_depth, max_nodes)

    i += 1


if __name__ == '__main__':
  # Get tree specs
  max_els, max_nodes, max_depth = get_tree_specs()

  # Split the data by ratio
  split_data = get_split_data(max_els, max_depth)

  # Name the dataset file according to its specs
  dataset_name = 'dataset-{}e-{}c'.format(max_els, MAX_CONNECTIONS)

  # Open dataset file
  dataset = h5py.File('{}/{}.hdf5'.format(data_dir, dataset_name), 'w')

  # For each set, create a group, and add 'images', 'classes', and 'connections' datasets
  for set_name, filenames in split_data.iteritems():
    create_grouped_datasets(dataset, set_name, filenames, max_nodes, max_depth)

  # Close dataset file
  dataset.close()