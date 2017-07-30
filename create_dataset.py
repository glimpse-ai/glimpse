import os
import sys
import h5py
from glimpse.helpers.definitions import data_dir, image_dir, image_color_repr, image_height, image_width
from glimpse.utils.vocab import dml2vec, pad_char, vocab
import numpy as np
from math import ceil
from scipy import misc
from argparse import ArgumentParser
from random import shuffle

# Change this to what you want
dml_dir = data_dir + '/tmpdml'

# Specify params
dt = np.float32 # for images and labels
unicode_dt = h5py.special_dtype(vlen=unicode) # for filenames
log_progress_step = 100
data_split = {'train': 0.6, 'val': 0.2, 'test': 0.2}


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('--limit', type=int, default=None)
  return parser.parse_args(sys.argv[1:])


def normalize(arr):
  return (1.0 * arr) / 255


def get_split_data(limit=None):
  names = [n[:-4] for n in os.listdir(dml_dir) if n.endswith('.dml') and os.path.exists('{}/{}.png'.format(image_dir, n[:-4]))]
  shuffle(names)
  
  if limit:
    names = names[:limit]

  train_split_index = int(ceil(data_split['train'] * len(names)))
  val_split_index = train_split_index + int(ceil(data_split['val'] * len(names)))

  split = {}
  split['train'] = names[:train_split_index]
  split['val'] = names[train_split_index:val_split_index]
  split['test'] = names[val_split_index:]
  
  return split


def create_grouped_datasets(f, set_name, filenames, max_dml_len):
  g = f.create_group(set_name)
  num_records = len(filenames)
  label_len = max_dml_len + 1
  
  print 'Formatting {} dataset with {} records...'.format(set_name, num_records)
  
  images = g.create_dataset('images', shape=(num_records, image_height, image_width, len(image_color_repr)), dtype=np.float32)
  labels = g.create_dataset('labels', shape=(num_records, label_len, len(vocab)), dtype=np.float32)
  label_lens = g.create_dataset('label_lens', shape=(num_records,), dtype='i')
  g.create_dataset('filenames', data=filenames, dtype=unicode_dt)
  
  i = 0
  for n in filenames:
    if (i + 1) % log_progress_step == 0:
      print 'Done with {} of {}.'.format(i, num_records)
    
    images[i] = misc.imread('{}/{}.png'.format(image_dir, n), mode=image_color_repr)
    
    with open('{}/{}.dml'.format(dml_dir, n)) as dml_file:
      dml = dml_file.read()
      dml_len = len(dml)
      label_lens[i] = dml_len
      labels[i] = dml2vec(dml + ((label_len - dml_len) * pad_char))
    
    i += 1


if __name__ == '__main__':
  args = parse_args()
  dataset_name = 'dataset'

  if args.limit:
    dataset_name += '-{}'.format(args.limit)
  
  # Get split data by set ratio: train:val:test
  split_data = get_split_data(limit=args.limit)

  # Open hdf5 dataset file
  dataset = h5py.File('{}/{}.hdf5'.format(data_dir, dataset_name), 'w')
  all_dml_names = split_data['train'] + split_data['val'] + split_data['test']
  
  lengths = []
  for n in all_dml_names:
    with open('{}/{}.dml'.format(dml_dir, n)) as f:
      lengths.append(len(f.read()))

  max_dml_len = max(lengths)
  
  # For each set, create a group, and add 'images', 'labels', and 'filenames' datasets
  for set_name, filenames in split_data.iteritems():
    create_grouped_datasets(dataset, set_name, filenames, max_dml_len)
  
  # Close dataset file
  dataset.close()