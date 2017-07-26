import os
import sys
import h5py
from glimpse.helpers.definitions import dataset_path, image_dir, dml_dir, image_ext, image_color_repr
from glimpse.utils.vocab import dml2vec, pad_char
import numpy as np
from math import ceil
from scipy import misc
from argparse import ArgumentParser
from random import shuffle


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
  image_names = [n for n in os.listdir(image_dir) if n.endswith(image_ext)]
  shuffle(image_names)
  
  if limit:
    image_names = image_names[:limit]
  
  data = []
  dml_lengths = []
  
  # Group DML text with image names
  for n in image_names:
    dml_path = '{}/{}.dml'.format(dml_dir, n[:(-1 * len(image_ext))])
    
    if not os.path.exists(dml_path):
      continue
    
    with open(dml_path) as f:
      dml = f.read()

    dml_lengths.append(len(dml))
    data.append({'image_name': n, 'dml': dml})
  
  max_dml_length = max(dml_lengths)
  num_data_entries = len(data)
  
  print 'Found {} total Image-DML pairs.'.format(num_data_entries)
  print 'Padding all DML to length: {}'.format(max_dml_length)
  
  # Pad DML entries to that of longest length
  for info in data:
    dml_len_diff = max_dml_length - len(info['dml'])
    
    if dml_len_diff > 0:
      info['dml'] += (pad_char * dml_len_diff)

  train_split_index = int(ceil(data_split['train'] * num_data_entries))
  val_split_index = train_split_index + int(ceil(data_split['val'] * num_data_entries))
  
  train_data = data[:train_split_index]
  val_data = data[train_split_index:val_split_index]
  test_data = data[val_split_index:]

  return {
    'train': train_data,
    'val': val_data,
    'test': test_data
  }


def create_grouped_datasets(f, set_name, data):
  g = f.create_group(set_name)
  
  print 'Formatting {} dataset with {} records...'.format(set_name, len(data))
  
  images = []
  labels = []
  filenames = []
  
  i = 1
  for info in data:
    if not i % log_progress_step:
      print 'Done with {} of {}.'.format(i, len(data))
    
    image_name = info['image_name']
    
    # Add generic filename to filenames list
    filenames.append(image_name[:(-1 * len(image_ext))].encode('utf8'))
    
    # Add image as array to images list
    image_as_array = misc.imread('{}/{}'.format(image_dir, image_name), mode=image_color_repr)
    images.append(image_as_array)
    
    # Add DML as array to labels path
    dml_as_array = dml2vec(info['dml'])
    labels.append(dml_as_array)

    i += 1

  # Convert our lists to numpy arrays and normalize the image array
  images = normalize(np.array(images, dtype=dt))
  labels = np.array(labels, dtype=dt)

  # Create datasets for this group
  g.create_dataset('images', data=images)
  g.create_dataset('labels', data=labels)
  g.create_dataset('filenames', data=filenames, dtype=unicode_dt)

  print 'Done formatting {} dataset.'.format(set_name)


if __name__ == '__main__':
  args = parse_args()
  
  # Get split data by set ratio: train:val:test
  split_data = get_split_data(limit=args.limit)

  # Open hdf5 dataset file
  dataset = h5py.File(dataset_path, 'w')
  
  # For each set, create a group, and add 'images', 'labels', and 'filenames' datasets
  for set_name, data in split_data.iteritems():
    create_grouped_datasets(dataset, set_name, data)
  
  # Close dataset file
  dataset.close()