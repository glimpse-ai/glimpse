import os
import sys
import h5py
from glimpse.helpers.definitions import data_dir, image_dir, image_ext, image_color_repr
from glimpse.utils.vocab import dml2vec, pad_char
import numpy as np
from math import ceil
from scipy import misc
from argparse import ArgumentParser
from random import shuffle

# Change this to what you want
dml_dir = data_dir + '/charlimit-15000/dml'

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
  dml_names = [n for n in os.listdir(dml_dir) if n.endswith('.dml')]
  shuffle(dml_names)
  
  if limit:
    dml_names = dml_names[:limit]
    
  data = []
  dml_lengths = []
  
  for n in dml_names:
    image_name = n[:-4] + image_ext
    
    # continue if image doesn't exist
    if not os.path.exists('{}/{}'.format(image_dir, image_name)):
      continue
    
    with open('{}/{}'.format(dml_dir, n)) as f:
      dml = f.read()

    dml_lengths.append(len(dml))
    data.append({'image_name': image_name, 'dml': dml})
  
  max_dml_length = max(dml_lengths)
  num_data_entries = len(data)
  
  print 'Found {} total Image-DML pairs.'.format(num_data_entries)
  print 'Padding all DML to length: {}'.format(max_dml_length)
  
  # Pad DML entries to that of longest length
  for info in data:
    dml_len = len(info['dml'])
    
    # store original dml length
    info['dml_len'] = dml_len
    
    dml_len_diff = max_dml_length - dml_len
    
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
  label_len = []
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
    
    # Add label_len as key (non-padded length)
    label_len.append(info['dml_len'])
    
    i += 1

  # Convert our lists to numpy arrays and normalize the image array
  images = normalize(np.array(images, dtype=dt))
  labels = np.array(labels, dtype=dt)
  label_len = np.array(label_len, dtype='i')

  # Create datasets for this group
  g.create_dataset('images', data=images)
  g.create_dataset('labels', data=labels)
  g.create_dataset('label_lens', data=label_len)
  g.create_dataset('filenames', data=filenames, dtype=unicode_dt)

  print 'Done formatting {} dataset.'.format(set_name)


if __name__ == '__main__':
  args = parse_args()
  dataset_name = 'dataset'
  
  if args.limit:
    dataset_name += '-{}'.format(args.limit)
  
  # Get split data by set ratio: train:val:test
  split_data = get_split_data(limit=args.limit)

  # Open hdf5 dataset file
  dataset = h5py.File('{}/{}.hdf5'.format(data_dir, dataset_name), 'w')
  
  # For each set, create a group, and add 'images', 'labels', and 'filenames' datasets
  for set_name, data in split_data.iteritems():
    create_grouped_datasets(dataset, set_name, data)
  
  # Close dataset file
  dataset.close()