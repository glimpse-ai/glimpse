"""
FORMAT of .pkl files saved:

{
  'data': np.array([
    [], # image to numpy array
    [], # image to numpy array
    [], ...
    []  ...
  ]),
  'labels': [
    0,  # contents of file "myimage.dml"
    3,  # contents of file "another-image.dml"
    2,  # ...
    1   # ...
  ],
  'batch_label': 'test batch',
  'filenames': [
    'myimage.png',
    'another-image.png',
    ...
    ...
  ]
}

"""

import os
from definitions import data_dir, image_dir, dml_dir
import numpy as np
from scipy import misc
from helpers.utils import dump_pickle
from helpers.image import normalize


image_sets = {k: os.listdir('{}/{}'.format(image_dir, k)) for k in ['train', 'validation', 'test']}

for set_name, image_names in image_sets.iteritems():
  image_names = [n for n in image_names if n.endswith('.png')]
  image_names.sort()

  print 'Formatting {} dataset with {} records.'.format(set_name, len(image_names))

  info = {
    'data': [],
    'labels': [],
    'batch_label': '{} batch'.format(set_name),
    'filenames': image_names
  }
  
  i = 1
  for n in image_names:
    if not i % 100:
      print 'Done with {} of {}'.format(i, len(image_names))
    
    image_path = '{}/{}/{}'.format(image_dir, set_name, n)
    dml_path = '{}/{}.dml'.format(dml_dir, n[:-4])
    
    assert os.path.exists(dml_path), 'No DML file at path {}'.format(dml_path)
    
    with open(dml_path) as f:
      dml = f.read().strip()
    
    info['labels'].append(dml)
    
    image_as_array = misc.imread(image_path, mode='RGB')
    
    info['data'].append(image_as_array)
    
    i += 1
  
  info['labels'] = np.array(info['labels'])
  info['data'] = normalize(np.array(info['data']))
  
  pkl_path = '{}/{}.pkl'.format(data_dir, set_name)
  
  print 'Saving {} dataset...'.format(set_name)
  
  dump_pickle(pkl_path, info)
  
  print 'Done'