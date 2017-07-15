# FORMAT:
#
# {
#   'data': np.array([
#     [],
#     [],
#     [],
#     []  # images converted to vectors somehow
#   ]),
#   'labels': [
#     0,  # contents of file "myimage.png"
#     3,  # contents of file "another-image.png"
#     2,  # ...
#     1  # ...
#   ],
#   'batch_label': 'training batch 1 of 5',
#   'filenames': [
#     'myimage.png',
#     'another-image.png',
#     'again.png',
#     'blah.png'
#   ]
# }

import os
from definitions import image_dir, dml_dir
import numpy as np
import code


image_sets = {k: os.listdir('{}/{}'.format(image_dir, k)) for k in ['train', 'validation', 'test']}

for set_name, image_names in image_sets.iteritems():
  image_names.sort()
  
  info = {
    'data': [],
    'labels': [],
    'batch_label': '{} batch'.format(set_name),
    'filenames': image_names
  }
  
  for n in image_names:
    dml_path = '{}/{}.dml'.format(dml_dir, n[:-4])
    
    assert os.path.exists(dml_path), 'No DML file at path {}'.format(dml_path)
    
    with open(dml_path) as f:
      dml = f.read().strip()
    
    info['labels'].append(dml)
    
    # Figure out how to convert image to it's pixel info
    image_data = []
    
    info['data'].append(image_data)
  
  info['data'] = np.array(info['data'])

  code.interact(locals=locals())