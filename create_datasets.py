import os
import h5py
from helpers.definitions import image_dir, dml_dir, image_type
from helpers.vocab import dml2vec
import numpy as np
from scipy import misc
from deeplearning.util import normalize

dt = np.float32
unicode_dt = h5py.special_dtype(vlen=unicode)
print_step = 100

image_sets = {k: os.listdir('{}/{}'.format(image_dir, k)) for k in ['train', 'val', 'test']}

f = h5py.File('test.hdf5', 'w')

for set_name, image_names in image_sets.iteritems():
  image_names = [n.encode('utf8') for n in image_names if n.endswith('.png')]
  image_names.sort()
  
  print 'Formatting {} dataset with {} records...'.format(set_name, len(image_names))
  
  # Create 'train', 'val', or 'test' group
  g = f.create_group(set_name)
  
  images = []
  labels = []
  
  i = 1
  for n in image_names:
    if not i % print_step:
      print 'Done with {} of {}'.format(i, len(image_names))
    
    image_path = '{}/{}/{}'.format(image_dir, set_name, n)
    dml_path = '{}/{}.dml'.format(dml_dir, n[:-4])
    
    # Make sure we have a label for this image
    assert os.path.exists(dml_path), 'No DML file at path {}'.format(dml_path)
    
    with open(dml_path) as f:
      dml = f.read().strip()
    
    # Convert image to array
    image_as_array = misc.imread(image_path, mode=image_type)
    images.append(image_as_array)
    
    # Convert label to array
    dml_as_array = dml2vec(dml)
    labels.append(dml_as_array)

    i += 1
  
  # Convert our lists to numpy arrays and normalize image array
  images = normalize(np.array(images, dtype=dt))
  labels = np.array(labels, dtype=dt)
  
  # Create datasets for this group
  g.create_dataset('images', data=images)
  g.create_dataset('labels', data=labels)
  g.create_dataset('filenames', data=image_names, dtype=unicode_dt)
  
  print 'Done formatting {} dataset.'.format(set_name)

# Close our hdf5 file
f.close()