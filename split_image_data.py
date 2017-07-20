import os
from helpers.definitions import tmp_dir, dml_dir, image_dir
from math import ceil

samples_dir = tmp_dir + '/samples'

train_split = 0.6
validation_split = 0.2
test_split = 0.2

images = [f for f in os.listdir(samples_dir) if f.endswith('.png')]

num_images = len(images)

print 'Found {} images'.format(num_images)

train_split_index = int(ceil(train_split * num_images))
validation_split_index = train_split_index + int(ceil(validation_split * num_images))

train_images = images[:train_split_index]
validation_images = images[train_split_index:validation_split_index]
test_images = images[validation_split_index:]

sets = {
  'train': train_images,
  'validation': validation_images,
  'test': test_images
}


def dml_exists(img_name):
  dml_name = img_name[:-4] + '.dml'
  return os.path.exists('{}/{}'.format(dml_dir, dml_name))


for set, image_names in sets.iteritems():
  for n in image_names:
    if dml_exists(n):
      os.rename('{}/{}'.format(samples_dir, n), '{}/{}/{}'.format(image_dir, set, n))
    else:
      print 'No DML file for {}'.format(n[:-4])