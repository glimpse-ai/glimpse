import os
from helpers.definitions import data_dir


s3_glimpse_dir = os.environ.get('S3_GLIMPSE_DIR').rstrip('/')

# Create data dir if not already there
if not os.path.exists(data_dir):
  os.mkdir(data_dir)

# TODO: Update this following the use of hdf5

# Pull the following files from S3 bucket and move them into the data dir
for f in ['train.pkl', 'val.pkl', 'test.pkl', 'vocab.json']:
  dest_path = '{}/{}'.format(data_dir, f)
  
  if os.path.exists(dest_path):
    continue
  
  os.system('wget {}/{}'.format(s3_glimpse_dir, f))
  os.system('mv {} {}'.format(f, dest_path))