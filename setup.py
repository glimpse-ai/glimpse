import os
from helpers.definitions import data_dir


glimpse_dir = os.environ.get('S3_GLIMPSE_DIR').rstrip('/')


if not os.path.exists(data_dir):
  os.mkdir(data_dir)

for f in ['train.pkl', 'val.pkl', 'test.pkl', 'vocab.json']:
  dest_path = '{}/{}'.format(data_dir, f)
  
  if os.path.exists(dest_path):
    continue
  
  os.system('wget {}/{}'.format(glimpse_dir, f))
  os.system('mv {} {}'.format(f, dest_path))