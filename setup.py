import os
from glimpse.helpers.definitions import data_dir

s3_glimpse_dir = os.environ.get('S3_GLIMPSE_DIR').rstrip('/')


def pull_s3_data_dir(dir_name):
  compressed_fname = dir_name + '.tar.gz'
  extracted_dest_path = '{}/{}/'.format(data_dir, dir_name)
  
  if os.path.exists(extracted_dest_path):
    print '{} already exists...skipping'.format(extracted_dest_path)
    return
  
  # Pull .tar.gz file from S3
  os.system('wget {}/{}'.format(s3_glimpse_dir, compressed_fname))
  
  # Extract it
  os.system('tar -zxvf {}'.format(compressed_fname))
  
  # Move folder into data dir
  os.system('mv {}/ {}'.format(dir_name, extracted_dest_path))
  
  # Remove compressed file
  os.system('rm {}'.format(compressed_fname))


if __name__ == '__main__':
  # Create data dir if not already there
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  
  # Pull images and dml
  [pull_s3_data_dir(f) for f in ['images', 'dml']]
  
  # Pull vocab.json
  vocab_fname = 'vocab.json'
  os.system('wget {}/{}'.format(s3_glimpse_dir, vocab_fname))
  os.system('mv {} {}/{}'.format(vocab_fname, data_dir, vocab_fname))