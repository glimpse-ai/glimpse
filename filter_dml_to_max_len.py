import sys
import os
from argparse import ArgumentParser
from glimpse.helpers.definitions import data_dir, dml_dir


parser = ArgumentParser()
parser.add_argument('--limit', type=int, default=None)
args = parser.parse_args(sys.argv[1:])

if not args.limit:
  print 'No limit specified. Exiting.'
  exit(1)

new_data_dir = '{}/limit-{}'.format(data_dir, args.limit)
new_dml_dir = '{}/dml'.format(new_data_dir)

if os.path.exists(new_data_dir):
  print '{} already exists. Exiting.'.format(new_data_dir)
  exit(1)
  
os.mkdir(new_data_dir)
os.mkdir(new_dml_dir)

all_dml_files = [n for n in os.listdir(dml_dir) if n.endswith('.dml')]

i = 1
for n in all_dml_files:
  if not i % 100:
    print 'Done with {} of {}'.format(i, len(all_dml_files))
  
  dml_src = '{}/{}'.format(dml_dir, n)
  dml_dest = '{}/{}'.format(new_dml_dir, n)
  
  with open(dml_src) as f:
    dml = f.read()
    
  if len(dml) <= args.limit:
    os.system('cp {} {}'.format(dml_src, dml_dest))
  
  i += 1