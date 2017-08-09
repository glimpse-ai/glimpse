import os
from glimpse.helpers.definitions import data_dir

dml_dir = data_dir + '/charlimit-8000/dml'
lower_bound = 40

files = [f for f in os.listdir(dml_dir) if f.endswith('.dml')]
remove_count = 0

for name in files:
  fpath = '{}/{}'.format(dml_dir, name)
  
  with open(fpath) as f:
    dml = f.read()
  
  if len(dml) < lower_bound:
    remove_count += 1
    os.remove(fpath)


print 'Done. Removed {} DML files that were too short.'.format(remove_count)