import os
from glimpse.helpers.definitions import data_dir, tmp_dir
from math import ceil

dml_dir = data_dir + '/charlimit-15000/dml'

if __name__ == '__main__':
  files = [f for f in os.listdir(dml_dir) if f.endswith('.dml')]

  len_ranges = range(51000)[::1000][1:]

  len_map = {str(k): 0 for k in len_ranges}

  for f in files:
    with open('{}/{}'.format(dml_dir, f)) as dml_file:
      dml = dml_file.read()

    rounded_len = str(int(ceil(len(dml) / 1000.0)) * 1000)

    if len_map[rounded_len] is not None:
      len_map[rounded_len] += 1

  dest_path = '{}/dml_lengths.txt'.format(tmp_dir)

  if os.path.exists(dest_path):
    os.remove(dest_path)

  os.system('touch {}'.format(dest_path))

  for r in len_ranges:
    os.system("echo '{}: {}' >> {}".format(r, len_map[str(r)], dest_path))