import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = basedir + '/data'
dml_dir = data_dir + '/dml'
image_dir = data_dir + '/images'
tmp_dir = basedir + '/tmp'
params_dir = basedir + '/params'

dataset_path = '{}/dataset.hdf5'.format(data_dir)

image_width = 640
image_height = 1250
image_ext = '.png'
image_color_repr = 'RGB'