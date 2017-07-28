import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = basedir + '/data'
dml_dir = data_dir + '/dml'
image_dir = data_dir + '/images'
params_dir = basedir + '/glimpse/params'

vocab_path = '{}/vocab.json'.format(data_dir)

dataset_path = '{}/dataset-100.hdf5'.format(data_dir)

image_width = 640
image_height = 1250
image_ext = '.png'
image_color_repr = 'RGB'

model_name = 'model.ckpt'
model_path = '{}/{}'.format(data_dir, model_name)
