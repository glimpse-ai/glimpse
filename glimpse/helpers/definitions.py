import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = basedir + '/data'
dml_dir = data_dir + '/dml'
image_dir = data_dir + '/images'
params_dir = basedir + '/glimpse/params'

vocab_path = '{}/vocab.json'.format(data_dir)

dataset_path = '{}/dataset.hdf5'.format(data_dir)

image_width = 640
image_height = 1250
image_ext = '.png'
image_color_repr = 'RGB'

model_name = 'model.ckpt'
model_dir = data_dir + '/model'
model_path = '{}/{}'.format(model_dir, model_name)

tmp_dir = basedir + '/tmp'

translators_dir = basedir + '/glimpse/translators'

placeholder_image_name = 'placeholder.jpg'
placeholder_input_text = 'Enter text'

templates_dir = basedir + '/glimpse/templates'

global_step_path = data_dir + '/global_step.json'