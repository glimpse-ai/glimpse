import os
from glimpse.helpers.definitions import data_dir, tmp_dir
from glimpse.dml import DML
from glimpse.utils.vocab import dml2vec, vec2dml

dml_dir = data_dir + '/charlimit-8000/dml'
results_dir = tmp_dir + '/results'

if not os.path.exists(tmp_dir):
  os.mkdir(tmp_dir)

if not os.path.exists(results_dir):
  os.mkdir(results_dir)

dml_names = [n for n in os.listdir(dml_dir) if n.endswith('.dml')][::250]

dml = DML()

i = 1
for dml_name in dml_names:
  if not i % 10:
    print '{}/{}'.format(i, len(dml_names))
  
  with open('{}/{}'.format(dml_dir, dml_name)) as f:
    dml_input = f.read()
  
  # Convert DML to input vector
  input_vec = dml2vec(dml_input)

  # Assuming 100% accuracy
  output_vec = input_vec

  # Convert output vector to DML
  output_dml = vec2dml(output_vec)
  
  dml.source = output_dml
  
  html = dml.to_html()
  
  with open('{}/{}.html'.format(results_dir, dml_name[:-4]), 'w+') as f:
    f.write(html)
  
  i += 1