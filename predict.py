import h5py
from glimpse.helpers.definitions import dataset_path
from glimpse.utils.vocab import vec2dml
from glimpse.model import Model

if __name__ == '__main__':
  # Extract test data
  dataset = h5py.File(dataset_path, 'r')
  test_set = dataset.get('test')
  X_test, Y_test = test_set.get('images'), test_set.get('labels')
  
  # Restore model as class
  model = Model(feed_previous=True)
  
  predictions = model.batch_predict(X_test[0:4])
  
  for dml_vec in predictions:
    dml = vec2dml(dml_vec)
    print dml
    print ''