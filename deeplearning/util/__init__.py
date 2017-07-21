import numpy as np
from helpers.definitions import data_dir
try:
  import cPickle as pickle
except:
  import pickle


def extract_data(set):
  print 'Unpickling {}.pkl'.format(set)
  data = load_pickle('{}/{}.pkl'.format(data_dir, set))
  return data['data'], data['labels']

 
def load_pickle(path):
  if not path.endswith('.pkl'):
    path += '.pkl'
  
  with open(path, 'rb') as f:
    return pickle.load(f)


def dump_pickle(path, data):
  if not path.endswith('.pkl'):
    path += '.pkl'
  
  with open(path, 'w+') as f:
    pickle.dump(data, f)


def get_batch(X, Y, N, n=32):
  inds = np.random.choice(range(N), size=n, replace=False)
  return X[inds], Y[inds]


def normalize(arr):
  return (1.0 * arr) / 255