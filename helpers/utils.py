try:
  import cPickle as pickle
except:
  import pickle


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