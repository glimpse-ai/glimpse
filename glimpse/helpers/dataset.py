import h5py
from glimpse.helpers.definitions import dataset_path

data = h5py.File(dataset_path, 'r')


def train():
  return get_dataset('train')


def val():
  return get_dataset('val')


def test():
  return get_dataset('test')


def get_dataset(setname):
  set = data.get(setname)

  if not set:
    raise BaseException('{} set does not exist yet in dataset'.format(setname))

  return set.get('images'), set.get('labels'), set.get('label_lens')
