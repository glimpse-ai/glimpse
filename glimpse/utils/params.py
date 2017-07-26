from glimpse.helpers.definitions import params_dir
import json


class Params:
  
  def __init__(self, filename):
    with open('{}/{}.json'.format(params_dir, filename)) as f:
      params = json.load(f) or {}

    [setattr(self, k, v) for k, v in params.iteritems()]