from glimpse.helpers.definitions import params_dir
import json


class Params:
  
  def __init__(self, filename):
    self.path = '{}/{}.json'.format(params_dir, filename)
    
    with open(self.path) as f:
      params = json.load(f) or {}
      self.og_param_keys = params.keys()

    [setattr(self, k, v) for k, v in params.iteritems()]
  
  def save(self):
    params = {k: getattr(self, k) for k in self.og_param_keys}
    
    with open(self.path, 'w+') as f:
      f.write(json.dumps(params, sort_keys=True, indent=2))