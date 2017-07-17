import json


class Params:
  filename = 'params.json'
  
  def __init__(self, nn=None, dir=None):
    self.nn = nn
    self.dir = dir
    
    params = self.open_params().get(self.nn) or {}
    
    [setattr(self, k, v) for k, v in params.iteritems()]
    
  def open_params(self):
    with open('{}/{}'.format(self.dir, self.filename)) as f:
      return json.load(f)
