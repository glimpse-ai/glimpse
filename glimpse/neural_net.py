"""
Abstract NeuralNetwork class that automatically pulls params from params.json
"""
import inspect
from params import Params


class NeuralNet:
  
  def __init__(self):
    child_class = self.__class__
    
    child_class_dir = '/'.join(inspect.getfile(child_class).split('/')[:-1])
    nn = child_class.__name__.lower()
    
    self.params = Params(nn=nn, dir=child_class_dir)