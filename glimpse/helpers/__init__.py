from bs4 import BeautifulSoup
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def invert_map(m):
  return {v: k for k, v in m.iteritems()}


def invert_nested_map(m):
  new_m = {}
  
  for k, v in m.iteritems():
    val_type = type(v).__name__
    
    if val_type in ['str', 'unicode']:
      new_m[v] = k
    elif val_type == 'dict':
      new_m[k] = invert_map(v)
    else:
      new_m[k] = v

  return new_m


def soupify(html):
  try:
    soup = BeautifulSoup(html, 'html.parser')
  except BaseException, e:
    print 'Error parsing html into BeautifulSoup: {}'.format(e)

  return soup