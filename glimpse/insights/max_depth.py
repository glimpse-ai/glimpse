import os
from glimpse.helpers.definitions import html_dir
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

depth_count_map = {}


def find_max_depth(el):
  if hasattr(el, 'contents') and el.contents:
    return max([find_max_depth(c) for c in el.contents]) + 1
  else:
    return 0


if __name__ == '__main__':
  filenames = [n for n in os.listdir(html_dir) if n.endswith('.html')]

  i = 0
  for n in filenames:
    if i % 100 == 0 and i > 0:
      print '{} / {}'.format(i, len(filenames))

    with open('{}/{}'.format(html_dir, n)) as f:
      soup = BeautifulSoup(f.read(), 'html.parser')

    depth = find_max_depth(soup.body) - 1

    if depth in depth_count_map:
      depth_count_map[depth] += 1
    else:
      depth_count_map[depth] = 1

    i += 1

  keys = [int(k) for k in depth_count_map.keys()]
  vals = depth_count_map.values()
  plt.bar(keys, vals, color='g')
  plt.show()