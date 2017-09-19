import os
import json
from glimpse.helpers.definitions import html_dir, insights_dir
from bs4 import BeautifulSoup, Tag
import matplotlib.pyplot as plt

el_count_map = {}
el_counts_path = '{}/el_counts.json'.format(insights_dir)


def plot(data):
  keys = [int(k) for k in data.keys()]
  vals = data.values()
  plt.bar(keys, vals, color='g')
  plt.show()


if __name__ == '__main__':
  if os.path.exists(el_counts_path):
    with open(el_counts_path) as f:
      saved_data = json.load(f)

    plot(saved_data)
  else:
    filenames = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    num_files = len(filenames)

    i = 0
    for name in filenames:
      if i % 100 == 0 and i > 0:
        print '{} / {}'.format(i, num_files)

      with open('{}/{}'.format(html_dir, name)) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

      num_els = len([el for el in soup.body.recursiveChildGenerator() if type(el) == Tag])

      if num_els in el_count_map:
        el_count_map[num_els] += 1
      else:
        el_count_map[num_els] = 1

      i += 1

    with open(el_counts_path, 'w+') as f:
      f.write(json.dumps(el_count_map, sort_keys=True, indent=2))

    plot(el_count_map)