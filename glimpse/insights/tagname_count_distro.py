import os
import json
from glimpse.helpers.definitions import html_dir, insights_dir
from glimpse.translators import get_tags_map
from bs4 import BeautifulSoup, Tag
import matplotlib.pyplot as plt

tags_map = get_tags_map()

tagname_count_map = {}
tagname_counts_path = '{}/tagname_counts.json'.format(insights_dir)


def plot(data):
  keys = [int(k) for k in data.keys()]
  vals = data.values()
  plt.bar(keys, vals, color='g')
  plt.show()


if __name__ == '__main__':
  if os.path.exists(tagname_counts_path):
    with open(tagname_counts_path) as f:
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

      tagnames = [el.name for el in soup.body.recursiveChildGenerator() if type(el) == Tag]

      for tag in tagnames:
        tagname_repr = tags_map.get(tag)

        if not tagname_repr:
          continue

        if tagname_repr in tagname_count_map:
          tagname_count_map[tagname_repr] += 1
        else:
          tagname_count_map[tagname_repr] = 1

      i += 1

    with open(tagname_counts_path, 'w+') as f:
      f.write(json.dumps(tagname_count_map, sort_keys=True, indent=2))

    plot(tagname_count_map)