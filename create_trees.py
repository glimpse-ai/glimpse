import os
import json
from glimpse.helpers.definitions import html_dir
from bs4 import BeautifulSoup, Tag

tags_map = {
  'div': [1, 0, 0],
  'a': [0, 1, 0],
  'p': [0, 0, 1]
}


def soup_from_filename(filename):
  with open('{}/{}'.format(html_dir, filename)) as f:
    return BeautifulSoup(f.read(), 'html.parser')


def children(el):
  return [c for c in el.children if type(c) == Tag]


def add_el(el, container):
  new_el = []
  new_el.append(tags_map.get(el.name))

  new_el_children = []
  for c in children(el):
    new_el_children = add_el(c, new_el_children)

  new_el.append(new_el_children)
  container.append(new_el)

  return container


if __name__ == '__main__':
  # html_names = [n for n in os.listdir(html_dir) if n.endswith('.html')]
  html_names = ['test.html']

  for n in html_names:
    # soup = soup_from_filename(n)
    with open(n) as f:
      soup = BeautifulSoup(f.read(), 'html.parser')

    body = []
    for el in children(soup.body):
      body = add_el(el, body)

    import code; code.interact(local=locals())

    with open('test.json', 'w+') as f:
      f.write(json.dumps(body, indent=2))