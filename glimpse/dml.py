import re
from glimpse.helpers.definitions import placeholder_image_name, placeholder_input_text
from glimpse.translators import get_attrs_map, get_attr_vals_map, get_tags_map, lorem_ipsum
from glimpse.helpers import invert_map, invert_nested_map, soupify
from glimpse.templates import wrap_body
from bs4 import Tag
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class DML:

  def __init__(self, source=None):
    # DML text source
    self.source = source
    
    # Translate maps
    self.attrs_map = invert_map(get_attrs_map())
    self.attr_vals_map = invert_nested_map(get_attr_vals_map())
    self.tags_map = invert_map(get_tags_map())

  def to_html(self):
    html = self.source
    
    decoders = [
      self.decode_tag_names,
      self.decode_attributes,
      self.populate_text,
      self.move_attrs_to_style,
      self.add_special_attrs,
      wrap_body,
      self.prettify
    ]
    
    for d in decoders:
      html = d(html)
      
    return html

  def decode_tag_names(self, dml):
    tag_for_val = self.tag_for_val
    
    def opening_with_attrs(m):
      return '<{} '.format(tag_for_val(m.group(1)))
  
    def opening_no_attrs(m):
      return '<{}>'.format(tag_for_val(m.group(1)))
  
    def closing(m):
      return '</{}>'.format(tag_for_val(m.group(1)))
  
    dml = re.sub('<([0-9]+) ', opening_with_attrs, dml)
    dml = re.sub('<([0-9]+)>', opening_no_attrs, dml)
    dml = re.sub('</([0-9]+)>', closing, dml)
    
    return dml

  def decode_attributes(self, html):
    attr_name_for_val = self.attr_name_for_val
    attr_val_for_val = self.attr_val_for_val
    
    def sub(m):
      attr = attr_name_for_val(m.group(1))
      attr_val = attr_val_for_val(attr, m.group(2))
      return ' {}="{}"'.format(attr, attr_val)
  
    return re.sub(' ([a-z]+)="(.*?)"', sub, html)

  def populate_text(self, html):
    def sub(m):
      return lorem_ipsum[0:int(m.group(1))]

    return re.sub('t\(([0-9]+)\)', sub, html)

  def move_attrs_to_style(self, html):
    attrs_to_keep = ['class', 'type', 'all-bb']
    
    soup = soupify(html)
    elements = [el for el in soup.recursiveChildGenerator() if type(el) == Tag]

    for el in elements:
      if not el.attrs:
        continue

      new_attrs = {}
      for a in attrs_to_keep:
        attr_val = el.attrs.get(a)

        if attr_val:
          new_attrs[a] = attr_val

      new_attrs['style'] = '; '.join(
        ['{}:{}'.format(a, v) for a, v in el.attrs.iteritems() if a not in attrs_to_keep])

      el.attrs = new_attrs

    return soup.encode('utf-8')

  @staticmethod
  def add_special_attrs(html):
    soup = soupify(html)
    elements = [el for el in soup.recursiveChildGenerator() if type(el) == Tag]

    for el in elements:
      attrs = el.attrs or {}

      if el.name == 'img':
        attrs['src'] = placeholder_image_name
      elif el.name == 'a':
        attrs['href'] = '#'
      elif el.name == 'input' and attrs.get('type') == 'text':
        el.attrs['placeholder'] = placeholder_input_text

      if attrs:
        el.attrs = attrs

    return soup.encode('utf-8')
  
  @staticmethod
  def prettify(html):
    return soupify(html).prettify().encode('utf-8')
  
  def tag_for_val(self, val):
    return self.tags_map.get(val) or val
  
  def attr_name_for_val(self, val):
    return self.attrs_map.get(val) or val
  
  def attr_val_for_val(self, attr_name, val):
    if attr_name == 'class' and val:
      encoded_classes = [c.strip() for c in val.split(',')]
      classes = [self.attr_vals_map['class'].get(c) for c in encoded_classes]
      class_str = ' '.join([c for c in classes if c])

      if class_str:
        return class_str
      
    return (self.attr_vals_map.get(attr_name) or {}).get(val) or val