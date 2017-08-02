from glimpse.helpers.definitions import templates_dir
from glimpse.helpers import soupify
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def wrap_body(body):
  with open('{}/wrapper.html'.format(templates_dir)) as f:
    wrapper = f.read()

  soup = soupify(body)
  
  all_bb_attr = 'all-bb'
  extra_css = ''

  if (soup.body.attrs or {}).get(all_bb_attr):
    extra_css = '\n\n\t\tbody * {\n\t\t\tbox-sizing: border-box;\n\t\t}'
    soup.body.attrs.pop(all_bb_attr)

  return wrapper.replace('EXTRA_CSS', extra_css).replace('<BODY>', soup.encode('utf-8'))