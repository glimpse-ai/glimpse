import os
import json
from glimpse.helpers.definitions import dml_dir, vocab_path

pad_char = '?'


def create_vocab():
  dml_files = [f for f in os.listdir(dml_dir) if f.endswith('.dml')]
  char_map = {}
  
  i = 1
  for f in dml_files:
    if not i % 50:
      print 'Done with {} of {}'.format(i, len(dml_files))
    
    with open('{}/{}'.format(dml_dir, f)) as dml:
      text = dml.read()
    
    for char in text:
      char_map[char] = 1
      
    i += 1

  vocab = char_map.keys()
  vocab.sort()
  vocab.append(pad_char)
  
  print 'Created vocab of length {}'.format(len(vocab))

  with open(vocab_path, 'w+') as f:
    f.write(json.dumps(vocab, sort_keys=True, indent=2))
  

if __name__ == '__main__':
  create_vocab()