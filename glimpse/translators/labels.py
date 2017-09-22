from glimpse.translators import one_hot_for_index


def storage2tree(classes, connections):
  return attach_branches(classes, connections, [0])[0]


def attach_branches(classes, connections, indexes):
  els = []
  for i in indexes:
    el_class = classes[i]

    if el_class == -1:
      continue

    new_el = [one_hot_for_index(el_class)]
    el_conns = [j for j in connections[i] if j != -1]

    if el_conns:
      new_el_children = attach_branches(classes, connections, el_conns)
    else:
      new_el_children = []

    new_el.append(new_el_children)

    els.append(new_el)

  return els