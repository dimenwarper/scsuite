import importlib
import matplotlib.colors as colors

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'tan', 'brown', 'salmon']
COLORS += [c for c in colors.cnames if c not in COLORS and c != 'gray'] + ['gray']


def get_color(idx_or_idx_list):
    if hasattr(idx_or_idx_list, '__getitem__') and hasattr(idx_or_idx_list, '__iter__'):
        return [COLORS[idx] for idx in idx_or_idx_list]
    else:
        return COLORS[idx_or_idx_list]


def to_kwargs(dic):
    kwargs = {}
    for k, v in dic.items():
        if type(v) == dict and 'class' in v:
            kwargs[k] = v['class'](**to_kwargs(v['params']))
        else:
            kwargs[k] = v
    return kwargs

def str_to_class(s):
    modnames = s.split('.')
    clsname = modnames[-1]
    module = '.'.join(modnames[:-1])
    return getattr(importlib.import_module(module), clsname)

