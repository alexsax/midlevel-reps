import collections
import torch
import pprint
import string

remove_whitespace = str.maketrans('', '', string.whitespace)

def cfg_to_md(cfg, uuid):
    ''' Because tensorboard uses markdown'''
    return uuid + "\n\n    " + pprint.pformat((cfg)).replace("\n", "    \n").replace("\n \'", "\n    \'") + ""

def is_interactive():
    try:
        ip = get_ipython()
        return ip.has_trait('kernel')
    except:
        return False

def is_cuda(model):
    return next(model.parameters()).is_cuda


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        self._keys, self._vals = zip(*adict.items())
        self._keys, self._vals = list(self._keys), list(self._vals)

    def keys(self):
        return self._keys

    def vals(self):
        return self._vals


def compute_weight_norm(parameters):
    ''' no grads! '''
    total = 0.0
    count = 0
    for p in parameters:
        total += torch.sum(p.data**2)
        # total += p.numel()
        count += p.numel()
    return (total / count)

def get_number(name):
    """
    use regex to get the first integer in the name
    if none exists, return -1
    """
    try:
        num = int(re.findall("[0-9]+", name)[0])
    except:
        num = -1
    return num


def update_dict_deepcopy(d, u):  # we need a deep dictionary update
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_dict_deepcopy(d.get(k, {}), v)
        else:
            d[k] = v
    return d

