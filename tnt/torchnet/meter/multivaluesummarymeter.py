import math
from . import meter, ValueSummaryMeter
import torch
import numpy as np


class MultiValueSummaryMeter(ValueSummaryMeter):
    def __init__(self, keys):
        '''
            Args:
                keys: An iterable of keys
        '''
        super(MultiValueSummaryMeter, self).__init__()
        self.keys = list(keys)
