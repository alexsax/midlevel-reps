import math
from . import meter
import torch
import numpy as np


class ValueSummaryMeter(meter.Meter):
    def __init__(self):
        super(ValueSummaryMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = self.sum + 0.0  # This is to force a copy in torch/numpy
            self.min = self.mean + 0.0
            self.max = self.mean + 0.0
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))
            self.min = np.minimum(self.min, value)
            self.max = np.maximum(self.max, value)

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.min = np.nan
        self.max = np.nan
        
    def __str__(self): 
        old_po = np.get_printoptions()
        np.set_printoptions(precision=3)
        res = "mean(std) {} ({}) \tmin/max {}/{}\t".format(
            *[np.array(v) for v in [self.mean, self.std, self.min, self.max]])
        np.set_printoptions(**old_po)
        return res
