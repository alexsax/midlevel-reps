import math
from . import meter
from .valuesummarymeter import ValueSummaryMeter
import numpy as np
import warnings

class AverageValueMeter(ValueSummaryMeter):
    def __init__(self):
        warnings.warn('AverageValueMeter is deprecated in favor of ValueSummaryMeter and will be removed in a future version', FutureWarning)
        super(AverageValueMeter, self).__init__()
