import copy 
import numpy as np
import torch
import warnings

from .. import meter as Meter

class MeterLogger(object):
    ''' A class to package and print meters. '''
    def __init__(self, modes=('train', 'val')):
        self.modes = list(modes)
        self.meter = {}
        self.logger = {}
        for mode in modes:
            self.meter[mode] = {}
            self.logger[mode] = {}
        self.timer = Meter.TimeMeter(None)
        self.metername_to_ptype = {}

    def _ver2tensor(self, target):
        target_mat = torch.zeros(target.shape[0], self.nclass)
        for i, j in enumerate(target):
            target_mat[i][j] = 1
        return target_mat

    def _to_tensor(self, var):
        if isinstance(var, torch.autograd.Variable):
            var = var.data
        if not torch.is_tensor(var):
            if isinstance(var, np.ndarray):
                var = torch.from_numpy(var)
            else:
                var = torch.Tensor([var])
        return var

    def add_meter(self, meter_name, meter):
        for mode in self.modes:
            self.meter[mode][meter_name] = copy.deepcopy(meter)


    def update_meter(self, output, target=None, meters={'METER_NAME_HERE'}, phase='train'):
        for meter_name in meters:
            assert meter_name in self.meter[phase].keys(), "Unrecognized meter name {}".format(meter_name)
            meter = self.meter[phase][meter_name]
            if not isinstance(meter, Meter.SingletonMeter): # Singleton stores as-is
                output = self._to_tensor(output)
                if target is not None:
                    target = self._to_tensor(target)
            if isinstance(meter, Meter.APMeter) or \
               isinstance(meter, Meter.mAPMeter) or \
               isinstance(meter, Meter.ConfusionMeter):
                assert target is not None, "Meter '{}' of type {} requires 'target' is not None".format(meter_name, type(meter))
                target_th = self._ver2tensor(target)
                meter.add(output, target_th)
            elif target is not None:
                meter.add(output, target)
            else:
                meter.add(output)


    def peek_meter(self, phase='train'):
        '''Returns a dict of all meters and their values.'''
        result = {}
        for key in self.meter[phase].keys():
            val = self.meter[phase][key].value()
            val = val[0] if isinstance(val, (list, tuple)) else val
            result[key] = val
        return result

    def reset_meter(self, meterlist=None, phase='train'):
        self.timer.reset()
        if meterlist is None:
            meterlist = self.meter[phase].keys()
        for meter_name in meterlist:
            assert meter_name in self.meter[phase].keys(), "Unrecognized meter name {}".format(meter_name)
            self.meter[phase][meter_name].reset()

    def print_meter(self, mode, iepoch, ibatch=1, totalbatch=1, meterlist=None):
        assert mode in self.modes, f'{mode} is not any phase'
        pstr = "%s:\t[%d][%d/%d] \t"
        tval = []
        tval.extend([mode, iepoch, ibatch, totalbatch])
        if meterlist is None:
            meterlist = self.meter[mode].keys()
        for meter_name in meterlist:
            assert meter_name in self.meter[mode].keys(), "Unrecognized meter name {}".format(meter_name)
            meter = self.meter[mode][meter_name]
            if isinstance(meter, Meter.ConfusionMeter):
                continue
            if isinstance(meter, Meter.ClassErrorMeter):
                # Printing for this could be significantly improved
                pstr += "Acc@1 %.2f%% \t Acc@" + str(self.topk) + " %.2f%% \t"
                tval.extend([self.meter[mode][meter_name].value()[0], self.meter[mode][meter_name].value()[1]])
            elif isinstance(meter, Meter.mAPMeter):
                pstr += "mAP %.3f \t"
                tval.extend([self.meter[mode][meter_name].value()])
            elif isinstance(meter, Meter.AUCMeter):
                pstr += "AUC %.3f \t"
                tval.extend([self.meter[mode][meter_name].value()])
            elif isinstance(meter, Meter.ValueSummaryMeter) or isinstance(meter, Meter.MSEMeter):
                pstr += "{}: {}".format(meter_name, self.meter[mode][meter_name])
            elif isinstance(meter, Meter.MultiValueSummaryMeter):
                pstr += "{}: {}".format(meter_name, self.meter[mode][meter_name])
            else:
                warnings.warn("Can't print meter '{}' of type {}".format(meter_name, type(meter)),
                              RuntimeWarning)
        pstr += " %.2fs/its\t"
        tval.extend([self.timer.value()])
        print(pstr % tuple(tval))