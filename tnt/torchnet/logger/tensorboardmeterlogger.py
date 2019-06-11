import os
from . import MeterLogger
from .. import meter as Meter
import numpy as np
import torch
import functools

IS_IMPORTED_TENSORBOARDX = False
try:
    import tensorboardX
    IS_IMPORTED_TENSORBOARDX = True
except:
    pass

class TensorboardMeterLogger(MeterLogger):
    ''' A class to package and visualize meters.

    Args:
        log_dir: Directory to write events to (log_dir/env)
        env: Tensorboard environment to log to.
        plotstylecombined: Whether to plot curves in the same window.
        loggers: All modes: defaults to ['train', 'val']. If plotstylecombined, these will be superimposed in one plot.
    '''
    def __init__(self, env, log_dir=None, plotstylecombined=True, loggers=('train', 'val')):
        super().__init__(modes=loggers) 
        self.env = env
        self.log_dir = os.path.join(log_dir, env)

        self.logger = {}
        self.writer = {}
        for logger in loggers:
            self.logger[logger] = {}
            self.writer[logger] = tensorboardX.SummaryWriter(logdir=self.log_dir + "-{}".format(logger))

        self.metername_to_ptype = {}
        self.plotstylecombined = plotstylecombined

    def __addlogger(self, meter, ptype, kwargs={}):
        for key in self.writer.keys():
            self.metername_to_ptype[meter] = ptype
        if ptype == 'stacked_line':
            raise NotImplementedError("stacked_line not yet implemented for TensorboardX meter")
        elif ptype == 'line':
            if self.plotstylecombined:
                for key in self.writer.keys():
                    self.logger[key][meter] = functools.partial(self.writer[key].add_scalar, tag=meter)
            else:
                for key in self.writer.keys():
                    self.logger[key][meter] = functools.partial(self.writer[key].add_scalar, tag=meter)
        elif  ptype == 'image':
            if self.plotstylecombined:
                for key in self.writer.keys():
                    self.logger[key][meter] = functools.partial(self.writer[key].add_image, tag=meter)
            else:
                for key in self.writer.keys():
                    self.logger[key][meter] = functools.partial(self.writer[key].add_image, tag=meter)
        elif  ptype == 'histogram':
            if self.plotstylecombined:
                for key in self.writer.keys():
                    self.logger[key][meter] = functools.partial(self.writer[key].add_histogram, tag=meter)
            else:
                for key in self.writer.keys():
                    self.logger[key][meter] = functools.partial(self.writer[key].add_histogram, tag=meter)
        elif ptype == 'heatmap':
            raise NotImplementedError("heatmap not yet implemented for TensorboardX meter")
        elif ptype == 'text':
            for key in self.writer.keys():
                self.logger[key][meter] = functools.partial(self.writer[key].add_text, tag=meter)
        elif ptype == 'video':
            for key in self.writer.keys():
                self.logger[key][meter] = functools.partial(self.writer[key].add_video, tag=meter, **kwargs)


    def add_meter(self, meter_name, meter, ptype=None, kwargs={}):
        super().add_meter(meter_name, meter)
        if ptype:  # Use `ptype` for manually selecting the plot type
            self.__addlogger(meter_name, ptype, kwargs)
        elif isinstance(meter, Meter.ClassErrorMeter):
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.mAPMeter):
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.AUCMeter):
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.ConfusionMeter):
            self.__addlogger(meter_name, 'heatmap')
        elif isinstance(meter, Meter.MSEMeter):
            self.__addlogger(meter_name, 'line')
        elif type(meter) == Meter.ValueSummaryMeter:
            self.__addlogger(meter_name, 'line')
        elif isinstance(meter, Meter.MultiValueSummaryMeter):
            self.__addlogger(meter_name, 'stacked_line')
        else:
            raise NotImplementedError("Unknown meter type (and pytpe): {} ({})".format(type(meter), ptype))
            
    def reset_meter(self, iepoch, mode='train', meterlist=None):
        self.timer.reset()
        for meter_name, meter in self.meter[mode].items():
            if meterlist is not None and meter_name not in meterlist:
                continue
            val = self.meter[mode][meter_name].value()
            val = val[0] if isinstance(val, (list, tuple)) else val
            should_reset_and_continue = False
            if isinstance(val, str) or val is None:
                should_reset_and_continue = (val is None)
            elif isinstance(val, np.ndarray):
                should_reset_and_continue = np.isnan(val).any()
            elif isinstance(val, torch.Tensor):
                should_reset_and_continue = torch.isnan(val).any()
            else:
                should_reset_and_continue = np.isnan(val)

            if should_reset_and_continue:
                self.meter[mode][meter_name].reset()
                continue


            if isinstance(meter, Meter.ConfusionMeter):
                self.logger[mode][meter_name].log(val, global_step=iepoch)
            elif 'image' == self.metername_to_ptype[meter_name]:
                try:
                    self.logger[mode][meter_name](img_tensor=val, global_step=iepoch)
                except ValueError as e:
                    print(f'trouble logging {meter_name} {e}')
                    print('probably due to fake 0 data the data is all at 0')
            elif 'histogram' == self.metername_to_ptype[meter_name]:
                try:
                    self.logger[mode][meter_name](values=val, global_step=iepoch)
                except ValueError as e:
                    print(f'trouble logging {meter_name} {e}')
                    print('probably due to fake 0 data the data is all at 0')
            elif 'text' == self.metername_to_ptype[meter_name]:
                if val is not None:
                    self.logger[mode][meter_name](text_string=val, global_step=iepoch)
            elif 'video' == self.metername_to_ptype[meter_name]:
                if val is not None:
                    self.logger[mode][meter_name](vid_tensor=val, global_step=iepoch)
            elif isinstance(self.meter[mode][meter_name], Meter.MultiValueSummaryMeter):
                self.logger[mode][meter_name](scalar_val=np.array(np.cumsum(val), global_step=iepoch)) # keep mean
            else:
                self.logger[mode][meter_name](scalar_value=val, global_step=iepoch)
            self.meter[mode][meter_name].reset()

