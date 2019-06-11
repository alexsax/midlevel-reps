import torch
from tnt.torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
from . import MeterLogger
from .. import meter as Meter
import numpy as np

class VisdomMeterLogger(MeterLogger):
    ''' A class to package and visualize meters.

    Args:
        server: The uri of the Visdom server
        env: Visdom environment to log to.
        port: Port of the visdom server.
        title: The title of the MeterLogger. This will be used as a prefix for all plots.
        plotstylecombined: Whether to plot train/test curves in the same window.
    '''
    def __init__(self, server="localhost", env='main', port=8097, title="DNN", nclass=21, plotstylecombined=True, log_to_filename=None, loggers=('train', 'val')):
        super(VisdomMeterLogger, self).__init__()        
        self.server = server
        self.env = env
        self.port = port
        self.title = title
        self.logger = {}
        for logger in loggers:
            self.logger[logger] = {}
        self.plotstylecombined = plotstylecombined
        self.log_to_filename = log_to_filename
        self.metername_to_ptype = {}

    def __addlogger(self, meter, ptype):
        first_logger = None
        for logger_name, logger in self.logger.items():
            if ptype == 'stacked_line':
                opts = {'title': '{} {} ({})'.format(self.title, meter, logger_name),
                        'fillarea': True,
                        'legend': self.meter[logger_name][meter].keys}
                logger[meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                 port=self.port, log_to_filename=self.log_to_filename,
                                                 opts=opts)
            elif ptype == 'line':
                if self.plotstylecombined:
                    if first_logger is None:
                        opts = {'title': self.title + ' ' + meter}
                        logger[meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                         port=self.port, log_to_filename=self.log_to_filename,
                                                         opts=opts)
                    else:
                        logger[meter] = self.logger[first_logger][meter]
                else:
                    opts = {'title': self.title + '{} '.format(logger_name) + meter}
                    logger[meter] = VisdomPlotLogger(ptype, env=self.env, server=self.server,
                                                     port=self.port, log_to_filename=self.log_to_filename,
                                                     opts=opts)
            elif ptype == 'heatmap':
                names = list(range(self.nclass))
                opts = {'title': '{} {} {}'.format(self.title, logger_name, meter) + meter, 'columnnames': names, 'rownames': names}
                logger[meter] = VisdomLogger('heatmap', env=self.env, server=self.server,
                                             port=self.port, log_to_filename=self.log_to_filename,
                                             opts=opts)

            # >>> # Image example
            # >>> img_to_use = skimage.data.coffee().swapaxes(0,2).swapaxes(1,2)
            # >>> image_logger = VisdomLogger('image')
            # >>> image_logger.log(img_to_use)
            elif ptype == 'image':
                opts = {'title': '{} {} {}'.format(self.title, logger_name, meter) + meter}
                logger[meter] = VisdomLogger(ptype, env=self.env, server=self.server,
                                             port=self.port, log_to_filename=self.log_to_filename,
                                             opts=opts)

            # >>> # Histogram example
            # >>> hist_data = np.random.rand(10000)
            # >>> hist_logger = VisdomLogger('histogram', , opts=dict(title='Random!', numbins=20))
            # >>> hist_logger.log(hist_data)
            elif ptype == 'histogram':
                opts = {'title': '{} {} {}'.format(self.title, logger_name, meter) + meter, 'numbins': 20}
                logger[meter] = VisdomLogger(ptype, env=self.env, server=self.server,
                                             port=self.port, log_to_filename=self.log_to_filename,
                                             opts=opts)
            elif ptype == 'text':
                opts = {'title': '{} {} {}'.format(self.title, logger_name, meter) + meter}
                logger[meter] =  VisdomTextLogger(env=self.env, server=self.server,
                                                           port=self.port, log_to_filename=self.log_to_filename,
                                                           update_type='APPEND',
                                                           opts=opts)
            elif ptype =='video':
                opts = {'title': '{} {} {}'.format(self.title, logger_name, meter) + meter}
                logger[meter] = VisdomLogger(ptype, env=self.env, server=self.server,
                                             port=self.port, log_to_filename=self.log_to_filename,
                                             opts=opts)


    def add_meter(self, meter_name, meter, ptype=None):
        super(VisdomMeterLogger, self).add_meter(meter_name, meter)
        # for key in self.writer.keys():
        #     self.metername_to_ptype[meter] = ptype
        self.metername_to_ptype[meter_name] = ptype
        if ptype:  # Use `ptype` for manually selecting the plot type
            self.__addlogger(meter_name, ptype)
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

            if isinstance(meter, Meter.ConfusionMeter) or self.metername_to_ptype[meter_name] in ['histogram', 'image', 'text']:
                self.logger[mode][meter_name].log(val)
            elif isinstance(self.meter[mode][meter_name], Meter.MultiValueSummaryMeter):
                self.logger[mode][meter_name].log( np.array([iepoch]*len(val)), np.array(np.cumsum(val)), name=mode) # keep mean
            elif meter_name in self.metername_to_ptype and self.metername_to_ptype[meter_name] == 'video':
                self.logger[mode][meter_name].log(videofile=val)  # video takes in a string
            else:
                self.logger[mode][meter_name].log(iepoch, val, name=mode)
            self.meter[mode][meter_name].reset()
