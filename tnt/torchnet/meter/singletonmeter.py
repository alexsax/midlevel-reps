from . import meter


class SingletonMeter(meter.Meter):
    '''Stores exactly one value which can be regurgitated'''

    def __init__(self, maxlen=1):
        super(SingletonMeter, self).__init__()
        self.__val = None

    def reset(self):
        '''Resets the meter to default settings.'''
        old_val = self.__val
        self.__val = None
        return old_val
        
    def add(self, value):
        '''Log a new value to the meter

        Args:
            value: Next restult to include.
        '''
        self.__val = value
        
    def value(self):
        '''Get the value of the meter in the current state.'''
        return self.__val

