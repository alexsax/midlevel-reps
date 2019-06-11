class SensorPack(dict):
    ''' Fun fact, you can slice using np.s_. E.g.
        sensors.at(np.s_[:2])
    '''
    
    def at(self, val):
        return SensorPack({k: v[val] for k, v in self.items()})
    
    def apply(self, lambda_fn):
        return SensorPack({k: lambda_fn(k, v) for k, v in self.items()})
    
    def size(self, idx, key=None):
        assert idx == 0, 'can only get batch size for SensorPack'
        if key is None:
            key = list(self.keys())[0]
        return self[key].size(idx)
        