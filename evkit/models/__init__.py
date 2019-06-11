import torch.nn as nn

class SingleSensorModule(nn.Module):
    def __init__(self, module, sensor_name):
        super().__init__()
        self.module = module
        self.sensor_name = sensor_name
    
    def __call__(self, obs):
        # return {self.sensor_name: self.module(obs[self.sensor_name])}
        return self.module(obs[self.sensor_name])
