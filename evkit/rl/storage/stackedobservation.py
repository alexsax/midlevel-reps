from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch


from evkit.sensors import SensorDict





def _is_sets_equal(a, b):
    set_a = set(a)
    set_b = set(b)
    return set_a.issubset(set_b) and set_b.issubset(set_a)

def _is_subset_of(a, b):
    set_a = set(a)
    set_b = set(b)
    return set_a.issubset(set_b)


class StackedSensorDictStorage(object):
    ''' A FILO queue, internally stored as a dict of tensors. '''

    def __init__(self, n_rollouts, n_stack, sensor_obs_shapes):
        ''' 
            Parameters:
                n_rollouts: Number of rollouts to store
                n_stack: Number of observations to stack. Either an int or else a dict with each sensor name.
                sensor_obs_shapes: Dict from sensor names to sensor shapes
        '''
        self.sensor_observations = {}
        self.sensor_names = set()
        self.n_rollouts = n_rollouts
        self.n_stack = n_stack
        if hasattr(n_stack, '__getitem__'):
            self.n_stack = n_stack
        else:
            self.n_stack = defaultdict(lambda: n_stack)

        for sensor_name, sensor_obs_shape in sensor_obs_shapes.items():
            self[sensor_name] = StackedTensorStorage(n_rollouts, self.n_stack[sensor_name], sensor_obs_shape)
            self.sensor_names.add(sensor_name)
        self.obs_shape = {k: v.obs_shape for k, v in self.sensor_observations.items()}
    
    def insert(self, obs, mask_out_done=None):
        '''
            Parameters:
                obs: dict of sensor_name -> sensor_obs
                mask_out_done: mask blocking completed episodes
        '''
        assert len(obs) > 0, 'obs cannot be empty'
        assert _is_subset_of(self.sensor_names, obs.keys()), \
            'Updating observations must update all sensors: {} ! subset of {}'.format(self.sensor_names, set(obs.keys()))
        
        for sensor_name in self.sensor_names:
            sensor_obs = obs[sensor_name]
            self[sensor_name].insert(sensor_obs, mask_out_done)
    
    def clear_done(self, mask_out_done):
        assert _is_sets_equal(mask_out_done.keys(), self.sensor_names), f'Clearing observations must address all sensors, clearing {mask_out_done.keys()} not {self.sensor_names}'
        for sensor_name, sensor_mask in mask_out_done.items():
            self[sensor_name].clear_done(sensor_mask)

    def peek(self):
        return self.sensor_observations
    
    def cuda(self):
        for sensor_name in self.sensor_names:
            self[sensor_name].cuda()
        return self
    
    def __getitem__(self, sensor_name):
        return self.sensor_observations[sensor_name]
    
    def __setitem__(self, sensor_name, value):
        self.sensor_observations[sensor_name] = value
        

        
        
import time

class StackedTensorStorage(object):
    ''' A FILO queue, internally stored as a tensor. '''
    def __init__(self, n_rollouts, n_stack, env_obs_shape):
        '''
            Parameters:
                n_rollouts: number of rollouts to store
                n_stack: Number of stacked frames each obs
                env_obs_shape: A dict of sensor_names -> shapes
        '''
        self.obs_shape = (env_obs_shape[0] * n_stack, *env_obs_shape[1:])
        self.obs = torch.zeros(n_rollouts, *self.obs_shape)
        self.env_shape_dim0 = env_obs_shape[0]
        self.n_rollouts = n_rollouts
        self.n_stack = n_stack
    
    def insert(self, obs, mask_out_done=None):
        if mask_out_done is not None:
            self.clear_done(mask_out_done)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif isinstance(obs, list):
            obs = torch.stack(obs, dim=0)
        if self.n_stack > 1: #shift
            self.obs[:, :-self.env_shape_dim0] = self.obs[:, self.env_shape_dim0:]
        self.obs[:, -self.env_shape_dim0:] = obs

    def clear_done(self, mask_out_done):
        if self.obs.dim() == 4:
            # self.obs *= mask_out_done.unsqueeze(2).unsqueeze(2)
            self.obs[(mask_out_done < 0.5)[:, 0]] = 0.0
        else:
            self.obs *= mask_out_done
    
    def peek(self):
        return self.obs
    
    def cuda(self):
        self.obs = self.obs.cuda()
        return self
