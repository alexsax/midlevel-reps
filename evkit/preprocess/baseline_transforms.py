from collections import defaultdict
import numpy as np
import skimage
import torchvision as vision
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing.dummy as mp
import multiprocessing
from gym import spaces

from evkit.sensors import SensorPack


def blind(output_size, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    def _thunk(obs_space):
        pipeline = lambda x: torch.zeros(output_size)
        return pipeline, spaces.Box(-1, 1, output_size, dtype)
    return _thunk



def pixels_as_state(output_size, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    def _thunk(obs_space):
        obs_shape = obs_space.shape
        obs_min_wh = min(obs_shape[:2])
        output_wh = output_size[-2:]  # The out
        processed_env_shape = output_size

        base_pipeline = vision.transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.CenterCrop([obs_min_wh, obs_min_wh]),
            vision.transforms.Resize(output_wh)])
        
        grayscale_pipeline = vision.transforms.Compose([
            vision.transforms.Grayscale(),
            vision.transforms.ToTensor(),
            RESCALE_0_1_NEG1_POS1,
        ])
        
        rgb_pipeline = vision.transforms.Compose([
            vision.transforms.ToTensor(),
            RESCALE_0_1_NEG1_POS1,
        ])

        def pipeline(x):
            base = base_pipeline(x)
            rgb = rgb_pipeline(base)
            gray = grayscale_pipeline(base)
            
            n_rgb = output_size[0] // 3
            n_gray = output_size[0] % 3
            return torch.cat([rgb] * n_rgb + [gray] * n_gray)
        return pipeline, spaces.Box(-1, 1, output_size, dtype)
    return _thunk



