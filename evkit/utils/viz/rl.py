from .core import rescale_image

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_stacked_frames(stacked_observations, rollout_idx=0, axis_order='RCWH', im_val_scale=(0.0, 1.0), noshow=False):
    ''' Parameters:
        stacked_obs: a StackedObservationStorage object
        rollout_idx: which rollout to visualize
        axis_order: RCWH = RolloutIdx x Channel x Width x Height
        im_val_scale: If image is a float, the range of the inputs
        noshow: Not implemented
    '''
    if noshow:
        raise NotImplementedError("Bug! Noshow not yet implemented")
    assert axis_order in ['RCWH', 'RWHC']
    fig, ax = plt.subplots(figsize=(15, 15))
    n_stacked = stacked_observations.n_stack
    for i, start in enumerate(range(0, stacked_observations.obs_shape[0], stacked_observations.env_shape_dim0)):
        frame = stacked_observations.obs[rollout_idx, start:start+stacked_observations.env_shape_dim0]
        if frame.dtype in [np.float32, torch.float32]:
            frame = rescale_image(frame, (0.0, 1.0), current_scale=im_val_scale)
        if axis_order == 'RCWH':
            frame = np.rollaxis(frame, 0, 3)
        if frame.shape[-1] == 1: 
            frame = frame.squeeze(-1)
        plt.subplot(1, n_stacked, i + 1)
        print(frame.shape)
        plt.imshow(frame)
        plt.title('{} of {}'.format(i, n_stacked))
    plt.tight_layout()
    if not noshow:
        plt.show()

   
def plot_rollout_frames(rollouts, rollout_idx=0, axis_order='RCWH', im_val_scale=(0.0, 1.0), noshow=False,
                       n_channels=3, timestep=0):
    ''' Parameters:
        stacked_obs: a StackedObservationStorage object
        rollout_idx: which rollout to visualize
        axis_order: RCWH = RolloutIdx x Channel x Width x Height
        im_val_scale: If image is a float, the range of the inputs
        noshow: Not implemented
    '''
    obs = rollouts.observations[timestep]
    n_stacked = obs.shape[1] // n_channels
    if noshow:
        raise NotImplementedError("Bug! Noshow not yet implemented")
    assert axis_order in ['RCWH', 'RWHC']
    fig, ax = plt.subplots(figsize=(15, 15))
    for i, start in enumerate(range(0, obs.shape[1], n_channels)):
        frame = obs[rollout_idx, start:start+n_channels]
        
        if frame.dtype in [np.float32, torch.float32]:
            frame = rescale_image(frame, (0.0, 1.0), current_scale=im_val_scale)
        if axis_order == 'RCWH':
            frame = np.rollaxis(frame, 0, 3)
        if frame.shape[-1] == 1: 
            frame = frame.squeeze(-1)
        plt.subplot(1, n_stacked, i + 1)
        plt.imshow(frame)
        plt.title('{} of {}'.format(i, n_stacked))
    plt.tight_layout()
    if not noshow:
        plt.show()

from teas.utils.viz.core import rescale_image

def plot_rollout_sensor_frames(rollouts, rollout_idx=0, axis_order='RCWH', im_val_scale=(0.0, 1.0), noshow=False,
                       n_channels=3, timestep=0, sensor_name='color__pinhole'):
    ''' Parameters:
        stacked_obs: a StackedObservationStorage object
        rollout_idx: which rollout to visualize
        axis_order: RCWH = RolloutIdx x Channel x Width x Height
        im_val_scale: If image is a float, the range of the inputs
        noshow: Not implemented
    '''
    obs = rollouts.observations[sensor_name][timestep]
    n_stacked = obs.shape[1] // n_channels
    if noshow:
        raise NotImplementedError("Bug! Noshow not yet implemented")
    assert axis_order in ['RCWH', 'RWHC']
    fig, ax = plt.subplots(figsize=(15, 15))
    for i, start in enumerate(range(0, obs.shape[1], n_channels)):
        frame = obs[rollout_idx, start:start+n_channels]
        
        if frame.dtype in [np.float32, torch.float32]:
            frame = rescale_image(frame, (0.0, 1.0), current_scale=im_val_scale)
        if axis_order == 'RCWH':
            frame = np.rollaxis(frame, 0, 3)
        if frame.shape[-1] == 1: 
            frame = frame.squeeze(-1)
        plt.subplot(1, n_stacked, i + 1)
        plt.imshow(frame)
        plt.title('{} of {}'.format(i, n_stacked))
    plt.tight_layout()
    if not noshow:
        plt.show()