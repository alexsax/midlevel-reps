# from __future__ import print_function

from random import choice
from time import sleep

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
import scipy
import scipy.misc

import numpy as np
import math
import os
from os import listdir
from os.path import isfile, join

import math
import random

from vizdoom import *
import gym
from gym import spaces


from .utils.doom import DoomRoom
from .utils import commands
from .vizdoom_env import BaseVizdoomEnv, DoomAgent, VIZDOOM_NOOP
from ..base_embodied_env import BaseEmbodiedEnv

from .scenarios.beechwood_map_cfg import MapCfg


VIZDOOM_NOOP = [False, False, False]
# actions = [[True, False, False], [False, True, False], [False, False, True]]
actions = [[True, False, False], [False, True, False], [False, False, True]]


randomize_textures = {'ALL': 0, 'TRAIN_ONLY': 1, 'TEST_ONLY': 2}

RGB = 'RGB'
DEPTH = 'D'
LABELS = 'L'
OBJECT_ID_NAME = 'OBJECT_ID_NAME'
OBJECT_ID_NUM = 'OBJECT_ID_NUM'
OBJECT_LOC_3D = 'OBJECT_LOC_3D'
OBJECT_LOC_BBOX = 'OBJECT_LOC_BBOX'
TIME = 'TIME'
GOAL = 'GOAL'
AUTOMAP = 'AUTOMAP'

INPUT_TYPE_TO_SENSOR_NAME = {
    RGB: 'color',
    DEPTH: 'depth',
    LABELS: 'object_id',
    OBJECT_ID_NAME: 'object_id_name',
    OBJECT_ID_NUM: 'object_id_num',
    OBJECT_LOC_3D: 'object_loc_3d',
    OBJECT_LOC_BBOX: 'object_loc_bbox',  # x, y, w, h
    TIME: 'time',
    GOAL: 'goal',
    AUTOMAP: 'map'
}

FPS    = 30
EXTREME_VAL = 10000
DEFAULT_N_ACTIONS = 3
# DEFAULT_N_ACTIONS = 1
DISTANCE_TOLERANCE_NORMALIZED_COORDS = 0.2
ALLOW_VISUAL_EXPLORATION = True

VIABLE_SCREEN_RESOLUTIONS = [ScreenResolution.RES_640X480,
                             ScreenResolution.RES_320X240,
                             ScreenResolution.RES_160X120]

TORCH_RADIUS = 20
AGENT_RADIUS = 40
OCCUPANCY_VOXEL_LENGTH = 40


from .scenarios.beechwood_map_cfg import MapCfg

import math

# Visual navigation
DOOM_FOV_X = math.pi / 2
DOOM_FOV_Y = DOOM_FOV_X * 120 / 160
F_X = math.tan(DOOM_FOV_X/2.0) * 0.5
F_Y = math.tan(DOOM_FOV_Y/2.0) * 0.5
DEPTH_FOR_SENSING_MAX = 30
DEPTH_FOR_SENSING_MIN = 0
DEPTH_SCALE = 300. / 42.


class VizdoomExplorationEnv(BaseVizdoomEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {(): 2,
                (ord('a'),): 0,
                (ord('d'),): 1,
                (ord('w'),): 3,
                (ord('s'),): 4,
                (ord('q'),): 5,
                (ord('e'),): 6}
        return keys


    def __init__(self, 
                 map_cfg=None,
                 occupancy_voxel_length=OCCUPANCY_VOXEL_LENGTH,
                 allow_visual_exploration=ALLOW_VISUAL_EXPLORATION,
                 **kwargs):
        '''
            Args:
                wad_name: .wad file to use
                sensor_type: subset of ['RGB', 'D', 'L'] 
                    RGB: RGB image as uint8 
                    D: Depth as uint8
                    L: Labels (semantic, pixelwise)
                randomize_textures: wad must support randomizing textures
                randomize_maps: (map_number_low, map_number_high)
        '''
        if map_cfg is None:
            self.map_cfg = MapCfg()
        self.occupancy_voxel_length = occupancy_voxel_length
        self.allow_visual_exploration = allow_visual_exploration
        if 'input_types' in kwargs and self.allow_visual_exploration:
            kwargs['input_types'] = list(kwargs['input_types']) + [DEPTH]
        else: 
            kwargs['input_types'] = (RGB, TIME, AUTOMAP, DEPTH)
        super().__init__(wad_name=self.map_cfg.wad, **kwargs) 
        if self.screen_resolution == ScreenResolution.RES_320X240:
            self.screen_dim = (320, 240)
        else:
            raise NotImplementedError("Exploration accepts screen res 320x240")
        # self.game.init()
        # self._randomize_textures(randomize_textures)


    def reset(self,
              agent_location=None,
              goal_location=None,
              save_replay_file_path=""):
        if save_replay_file_path:
            save_replay_file_path = save_replay_file_path + ".lmp"
        self.episode_number += 1
        self.total_reward = 0
        self.last_reward = 0
        self.step_count = 0
        self.action_list = []
        if self.randomize_maps is not None:
            low, high = self.randomize_maps
            map_no = 'map{0:02d}'.format(
                np.random.randint(high - low + 1) + low )
            self.map_no = map_no
            self.game.set_doom_map(map_no)

        # agent_location = (-24, 660) #TODO(remove)
        # Make agent
        if agent_location is not None:
            agent_x, agent_y = agent_location
        else:
            agent_x, agent_y = self.map_cfg.task_configs['exploration']['starting_location_space'].sample()[:2]
        
        self.game.new_episode()

        # Send game commands at the end. Otherwise, multiple spawns will show up on the minimap
        self.game.send_game_command("pukename player_spawn")
        orientation = choice([0,1,2,3])
        # orientation = 2 #TODO(remove) 
        commands.spawn_agent(self.game, agent_x, agent_y, orientation=orientation)
        self.agent = DoomAgent(agent_x, agent_y, orientation * 90)                
        self._randomize_textures(self.randomize_textures)
        _game_reward = self.game.make_action(VIZDOOM_NOOP, 1)
        
        self.state = self.game.get_state()
        self.occupancy_map = OccupancyMap(self.map_cfg.X_OFFSET,
                                          self.map_cfg.X_OFFSET + self.map_cfg.MAP_SIZE_X,
                                          self.map_cfg.Y_OFFSET,
                                          self.map_cfg.Y_OFFSET + self.map_cfg.MAP_SIZE_Y,
                                          self.occupancy_voxel_length)
        self.obs = self._get_obs()
        self.update_occupancy_map(self.obs)
        return self.obs


    def step(self, action_id):
        # action_id = 0 # TODO: removes
        if self.visualize:
            _game_reward = 0
            for _ in range(self.repeat_count):
                _game_reward += self.game.make_action(actions[action_id], 1)
                sleep(self.interactive_delay)
        else:
            _game_reward = self.game.make_action(actions[action_id], self.repeat_count)
    
        self.step_count += self.repeat_count
        self.action_list.append(action_id)
        self.state = self.game.get_state()
        self.agent = DoomAgent(**self._get_agent_frame_of_reference())
        # self.last_reward = self._compute_reward(_game_reward)
        
        self.obs = self._get_obs()
        self.last_reward = self.update_occupancy_map(self.obs)
        self.total_reward += self.last_reward
        # raise NotImplementedError
        return self.obs, self.last_reward, self.done, self.info
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            assert len(set(self.obs.keys()).intersection(set(['color', 'map']))) == 2            
            _color = [v for k, v in self.obs.items() if k == 'color'][0]
            # _color = [v for k, v in self.obs.items() if k == 'depth'][0]
            # _color = np.concatenate([_color]*3, axis=2)
            _color[self.screen_dim[1]//2:,self.screen_dim[0]//2,0] = 255
            _color[self.max_idx-1:self.max_idx+1,self.screen_dim[0]//2 - 15:self.screen_dim[0]//2+15,2] = 255
            
            _map = [v for k, v in self.obs.items() if k == 'map'][0]
            _occupancy_map = np.stack([self._render_occupancy_map(_map.shape[:2])*255]*3, axis=2)
            alpha = 0.5
            _map = np.uint8((alpha) * _occupancy_map + (1. - alpha) * _map)
            return np.concatenate([_color, _map] , axis=1)
            # return np.concatenate([o for k, o in self.obs.items() if k in ['color', 'map']],
            #           axis=1)
        return self.obs['color']
    
    def _render_occupancy_map(self, new_size):
        min_len = int(min(new_size) * 0.9) # Doom pads the area around the map
        resize_shape = [min_len] * 2
        im = scipy.misc.imresize(np.flip(np.uint8(self.occupancy_map.bitmap.T), axis=0),
                                 resize_shape, interp='nearest')
        # print(new_size[0]//2,new_size[0]//2+min_len, new_size[1]//2,new_size[1]//2+min_len)
        x_start = (new_size[0] - min_len) // 2
        y_start = (new_size[1] - min_len) // 2
        full_im = cv2.copyMakeBorder(im, x_start, x_start, y_start, y_start, cv2.BORDER_CONSTANT, value=0)
        return full_im

    @property
    def done(self):
        done = self.game.is_episode_finished()
        if self.max_actions is not None and self.step_count >= self.max_actions:
            done = True
        # print("is_done:", done)
        # print(self.game.is_episode_finished(), 
        #       self.distance_to_a_goal(ord=1), self.distance_to_goal_thresh, 
        #       self.max_actions, self.step_count,
        #       done)
        return done
    
    def _compute_reward(self, _game_reward, tol=1):
        
        if not self.occupancy_map.get(self.agent.x, self.agent.y):
            return 1
        return 0

    def update_occupancy_map(self, obs):
        orig_found = np.sum(self.occupancy_map.bitmap)
        depth_im = np.clip(obs[INPUT_TYPE_TO_SENSOR_NAME[DEPTH]], DEPTH_FOR_SENSING_MIN, DEPTH_FOR_SENSING_MAX)
        xyz = self._reproject_depth_image(depth_im.squeeze())
        xx, yy = rotate_origin_only(xyz[self.screen_dim[1]//2:, self.screen_dim[0]//2, :], math.radians(90) - math.radians(self.agent.theta))
        xx += self.agent.x
        yy += self.agent.y        
        for x, y in zip(xx, yy):
            self.occupancy_map.update(x, y)
        self.occupancy_map.update(self.agent.x, self.agent.y)
        n_revealed = np.sum(self.occupancy_map.bitmap) - orig_found
        return float(n_revealed)
    
    
    def _reproject_depth_image(self, depth, unit_scale=6.6666):
        """Transform a depth image into a point cloud with one point for each
        pixel in the image, using the camera transform for a camera
        centred at cx, cy with field of view fx, fy.
        """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        # valid = (depth > 0) & (depth < 255)
        # print(depth[self.screen_dim[1]//2:,self.screen_dim[0]//2])
        self.max_idx = np.argmax(depth[self.screen_dim[1]//2:,self.screen_dim[0]//2]) + self.screen_dim[1]//2
        # print(np.argmax(depth[self.screen_dim[1]//2:,self.screen_dim[0]//2]),
        #       np.max(depth[self.screen_dim[1]//2:,self.screen_dim[0]//2]))
        y = depth * unit_scale
        x = y * ((c - self.screen_dim[0]//2) / F_X / self.screen_dim[0]//2)
        z = y * ((r - self.screen_dim[1]//2) / F_Y / self.screen_dim[1]//2)
        return np.dstack((x, y, z))

class OccupancyMap(object):
    
    def __init__(self, xmin, xmax, ymin, ymax, voxel_length):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.voxel_length = voxel_length      
        self.size_x = math.ceil((xmax - xmin) / voxel_length)
        self.size_y = math.ceil((ymax - ymin) / voxel_length)
        self.bitmap = np.full((self.size_x, self.size_y), False, dtype=np.bool_)
        
    def update(self, x, y, val=True):
        idx_x, idx_y = self._get_voxel_coords(x, y)
        self.bitmap[idx_x, idx_y] = val
        return self.bitmap
    
    def get(self, x, y):
        idx_x, idx_y = self._get_voxel_coords(x, y)
        return self.bitmap[idx_x, idx_y] 

    def _get_voxel_coords(self, x, y):
        idx_x = int((x - self.xmin) / self.voxel_length)
        idx_y = int((y - self.ymin) / self.voxel_length)
        idx_x = np.clip(idx_x, 0, self.size_x - 1)
        idx_y = np.clip(idx_y, 0, self.size_y - 1)
        if idx_y < 0 or idx_y < 0:
            raise ValueError("Trying to set occupancy in grid cell ({}, {})".format(idx_x, idx_y))
        return idx_x, idx_y


def point_cloud(depth, unit_scale=DEPTH_SCALE):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    # valid = (depth > 0) & (depth < 255)
    # print(depth.max())
    y = depth * unit_scale
    x = y * ((c - C_X) / F_X / C_X)
    z = y * ((r - C_Y) / F_Y / C_Y)
    return np.dstack((x, y, z))

def polar2cart(r, theta, phi):
    return np.stack([
         r * np.sin(theta) * np.cos(phi),
         r * np.sin(theta) * np.sin(phi),
         r * np.cos(theta)
    ])

def rotate_origin_only(xy, radians):
    x, y = xy[:,:2].T
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)
    return xx, yy