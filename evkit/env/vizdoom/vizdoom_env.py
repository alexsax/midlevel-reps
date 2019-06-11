from __future__ import print_function

from random import choice
from time import sleep

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
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
from ..base_embodied_env import BaseEmbodiedEnv


VIZDOOM_NOOP = [False, False, False]
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
VIABLE_SCREEN_RESOLUTIONS = [ScreenResolution.RES_640X480,
                             ScreenResolution.RES_320X240,
                             ScreenResolution.RES_160X120]

class BaseVizdoomEnv(BaseEmbodiedEnv):
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
                 wad_name='semantic_goal_map_dev.wad',
                 input_types=(RGB, TIME, AUTOMAP),
                 episode_start_time=14,
                 episode_timeout=1000,
                 max_actions=None,
                 set_window_visible=False,
                 interactive=False,
                 randomize_textures=1,
                 randomize_maps=None,
                 repeat_count=3,
                 n_actions=DEFAULT_N_ACTIONS,
                 set_automap_buffer_enabled=True,
                 set_render_crosshair=False,
                 set_render_corpses=False,
                 set_render_decals=False,
                 set_render_effects_sprites=False,
                 set_render_hud=False,
                 set_render_messages=False,
                 set_render_minimal_hud=False,
                 set_render_particles=False,
                 set_render_weapon=True,
                 set_screen_resolution=ScreenResolution.RES_320X240,
                 set_sound_enabled=False,
                 max_onscreen_objects=30,
                ):
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
        game = DoomGame()
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        game_path = os.path.join(os.path.dirname(__file__), 'freedoom2.wad')
        assert os.path.isfile(game_path)
        game.set_doom_scenario_path(os.path.join(scenarios_dir, wad_name))

        # if map_cfg is None:
        #     raise ValueError("map_cfg cannot be none--needed to normalize agent coords.")

        # game.set_doom_game_path(game_path)
        game.set_doom_map("map01")

        
        
        # Get observation spaces
        _obs_space = {}
        if set_screen_resolution not in VIABLE_SCREEN_RESOLUTIONS:
            raise NotImplementedError(
                'Some screen resolutions do not work with gym.Monitor. Screen resolution must be one of {}'.format(
                    VIABLE_SCREEN_RESOLUTIONS))
        else:
            game.set_screen_resolution(set_screen_resolution)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_render_hud(set_render_hud)
        game.set_render_minimal_hud(set_render_minimal_hud)  # If hud is enabled
        game.set_render_crosshair(set_render_crosshair)
        game.set_render_weapon(set_render_weapon)
        game.set_render_decals(set_render_decals)
        game.set_render_particles(set_render_particles)
        game.set_render_effects_sprites(set_render_effects_sprites)
        game.set_render_messages(set_render_messages)
        game.set_render_corpses(set_render_corpses)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.set_episode_timeout(episode_timeout)
        game.set_episode_start_time(episode_start_time)
        game.set_window_visible(set_window_visible)
        game.set_sound_enabled(set_sound_enabled)
        if interactive:
            game.set_mode(Mode.SPECTATOR)
        else:
            game.set_mode(Mode.PLAYER)

        assert n_actions==3, "Number of actions must be 3"
        game.add_available_button(Button.TURN_LEFT)
        game.add_available_button(Button.TURN_RIGHT)
        game.add_available_button(Button.MOVE_FORWARD)

        #game.add_game_args("-host 1 -deathmatch +sv_forcerespawn 1 +sv_noautoaim 1\
                #+sv_respawnprotect 1 +sv_cheats 1")  #  +sv_spawnfarthest 
        game.add_game_args("+sv_cheats 1")  #  +sv_spawnfarthest 1

        if RGB in input_types:
            _obs_space[INPUT_TYPE_TO_SENSOR_NAME[RGB]] = spaces.Box(0, 255, 
                                                        (game.get_screen_height(),
                                                         game.get_screen_width(),
                                                         game.get_screen_channels()), 
                                                        dtype=np.uint8)

        game.set_depth_buffer_enabled(DEPTH in input_types)
        if DEPTH in input_types:
            _obs_space[INPUT_TYPE_TO_SENSOR_NAME[DEPTH]] = spaces.Box(0, 255, 
                                                        (game.get_screen_height(),
                                                         game.get_screen_width(),
                                                         1),
                                                        dtype=np.uint8)
        
        game.set_labels_buffer_enabled(LABELS in input_types)
        if LABELS in input_types:
            _obs_space[INPUT_TYPE_TO_SENSOR_NAME[LABELS]] = spaces.Box(0, 255, 
                                                        (game.get_screen_height(),
                                                         game.get_screen_width(),
                                                         1),
                                                        dtype=np.uint8)
            _obs_space[INPUT_TYPE_TO_SENSOR_NAME[OBJECT_ID_NUM]] = spaces.Box(0, 255, 
                                                    (max_onscreen_objects,),
                                                    dtype=np.uint8)
            _obs_space[INPUT_TYPE_TO_SENSOR_NAME[OBJECT_LOC_BBOX]] = spaces.Box(0, 255, 
                                                    (max_onscreen_objects, 4),
                                                    dtype=np.uint8)
        if TIME in input_types:
            _obs_space[INPUT_TYPE_TO_SENSOR_NAME[TIME]] = spaces.Box(0, episode_timeout, 
                                                         (1,), dtype=np.float32)
        
        if GOAL in input_types:
            raise NotImplementedError("GOAL observation not defined in base environment")
        
        game.set_automap_buffer_enabled(AUTOMAP in input_types)
        #set_automap_buffer_enabled)
        if AUTOMAP in input_types:
            game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)
            game.add_game_args("+viz_am_center 1")  # Fixed map
            _obs_space[INPUT_TYPE_TO_SENSOR_NAME[AUTOMAP]] = spaces.Box(0, 255, 
                                                        (game.get_screen_height(),
                                                         game.get_screen_width(),
                                                         game.get_screen_channels()), 
                                                        dtype=np.uint8)        
        self.observation_space = spaces.Dict(_obs_space)
        self.action_space = spaces.Discrete(n_actions)
        self.game = game
        self.max_actions = max_actions
        self.max_onscreen_objects = max_onscreen_objects
        self.episode_number = 0
        self.randomize_maps = randomize_maps
        self.randomize_textures = randomize_textures
        self.visualize = set_window_visible
        self.interactive = interactive
        self.interactive_delay = 0.02
        self.repeat_count = repeat_count
        self.info = {'skip.repeat_count': self.repeat_count}
        self.screen_resolution = set_screen_resolution
        # self.viewer = None

        self.game.init()
        super().__init__()
        # self._randomize_textures(randomize_textures)


    def reset(self,
              agent_location=None,
              save_replay_file_path=""):
        raise NotImplementedError("reset() not implemented for base vizdoom environment")


    def step(self, action_id):
        raise NotImplementedError("step(action) not implemented for base vizdoom environment")

    

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.game.set_seed(seed)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return np.concatenate([o for k, o in self.obs.items() if k in ['color', 'map']],
                      axis=1)
        return self.obs['color']

    def _get_obs(self):        
        self.state = self.game.get_state()
        obs = {}
        if INPUT_TYPE_TO_SENSOR_NAME[RGB] in self.observation_space.spaces:
            obs[INPUT_TYPE_TO_SENSOR_NAME[RGB]] = self._zero(RGB) if self.done else self.state.screen_buffer
        
        if INPUT_TYPE_TO_SENSOR_NAME[DEPTH] in self.observation_space.spaces:
            if self.done:
                obs[INPUT_TYPE_TO_SENSOR_NAME[DEPTH]] = self._zero(DEPTH) 
            else:
                obs[INPUT_TYPE_TO_SENSOR_NAME[DEPTH]] = self._ensure_shape(DEPTH,
                                                                            self.state.depth_buffer)
        
        if INPUT_TYPE_TO_SENSOR_NAME[LABELS] in self.observation_space.spaces:
            obs[INPUT_TYPE_TO_SENSOR_NAME[OBJECT_LOC_BBOX]] = self._zero(OBJECT_LOC_BBOX)
            obs[INPUT_TYPE_TO_SENSOR_NAME[OBJECT_ID_NUM]] = self._zero(OBJECT_ID_NUM)
            if self.done:
                obs[INPUT_TYPE_TO_SENSOR_NAME[LABELS]] = self._zero(LABELS)
                for i, l in enumerate(self.state.labels):
                    if i >= self.max_onscreen_objects:
                        break
                    obs[INPUT_TYPE_TO_SENSOR_NAME[OBJECT_LOC_BBOX]][i] = np.uint8([l.x, l.y, l.width, l.height])
                    obs[INPUT_TYPE_TO_SENSOR_NAME[OBJECT_ID_NUM]][i] = l.object_id
            else:
                obs[INPUT_TYPE_TO_SENSOR_NAME[LABELS]] = self._ensure_shape(LABELS,
                                                                            self.state.labels_buffer)

        if INPUT_TYPE_TO_SENSOR_NAME[TIME] in self.observation_space.spaces:
            obs[INPUT_TYPE_TO_SENSOR_NAME[TIME]] = self.step_count
        
        if INPUT_TYPE_TO_SENSOR_NAME[GOAL] in self.observation_space.spaces:
            dists = np.stack([goal.relative_loc(self.agent.x, 
                                       self.agent.y, 
                                       self.agent.theta)
                              for goal in self.goals], axis=1)
            obs[INPUT_TYPE_TO_SENSOR_NAME[GOAL]] = dists
        if INPUT_TYPE_TO_SENSOR_NAME[AUTOMAP] in self.observation_space.spaces:
            obs[INPUT_TYPE_TO_SENSOR_NAME[AUTOMAP]] = self._zero(AUTOMAP) if self.done else self.state.automap_buffer
        # print(obs[INPUT_TYPE_TO_SENSOR_NAME[LABELS]].shape)
#         for l in self.state.labels:
#             print("Label:", l.value,
#                   "object id:", l.object_id,
#                   "object name:", l.object_name)
#             print("Object position x:", l.object_position_x, "y:", l.object_position_y, "z:", l.object_position_z)

#             # Other available fields:
#             #print("Object rotation angle", l.object_angle, "pitch:", l.object_pitch, "roll:", l.object_roll)
#             #print("Object velocity x:", l.object_velocity_x, "y:", l.object_velocity_y, "z:", l.object_velocity_z)
#             print("Bounding box: x:", l.x, "y:", l.y, "width:", l.width, "height:", l.height)
        
        return obs

    def _randomize_textures(self, val):
        ''' Randomizes textures 
        
            Args:
                val: 0 (train/test), 1 (train only), 2 test only
        '''
        self.game.send_game_command("pukename set_value always 4 %i" % self.randomize_textures)
        commands.pause_game(self.game)  # Need to step environment for changes to take effect
    
    def _zero(self, input_type):
        sensor_name = INPUT_TYPE_TO_SENSOR_NAME[input_type]
        # if 'object' in sensor_name:
        #     print(self.observation_space.spaces[sensor_name].shape)
        return np.zeros(self.observation_space.spaces[sensor_name].shape, dtype=np.uint8)   

    def _ensure_shape(self, input_type, arr):
        return arr.reshape(self.observation_space.spaces[INPUT_TYPE_TO_SENSOR_NAME[input_type]].shape)

    def _get_agent_frame_of_reference(self):
        x = self.game.get_game_variable(GameVariable.POSITION_X)
        y = self.game.get_game_variable(GameVariable.POSITION_Y)
        # x, y = self.map_cfg.normalized_coords(x, y)
        z = self.game.get_game_variable(GameVariable.POSITION_Z)
        theta = self.game.get_game_variable(GameVariable.ANGLE)

        agent_loc = {
            'x': x, 
            'y': y,
            'z': z,
            'theta': theta
        }
        return agent_loc


class DoomAgent(object):
    
    
    def __init__(self, x, y, theta, z=0.0):
        self.loc = np.float32([x, y, z])
        self._theta = theta

    @property
    def x(self):
        return self.loc[0]
    
    @property
    def y(self):
        return self.loc[1]

    @property
    def z(self):
        return self.loc[2]
    
    @property
    def theta(self):
        return self._theta

    
class PointGoal(object):
    
    def __init__(self, x, y, z=0.0):
        self.loc = np.float32([x, y, z])

    @property
    def x(self):
        return self.loc[0]
    
    @property
    def y(self):
        return self.loc[1]

    @property
    def z(self):
        return self.loc[2]
    
    def relative_loc(self, x, y, o):
        # x, y, o = a.x, a.y, a.theta
        gx = self.x
        gy = self.y
        dist = ((x-gx)**2 + (y-gy)**2)**0.5

        o = math.radians(o)
        ro = math.atan2(gy-y,gx-x)

        yr = dist*math.cos(o-ro)
        xr = dist*math.sin(o-ro)

        # yri = int(round(yr + 9))
        # xri = int(round(xr + 9))
        # print("Agent: ({}, {}, {}) | Goal: ({}, {}) | Dist: ({}, {})".format(
        #         x, y, o, gx, gy, xr, yr))
        # print(xr, yr)
        # print(np.asarray([0,0,0,0,xr,yr]))
        return np.asarray([0,0,0,0,xr,yr])




    
