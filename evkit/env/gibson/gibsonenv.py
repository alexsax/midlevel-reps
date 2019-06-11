import gym
from gym import spaces
import numpy as np
from scipy.misc import imresize

from gibson.envs.husky_env import HuskyNavigateSpeedControlEnv, HuskyNavigateEnv
from gibson.envs.random_env import HuskyRandomEnv
from gibson.envs.exploration_env import HuskyExplorationEnv, HuskyVisualExplorationEnv
from gibson.envs.visual_navigation_env import HuskyVisualNavigateEnv, HuskyVisualObstacleAvoidanceEnv, HuskyCoordinateNavigateEnv
import os

class GibsonEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, gibson_config=None, env_id='Gibson_HuskyNavigateEnv', blind=False, blank_sensor=False, start_locations_file=None, target_dim=16):
        self.target_dim = target_dim
        if env_id == "Gibson_HuskyRandomEnv":
            self.env_tag = 'random'
            if gibson_config is None:
                config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'husky_navigate_multienv.yaml')
            else:
                config = gibson_config
            self.env = HuskyRandomEnv(gpu_count=1, config=config)
        elif env_id == 'Gibson_HuskyExplorationEnv':
            self.env_tag = 'explore'
            if gibson_config is None:
                config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'husky_explore.yaml')
            else:
                config = gibson_config
            self.env = HuskyExplorationEnv(gpu_count=1, config=config, start_locations_file=None)
        elif env_id == 'Gibson_HuskyVisualExplorationEnv':
            self.env_tag = 'visualexplore'
            if gibson_config is None:
                config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'husky_visual_explore_train.yaml')
            else:
                config = gibson_config
            self.env = HuskyVisualExplorationEnv(gpu_count=1, config=config, start_locations_file=start_locations_file)
        elif env_id == 'Gibson_HuskyVisualNavigateEnv':
            self.env_tag = 'visualnavigate'
            if gibson_config is None:
                config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'husky_visual_navigate.yaml')
            else:
                config = gibson_config
            self.env = HuskyVisualNavigateEnv(gpu_count=1, config=config, valid_locations=start_locations_file)
        elif env_id == 'Gibson_HuskyVisualObstacleAvoidanceEnv':
            self.env_tag = 'visualobstacleavoidance'
            if gibson_config is None:
                config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'husky_visual_obstacle_avoidance.yaml')
            else:
                config = gibson_config
            self.env = HuskyVisualObstacleAvoidanceEnv(gpu_count=1, config=config)
        elif env_id == 'Gibson_HuskyCoordinateNavigateEnv':
            self.env_tag = 'navigate'
            if gibson_config is None:
                config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'husky_coordinate_navigate.yaml')
            else:
                config = gibson_config
            self.env = HuskyCoordinateNavigateEnv(gpu_count=1, config=config, start_locations=start_locations_file)
        self.image_dim = self.env.observation_space.shape[0]
        sensor_dim = 2
        if self.env_tag in ['random','visualnavigate', 'visualobstacleavoidance']:
            self.observation_space = spaces.Dict({
                "taskonomy": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3)),
                "rgb_filled": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim, 3)),
                })
        if self.env_tag in ['explore', 'visualexplore']:
            self.observation_space = spaces.Dict({
                "taskonomy": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3)),
                "rgb_filled": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim, 3)),
                "map": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim)),
                })
        if self.env_tag in ['navigate']:
            self.observation_space = spaces.Dict({
                "taskonomy": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3)),
                "rgb_filled": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim, 3)),
                "target": spaces.Box(low=0., high=1., shape=(3, self.target_dim, self.target_dim)),
                })
        self.action_space = self.env.action_space
        self.is_embodied = True
        self.blind = blind
        self.blank_sensor = blank_sensor
    
    def step_physics(self, action):
        return self.env.step_physics(action)

    def step(self, action):
        self.obs, rew, done, meta = self.env._step(action)
        self.obs["nonviz_sensor"] = self.obs["nonviz_sensor"][1:3]
        self.obs["taskonomy"] = self.obs["rgb_filled"]
        if self.blind:
            self.obs["rgb_filled"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
            self.obs["taskonomy"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
        if self.blank_sensor:
            self.obs["nonviz_sensor"] *= 0.0
        if "target" in self.obs.keys():
            self.obs["target"] = np.moveaxis(np.tile(self.obs["target"], (self.target_dim,self.target_dim,1)), -1, 0)
        return self.obs, rew, done, meta

    def reset(self):
        self.obs = self.env._reset()
        self.obs["nonviz_sensor"] = self.obs["nonviz_sensor"][1:3]
        self.obs["taskonomy"] = self.obs["rgb_filled"]
        if self.blind:
            self.obs["rgb_filled"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
            self.obs["taskonomy"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
        if self.blank_sensor:
            self.obs["nonviz_sensor"] *= 0.0
        if "target" in self.obs.keys():
            self.obs["target"] = np.moveaxis(np.tile(self.obs["target"], (self.target_dim,self.target_dim,1)), -1, 0)
        return self.obs
    
    def render_image_and_map(self):
        x = imresize(self.obs["rgb_filled"], (512, 512, 3))
        x_map = np.repeat(self.obs["map"][:,:,np.newaxis], 3, axis=2)
        x_map = imresize(x_map, (128, 128, 3))
        x[512-128:512, 512-128:512, :] = x_map
        return x

    def render_image_and_nav_map(self):
        x = imresize(self.obs["rgb_filled"], (256, 256, 3))
        x_map = imresize(self.obs["map"], (100, 100, 3))
        x[256-100:256, 256-100:256, :] = x_map
        return x

    def render(self, mode='human'):
        if mode == 'rgb_array' and self.env_tag == 'explore':
            return self.render_image_and_map()
        elif mode == 'rgb_array' and self.env_tag == 'navigate':
            return self.render_image_and_nav_map()
        elif mode == 'rgb_array':
            return self.obs["rgb_filled"]
        else:
            super(GibsonEnv, self).render(mode=mode)


# TODO Get the dummy working so we can iterate code faster
class DummyGibsonEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, gibson_config=None, env_id='Gibson_HuskyNavigateEnv', blind=False, blank_sensor=False,
                 start_locations_file=None, target_dim=16):
        self.target_dim = target_dim
        self.action_space = spaces.Discrete(3)
        self.env_tag = 'dummy'
        self.image_dim = 256
        sensor_dim = 2
        if self.env_tag in ['random', 'visualnavigate', 'visualobstacleavoidance']:
            self.observation_space = spaces.Dict({
                "taskonomy": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3)),
                "rgb_filled": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim, 3)),
            })
        if self.env_tag in ['explore', 'visualexplore']:
            self.observation_space = spaces.Dict({
                "taskonomy": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3)),
                "rgb_filled": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim, 3)),
                "map": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim)),
            })
        if self.env_tag in ['navigate', 'dummy']:
            self.observation_space = spaces.Dict({
                "taskonomy": spaces.Box(low=0, high=255, shape=(self.image_dim, self.image_dim, 3)),
                "rgb_filled": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim, 3)),
                "target": spaces.Box(low=0., high=1., shape=(3, self.target_dim, self.target_dim)),
                "map": spaces.Box(low=0., high=255., shape=(self.image_dim, self.image_dim)),
            })
        #         self.action_space = self.env.action_space
        self.is_embodied = True
        self.blind = blind
        self.blank_sensor = blank_sensor

    def step_physics(self, action):
        return None
        return self.env.step_physics(action)

    def step(self, action):
        self.obs = {k: np.zeros(v.shape, dtype=np.uint8) for k, v in self.observation_space.spaces.items()}
        rew = 0.0
        done = False
        meta = {}
        #         self.obs["nonviz_sensor"] = self.obs["nonviz_sensor"][1:3]
        if self.blind:
            self.obs["rgb_filled"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
            self.obs["taskonomy"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
        #         if self.blank_sensor:
        #             self.obs["nonviz_sensor"] *= 0.0
        if "target" in self.obs.keys():
            self.obs["target"] = np.moveaxis(np.tile(self.obs["target"], (self.target_dim, self.target_dim, 1)), -1, 0)
        return self.obs, rew, done, meta

    def reset(self):
        self.obs = {k: np.zeros(v.shape, dtype=np.uint8) for k, v in self.observation_space.spaces.items()}
        rew = 0.0
        done = False
        meta = {}

        #         self.obs["nonviz_sensor"] = self.obs["nonviz_sensor"][1:3]
        self.obs["taskonomy"] = self.obs["rgb_filled"]
        if self.blind:
            self.obs["rgb_filled"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
            self.obs["taskonomy"] = 127 * np.ones((self.image_dim, self.image_dim, 3), dtype=np.uint8)
        #         if self.blank_sensor:
        #             self.obs["nonviz_sensor"] *= 0.0
        if "target" in self.obs.keys():
            self.obs["target"] = np.moveaxis(np.tile(self.obs["target"], (self.target_dim, self.target_dim, 1)), -1, 0)
        return self.obs

    def render_image_and_map(self):
        x = imresize(self.obs["rgb_filled"], (512, 512, 3))
        x_map = np.repeat(self.obs["map"][:, :, np.newaxis], 3, axis=2)
        x_map = imresize(x_map, (128, 128, 3))
        x[512 - 128:512, 512 - 128:512, :] = x_map
        return x

    def render_image_and_nav_map(self):
        x = imresize(self.obs["rgb_filled"], (256, 256, 3))
        x_map = imresize(self.obs["map"], (100, 100, 3))
        x[256 - 100:256, 256 - 100:256, :] = x_map
        return x

    def render(self, mode='human'):
        if mode == 'rgb_array' and self.env_tag == 'explore':
            return self.render_image_and_map()
        elif mode == 'rgb_array' and self.env_tag == 'navigate':
            return self.render_image_and_nav_map()
        elif mode == 'rgb_array' or mode == 'human':
            return self.obs["rgb_filled"]
        else:
            super(GibsonEnv, self).render(mode=mode)
