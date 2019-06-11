# This code is a heavily modified version of a similar file in Habitat
import copy
from collections import deque
import cv2
from gym import spaces
import gzip
import numpy as np
import os
from PIL import Image
import random
from time import time
import torch
import torch.nn as nn
import warnings

from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.utils.visualizations import maps
from habitat.sims.habitat_simulator import SimulatorActions
import habitat

from .config.default import cfg as cfg_baseline
from evkit.env.wrappers import ProcessObservationWrapper, VisdomMonitor
from evkit.env.util.occupancy_map import OccupancyMap



class HabitatPreprocessVectorEnv(habitat.VectorEnv):
    def __init__(
        self,
        make_env_fn,
        env_fn_args,
        preprocessing_fn=None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        collate_obs_before_transform: bool = False,
    ):
        super().__init__(make_env_fn, env_fn_args, auto_reset_done, multiprocessing_start_method)
        obs_space = self.observation_spaces[0]
        
        # Preprocessing
        self.transform = None
        if preprocessing_fn is not None:
#             preprocessing_fn = eval(preprocessing_fn)
            self.transform, obs_space = preprocessing_fn(obs_space)

        for i in range(self.num_envs):
            self.observation_spaces[i] = obs_space

        self.collate_obs_before_transform = collate_obs_before_transform
        self.keys = []
        shapes, dtypes = {}, {}

        for key, box in obs_space.spaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def reset(self):
        observation_list = super().reset()
        if self.collate_obs_before_transform:
            self._save_init_obs(observation_list)
            if self.transform is not None:
                obs = self.transform(self.buf_init_obs)
            self._save_all_obs(obs)
        else:
            for e, obs in enumerate(observation_list):
                if self.transform is not None:
                    obs = self.transform(obs)
                self._save_obs(e, obs)
        return self._obs_from_buf()

    def step(self, action):
        results_list = super().step(action)
        for e, result in enumerate(results_list):
            self.buf_rews[e] = result[1]
            self.buf_dones[e] = result[2]
            self.buf_infos[e] = result[3]
        
        if self.collate_obs_before_transform:
            self._save_init_obs([r[0] for r in results_list])
            if self.transform is not None:
                obs = self.transform(self.buf_init_obs)
            self._save_all_obs(obs)
        else:
            for e, (obs, _, _, _) in enumerate(results_list):
                if self.transform is not None:
                    obs = self.transform(obs)
                self._save_obs(e, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy())

    def _save_init_obs(self, all_obs):
        self.buf_init_obs = {}
        for k in all_obs[0].keys():
            if k is None:
                self.buf_init_obs[k] = torch.stack([torch.Tensor(o) for o in all_obs])
            else:
                self.buf_init_obs[k] = torch.stack([torch.Tensor(o[k]) for o in all_obs])

    def _save_obs(self, e, obs):
        try:
            for k in self.keys:
                if k is None:
                    self.buf_obs[k][e] = obs
                else:
                    self.buf_obs[k][e] = obs[k]
        except Exception as e:
            print(k, e)
            raise e

    def _save_all_obs(self, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k] = obs
            else:
                self.buf_obs[k] = obs[k]

    def _obs_from_buf(self):
        if self.keys==[None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs


class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")  # for top down view
        config_env.TASK.SENSORS.append("HEADING_SENSOR") # for top down view
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        if self._distance_target() < self._config_env.SUCCESS_DISTANCE:
            action = SimulatorActions.STOP.value

        self._previous_action = action
        return super().step(action)

    def get_reward_range(self):
        return (
            self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
            self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._config_baseline.BASELINE.RL.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            self._previous_action == SimulatorActions.STOP.value
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()

        if self.get_done(observations):
            info["spl"] = self.habitat_env.get_metrics()["spl"]

        return info

def draw_top_down_map(info, heading, output_size):
    if info is None:
        return
    top_down_map = maps.colorize_topdown_map(info["top_down_map"]["map"])
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map

def transform_target(target):
    r = target[0]
    theta = target[1]
    return np.array([np.cos(theta), np.sin(theta), r])

def transform_observations(observations, target_dim=16, omap:OccupancyMap=None):
    new_obs = observations
    new_obs["rgb_filled"] = observations["rgb"]
    new_obs["taskonomy"] = observations["rgb"]
    new_obs["target"] = np.moveaxis(np.tile(transform_target(observations["pointgoal"]), (target_dim,target_dim,1)), -1, 0)
    if omap is not None:
        new_obs['map'] = omap.construct_occupancy_map()
        new_obs['global_pos'] = omap.get_current_global_pos()
    del new_obs['rgb']
    return new_obs

def get_obs_space(image_dim=256, target_dim=16, map_dim=None):
    if map_dim is not None:
        return spaces.Dict({
            "taskonomy": spaces.Box(low=0, high=255, shape=(image_dim, image_dim, 3), dtype=np.uint8),
            "rgb_filled": spaces.Box(low=0., high=255., shape=(image_dim, image_dim, 3), dtype=np.uint8),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3, target_dim, target_dim), dtype=np.float32),
            "map": spaces.Box(low=0., high=255., shape=(map_dim, map_dim, 3), dtype=np.uint8),
            "global_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,) , dtype=np.float32),
        })
    else:
        return spaces.Dict({
            "taskonomy": spaces.Box(low=0, high=255, shape=(image_dim, image_dim, 3), dtype=np.uint8),
            "rgb_filled": spaces.Box(low=0., high=255., shape=(image_dim, image_dim, 3), dtype=np.uint8),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(3, target_dim, target_dim), dtype=np.float32),
        })

class MidlevelNavRLEnv(NavRLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config_env, config_baseline, dataset, target_dim=7, map_kwargs={}, reward_kwargs={}):
        super().__init__(config_env, config_baseline, dataset)
        self.target_dim = target_dim
        self.image_dim = 256

        self.use_map = map_kwargs['map_building_size'] > 0
        self.map_dim = 84 if self.use_map else None
        self.map_kwargs = map_kwargs
        self.reward_kwargs = reward_kwargs
        self.last_map = None  # TODO unused

        self.observation_space = get_obs_space(self.image_dim, self.target_dim, self.map_dim)

        self.omap = None
        if self.use_map:
            self.omap = OccupancyMap(map_kwargs=map_kwargs)  # this one is not used


    def get_reward(self, observations):
        reward = self.reward_kwargs['slack_reward']

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self.reward_kwargs['use_visit_penalty'] and len(self.omap.history) > 5:
            reward += self.reward_kwargs['visit_penalty_coef'] * self.omap.compute_eps_ball_ratio(self.reward_kwargs['penalty_eps'])

        if self._episode_success():
            reward += self.reward_kwargs['success_reward']

        return reward

    def reset(self):
        self.info = None
        self.obs = super().reset()
        if self.use_map:
            self.omap = OccupancyMap(initial_pg=self.obs['pointgoal'], map_kwargs=self.map_kwargs)
        self.obs = transform_observations(self.obs, target_dim=self.target_dim, omap=self.omap)
        if 'map' in self.obs:
            self.last_map = self.obs['map']
        return self.obs

    def step(self, action):
        self.obs, reward, done, self.info = super().step(action)
        if self.use_map:
            self.omap.add_pointgoal(self.obs['pointgoal'])
            self.omap.step(action)  # our forward model needs to see how the env changed due to the action (via the pg)
        self.obs = transform_observations(self.obs, target_dim=self.target_dim, omap=self.omap)
        if 'map' in self.obs:
            self.last_map = self.obs['map']
        return self.obs, reward, done, self.info

    def render(self, mode='human'):
        if mode == 'rgb_array':
            im = self.obs["rgb_filled"]

            # Get the birds eye view of the agent
            if self.info is None:
                top_down_map = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                top_down_map = draw_top_down_map(
                    self.info, self.obs["heading"], im.shape[0]
                )
                top_down_map = np.array(Image.fromarray(top_down_map).resize((256,256)))

            if 'map' in self.obs:
                occupancy_map = self.obs['map']
                h,w,_ = occupancy_map.shape
                occupancy_map[int(h//2), int(w//2), 2] = 255   # for debugging
                occupancy_map = np.array(Image.fromarray(occupancy_map).resize((256,256)))
                output_im = np.concatenate((im, top_down_map, occupancy_map), axis=1)
            else:
                output_im = np.concatenate((im, top_down_map), axis=1)

            # Pad to make dimensions even ( will always be even )
            # npad = ((output_im.shape[0] % 2, 0), (output_im.shape[1] %2, 0), (0, 0))
            # output_im = np.pad(output_im, pad_width=npad, mode='constant', constant_values=0)
            return output_im
        else:
            super().render(mode=mode)

def make_habitat_vector_env(num_processes=2,
                            target_dim=7,
                            preprocessing_fn=None,
                            log_dir=None,
                            visdom_name='main',
                            visdom_log_file=None,
                            visdom_server='localhost',
                            visdom_port='8097',
                            vis_interval=200,
                            scenes=None,
                            val_scenes=['Greigsville', 'Pablo', 'Mosquito'],
                            num_val_processes=0,
                            swap_building_k_episodes=10,
                            gpu_devices=[0],
                            collate_obs_before_transform=False,
                            map_kwargs={},
                            reward_kwargs={},
                            seed=42
                           ):
    assert map_kwargs['map_building_size'] > 0, 'Map building size must be positive!'
    default_reward_kwargs = {
                'slack_reward': -0.01,
                'success_reward': 10,
                'use_visit_penalty': False,
                'visit_penalty_coef': 0,
                'penalty_eps': 999,
            }
    for k, v in default_reward_kwargs.items():
        if k not in reward_kwargs:
            reward_kwargs[k] = v
    
    habitat_path = os.path.dirname(os.path.dirname(habitat.__file__))
    task_config = os.path.join(habitat_path, 'configs/tasks/pointnav_gibson_train.yaml')
    basic_config = cfg_env(config_file=task_config)
    basic_config.defrost()
    basic_config.DATASET.POINTNAVV1.DATA_PATH = os.path.join(habitat_path, basic_config.DATASET.POINTNAVV1.DATA_PATH)
    basic_config.freeze()

    if scenes is None:
        scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
        random.shuffle(scenes)

    val_task_config = os.path.join(habitat_path, 'configs/tasks/pointnav_gibson_val_mini.yaml')
    val_cfg = cfg_env(config_file=val_task_config)
    val_cfg.defrost()
    val_cfg.DATASET.SPLIT = "val"
    val_cfg.freeze()

    scenes = [s for s in scenes if s not in val_scenes]
    if num_val_processes > 0 and  len(val_scenes) % num_val_processes != 0:
        warnings.warn("Please make num_val_processes ({}) evenly divide len(val_scenes) ({}) or some buildings may be overrepresented".format(num_val_processes, len(val_scenes)))

    env_configs = []
    baseline_configs = []
    encoders = []
    target_dims = []
    is_val = []
    
    # Assign specific buildings to each process
    train_process_scenes = [[] for _ in range(num_processes - num_val_processes)]
    for i, scene in enumerate(scenes):
        train_process_scenes[i % len(train_process_scenes)].append(scene)

    if num_val_processes > 0:
        val_process_scenes = [[] for _ in range(num_val_processes)]
        for i, scene in enumerate(val_scenes):
            val_process_scenes[i % len(val_process_scenes)].append(scene)

    for i in range(num_processes):
        config_env = cfg_env(task_config)
        config_env.defrost()
        config_env.DATASET.POINTNAVV1.DATA_PATH = os.path.join(habitat_path, basic_config.DATASET.POINTNAVV1.DATA_PATH)


        if i < num_processes - num_val_processes:
            config_env.DATASET.SPLIT = 'train'
            config_env.DATASET.POINTNAVV1.CONTENT_SCENES = train_process_scenes[i]
        else:
            config_env.DATASET.SPLIT = 'val'
            val_i = i - (num_processes - num_val_processes)
            config_env.DATASET.POINTNAVV1.CONTENT_SCENES = val_process_scenes[val_i]
        print("Env {}:".format(i), config_env.DATASET.POINTNAVV1.CONTENT_SCENES)

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_devices[i % len(gpu_devices)]
        agent_sensors = ["RGB_SENSOR"]
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.SIMULATOR.SCENE = os.path.join(habitat_path, config_env.SIMULATOR.SCENE)
        
        config_env.freeze()
        env_configs.append(config_env)
        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)
        encoders.append(preprocessing_fn)
        target_dims.append(target_dim)

    should_record = [(i == 0 or i == (num_processes - num_val_processes)) for i in range(num_processes)]
    envs = HabitatPreprocessVectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(env_configs,
                    baseline_configs,
                    range(num_processes),
                    target_dims,
                    [log_dir for _ in range(num_processes)],
                    [visdom_name for _ in range(num_processes)],
                    [visdom_log_file for _ in range(num_processes)],
                    [vis_interval for _ in range(num_processes)],
                    [visdom_server for _ in range(num_processes)],
                    [visdom_port for _ in range(num_processes)],
                    [swap_building_k_episodes for _ in range(num_processes)],
                    [map_kwargs for _ in range(num_processes)],
                    [reward_kwargs for _ in range(num_processes)],
                    should_record,
                    [seed + i for i in range(num_processes)],
                   )
            )
        ),
        preprocessing_fn=preprocessing_fn,
        collate_obs_before_transform=collate_obs_before_transform
    )
    envs.observation_space = envs.observation_spaces[0]
    envs.action_space = spaces.Discrete(3)
    envs.reward_range = None
    envs.metadata = None
    envs.is_embodied = True
    return envs


flatten = lambda l: [item for sublist in l for item in sublist]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def shuffle_episodes(env, swap_every_k=10):
    episodes = env.episodes
#     buildings_for_epidodes = [e.scene_id for e in episodes]
    episodes = env.episodes = random.sample([c for c in chunks(episodes, swap_every_k)], len(episodes) // swap_every_k)
    env.episodes = flatten(episodes)
    return env.episodes

def make_env_fn(config_env,
                config_baseline,
                rank,
                target_dim,
                log_dir,
                visdom_name,
                visdom_log_file,
                vis_interval,
                visdom_server,
                visdom_port,
                swap_building_k_episodes,
                map_kwargs,
                reward_kwargs,
                should_record,
                seed):    
    if config_env.DATASET.SPLIT == 'val':
        datasetfile_path = config_env.DATASET.POINTNAVV1.DATA_PATH.format(split=config_env.DATASET.SPLIT)
        dataset = PointNavDatasetV1()
        with gzip.open(datasetfile_path, "rt") as f:
            dataset.from_json(f.read())
    else:
        dataset = PointNavDatasetV1(config_env.DATASET)

    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    env = MidlevelNavRLEnv(config_env=config_env,
                           config_baseline=config_baseline,
                           dataset=dataset,
                           target_dim=target_dim,
                           map_kwargs=map_kwargs,
                           reward_kwargs=reward_kwargs)
    env.episodes = shuffle_episodes(env, swap_every_k=swap_building_k_episodes)
    env.seed(seed)
    if should_record and visdom_log_file is not None:
        print("SETTING VISDOM MONITOR WITH VIS INTERVAL", vis_interval)
        env = VisdomMonitor(env,
                       directory=os.path.join(log_dir, visdom_name),
                       video_callable=lambda x: x % vis_interval == 0,
                       uid=str(rank),
                       server=visdom_server,
                       port=visdom_port,
                       visdom_log_file=visdom_log_file,
                       visdom_env=visdom_name)

    return env


