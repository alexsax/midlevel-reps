# used for habitat challenges
# CHALLENGE_CONFIG_FILE=/root/perception_module/habitat-api/configs/tasks/pointnav_gibson_val_mini.yaml python -m scripts.evaluate_habitat /mnt/logdir/normal_encoding_multienv_map
# not sure on ^
# python -m scripts.evaluate_habitat /mnt/logdir/normal_encoding_multienv_map 1

import copy
from gym import spaces
import habitat
from habitat.core.agent import Agent
import json
import numpy as np
import os
import pprint
import scipy

import evkit
from evkit.models.architectures import AtariNet, TaskonomyFeaturesOnlyNet
from evkit.models.actor_critic_module import NaivelyRecurrentACModule
from evkit.env.habitat.habitatenv import transform_observations, get_obs_space
from evkit.env.habitat.utils import STOP_VALUE
from evkit.env.util.occupancy_map import OccupancyMap
from evkit.preprocess.transforms import rescale_centercrop_resize, rescale, grayscale_rescale, cross_modal_transform, identity_transform, rescale_centercrop_resize_collated, map_pool_collated, map_pool, taskonomy_features_transform, image_to_input_collated, taskonomy_multi_features_transform
from evkit.preprocess.baseline_transforms import blind, pixels_as_state
from evkit.preprocess import TransformFactory
from evkit.rl.algo.ppo import PPO
from evkit.rl.algo.ppo_replay import PPOReplay
from evkit.rl.storage import StackedSensorDictStorage, RolloutSensorDictReplayBuffer
from evkit.rl.policy import Policy, PolicyWithBase, BackoutPolicy, JerkAvoidanceValidator, TrainedBackoutPolicy
from evkit.utils.misc import Bunch, cfg_to_md, compute_weight_norm, is_interactive, remove_whitespace, update_dict_deepcopy
from evkit.utils.random import set_seed
from evkit.utils.logging import get_subdir
import tnt.torchnet as tnt
from tnt.torchnet.logger import FileLogger
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import runpy
import sys

try:
    from habitat.utils.visualizations.utils import images_to_video
except ImportError:
    pass

from sacred import Experiment
ex = Experiment(name="Habitat Evaluation", interactive=False)

try:
    LOG_DIR = sys.argv[1].strip()
    runpy.run_module('configs.habitat', init_globals=globals())
    sys.argv.pop(1)
except:
    print('need to set a logdir if using weights only / loading from habitat train configs')
    pass


runpy.run_module('configs.habitat_eval', init_globals=globals())
os.environ["IMAGEIO_FFMPEG_EXE"] = '/usr/bin/ffmpeg'  # figure out better way to do this

def split_and_cat(imgs):
    imgs_list = [imgs[:, 3*i: 3*(i+1)][0] for i in range(int(imgs.shape[1] / 3))]
    imgs_output = torch.cat(imgs_list, dim=1)
    return imgs_output


class HabitatAgent(Agent):
    def __init__(self, ckpt_path, config_data):
        # Load agent
        self.action_space = spaces.Discrete(3)
        if ckpt_path is not None:
            checkpoint_obj = torch.load(ckpt_path)
            start_epoch = checkpoint_obj["epoch"]
            print("Loaded learner (epoch {}) from {}".format(start_epoch, ckpt_path), flush=True)
            agent = checkpoint_obj["agent"]
        else:
            cfg = config_data['cfg']
            perception_model = eval(cfg['learner']['perception_network'])(
                cfg['learner']['num_stack'],
                **cfg['learner']['perception_network_kwargs'])
            base = NaivelyRecurrentACModule(
                perception_unit=perception_model,
                use_gru=cfg['learner']['recurrent_policy'],
                internal_state_size=cfg['learner']['internal_state_size'])
            actor_critic = PolicyWithBase(
                base, self.action_space,
                num_stack=cfg['learner']['num_stack'],
                takeover=None)
            if cfg['learner']['use_replay']:
                agent = PPOReplay(actor_critic,
                                                cfg['learner']['clip_param'],
                                                cfg['learner']['ppo_epoch'],
                                                cfg['learner']['num_mini_batch'],
                                                cfg['learner']['value_loss_coef'],
                                                cfg['learner']['entropy_coef'],
                                                cfg['learner']['on_policy_epoch'],
                                                cfg['learner']['off_policy_epoch'],
                                                lr=cfg['learner']['lr'],
                                                eps=cfg['learner']['eps'],
                                                max_grad_norm=cfg['learner']['max_grad_norm'])
            else:
                agent = PPO(actor_critic,
                                          cfg['learner']['clip_param'],
                                          cfg['learner']['ppo_epoch'],
                                          cfg['learner']['num_mini_batch'],
                                          cfg['learner']['value_loss_coef'],
                                          cfg['learner']['entropy_coef'],
                                          lr=cfg['learner']['lr'],
                                          eps=cfg['learner']['eps'],
                                          max_grad_norm=cfg['learner']['max_grad_norm'])
            weights_path = cfg['eval_kwargs']['weights_only_path']
            ckpt = torch.load(weights_path)
            agent.actor_critic.load_state_dict(ckpt['state_dict'])
            agent.optimizer = ckpt['optimizer']
        self.actor_critic = agent.actor_critic

        self.takeover_policy = None
        if config_data['cfg']['learner']['backout']['use_backout']:
            backout_type = config_data['cfg']['learner']['backout']['backout_type']
            if backout_type == 'hardcoded':
                self.takeover_policy = BackoutPolicy(
                    patience=config_data['cfg']['learner']['backout']['patience'],
                    num_processes=1,
                    unstuck_dist=config_data['cfg']['learner']['backout']['unstuck_dist'],
                    randomize_actions=config_data['cfg']['learner']['backout']['randomize_actions'],
                )
            elif backout_type == 'trained':
                backout_ckpt =config_data['cfg']['learner']['backout']['backout_ckpt_path']
                assert backout_ckpt is not None, 'need a checkpoint to use a trained backout'
                backout_checkpoint_obj = torch.load(backout_ckpt)
                backout_start_epoch = backout_checkpoint_obj["epoch"]
                print("Loaded takeover policy at (epoch {}) from {}".format(backout_start_epoch, backout_ckpt), flush=True)
                backout_policy = checkpoint_obj["agent"].actor_critic

                self.takeover_policy = TrainedBackoutPolicy(
                    patience=config_data['cfg']['learner']['backout']['patience'],
                    num_processes=1,
                    policy=backout_policy,
                    unstuck_dist=config_data['cfg']['learner']['backout']['unstuck_dist'],
                    num_takeover_steps=config_data['cfg']['learner']['backout']['num_takeover_steps'],
                )
            else:
                assert False, f'do not recognize backout type {backout_type}'
        self.actor_critic.takeover = self.takeover_policy

        self.validator = None
        if config_data['cfg']['learner']['validator']['use_validator']:
            validator_type = config_data['cfg']['learner']['validator']['validator_type']
            if validator_type == 'jerk':
                self.validator = JerkAvoidanceValidator()
            else:
                assert False, f'do not recognize validator {validator_type}'
        self.actor_critic.action_validator = self.validator

        # Set up spaces
        self.target_dim = config_data['cfg']['env']['env_specific_kwargs']['target_dim']

        map_dim = None
        self.omap = None
        if config_data['cfg']['env']['use_map']:
            self.map_kwargs = config_data['cfg']['env']['habitat_map_kwargs']
            map_dim = 84
            assert self.map_kwargs['map_building_size'] > 0, 'If we are using map in habitat, please set building size to be positive!'

        obs_space = get_obs_space(image_dim=256, target_dim=self.target_dim, map_dim=map_dim)

        preprocessing_fn_pre_agg = eval(config_data['cfg']['env']['transform_fn_pre_aggregation'])
        self.transform_pre_agg, obs_space = preprocessing_fn_pre_agg(obs_space)

        preprocessing_fn_post_agg = eval(config_data['cfg']['env']['transform_fn_post_aggregation'])
        self.transform_post_agg, obs_space = preprocessing_fn_post_agg(obs_space)

        self.current_obs = StackedSensorDictStorage(1,
                                               config_data['cfg']['learner']['num_stack'],
                                               {k: v.shape for k, v in obs_space.spaces.items()
                                                if k in config_data['cfg']['env']['sensors']})
        print(f'Stacked obs shape {self.current_obs.obs_shape}')

        self.current_obs = self.current_obs.cuda()
        self.actor_critic.cuda()

        self.hidden_size = config_data['cfg']['learner']['internal_state_size']
        self.test_recurrent_hidden_states = None
        self.not_done_masks = None

        self.episode_rgbs = []
        self.episode_pgs = []
        self.episode_entropy = []
        self.episode_num = 0
        self.t = 0
        self.episode_lengths = []
        self.episode_values = []
        self.last_action = None

        # Set up logging
        if config_data['cfg']['saving']['logging_type'] == 'visdom':
            self.mlog = tnt.logger.VisdomMeterLogger(
                title=config_data['uuid'], env=config_data['uuid'], server=config_data['cfg']['saving']['visdom_server'],
                port=config_data['cfg']['saving']['visdom_port'],
                log_to_filename=config_data['cfg']['saving']['visdom_log_file']
            )
            self.use_visdom = True
        elif config_data['cfg']['saving']['logging_type'] == 'tensorboard':
            self.mlog = tnt.logger.TensorboardMeterLogger(
                env=config_data['uuid'],
                log_dir=config_data['cfg']['saving']['log_dir'],
                plotstylecombined=True
            )
            self.use_visdom = False
        else:
            assert False, 'no proper logger!'

        self.log_dir = config_data['cfg']['saving']['log_dir']
        self.save_eval_videos = config_data['cfg']['saving']['save_eval_videos']
        self.mlog.add_meter('config', tnt.meter.SingletonMeter(), ptype='text')
        self.mlog.update_meter(cfg_to_md(config_data['cfg'], config_data['uuid']), meters={'config'}, phase='val')

    def reset(self):
        # reset hidden state and set done
        self.test_recurrent_hidden_states = torch.zeros(
            1, self.hidden_size
        ).cuda()
        self.not_done_masks = torch.zeros(1, 1).cuda()

        # reset observation storage (and verify)
        z = torch.zeros(1, 2).cuda()
        mask_out_done = { name: z for name in self.current_obs.sensor_names }
        if 'global_pos' in self.current_obs.sensor_names:
            mask_out_done['global_pos'] = torch.zeros(1,1).cuda()
        self.current_obs.clear_done(mask_out_done)
        for value in self.current_obs.peek().values():
            assert torch.sum(value.peek()).item() < 1e-6, 'did not clear the curent_obs properly'

        # log everything
        if len(self.episode_pgs) != 0:
            # log video (and save to log_dir)
            if self.save_eval_videos:
                images_to_video(images=self.episode_rgbs, output_dir=self.log_dir, video_name=f'test_{self.episode_num}')
                self.mlog.add_meter(f'diagnostics/rollout_{self.episode_num}', tnt.meter.SingletonMeter(), ptype='video')
                if self.use_visdom:
                    vid_path = os.path.join(self.log_dir, f'test_{self.episode_num}.mp4')
                    self.mlog.update_meter(vid_path, meters={f'diagnostics/rollout_{self.episode_num}'}, phase='val')
                else:
                    print('video support for TB is weak not recommended')
                    rgb_tensor = torch.Tensor(self.episode_rgbs).unsqueeze(dim=0)
                    self.mlog.update_meter(rgb_tensor, meters={f'diagnostics/rollout_{self.episode_num}'}, phase='val')

            # reset log
            self.mlog.reset_meter(self.episode_num, mode='val')

            # reset episode logs
            self.episode_rgbs = []
            self.episode_pgs = []
            self.episode_values = []
            self.episode_entropy = []
            self.episode_lengths.append(self.t)
            self.episode_num += 1
            self.t = 0
            self.last_action = None

    def act(self, observations):
        # tick
        self.t += 1

        # collect raw observations
        self.episode_rgbs.append(copy.deepcopy(observations['rgb']))
        self.episode_pgs.append(copy.deepcopy(observations['pointgoal']))

        # initialize or step occupancy map
        if self.map_kwargs['map_building_size'] > 0:
            if self.t == 1:
                self.omap = OccupancyMap(initial_pg=observations['pointgoal'], map_kwargs=self.map_kwargs)
            else:
                assert self.last_action is not None, 'This is not the first timestep, there must have been at least one action'
                self.omap.add_pointgoal(observations['pointgoal'])
                self.omap.step(self.last_action)

        # hard-coded STOP
        dist = observations['pointgoal'][0]
        if dist <= 0.2:
            return STOP_VALUE

        # preprocess and get observation
        observations = transform_observations(observations, target_dim=self.target_dim, omap=self.omap)
        observations = self.transform_pre_agg(observations)
        for k, v in observations.items():
            observations[k] = np.expand_dims(v, axis=0)
        observations = self.transform_post_agg(observations)
        self.current_obs.insert(observations)
        self.obs_stacked = {k: v.peek().cuda() for k, v in self.current_obs.peek().items()}

        # log first couple agent observation
        if self.t % 4 == 0 and 50 < self.t < 60:
            map_output = split_and_cat(self.obs_stacked['map']) * 0.5 + 0.5
            self.mlog.add_meter(f'diagnostics/map_{self.t}', tnt.meter.SingletonMeter(), ptype='image')
            self.mlog.update_meter(map_output, meters={f'diagnostics/map_{self.t}'}, phase='val')

        # act
        with torch.no_grad():
            value, action, act_log_prob, self.test_recurrent_hidden_states = self.actor_critic.act(
                self.obs_stacked,
                self.test_recurrent_hidden_states,
                self.not_done_masks,
            )
            action = action.item()
            self.not_done_masks = torch.ones(1, 1).cuda()  # mask says not done

        # log agent outputs
        assert self.action_space.contains(action), 'action from model does not fit our action space'
        self.last_action = action
        return action

    def finish_benchmark(self, metrics):
        self.mlog.add_meter('diagnostics/length_hist', tnt.meter.ValueSummaryMeter(), ptype='histogram')
        self.mlog.update_meter(self.episode_lengths, meters={'diagnostics/length_hist'}, phase='val')

        for k, v in metrics.items():
            print(k, v)
            self.mlog.add_meter(f'metrics/{k}',  tnt.meter.ValueSummaryMeter())
            self.mlog.update_meter(v, meters={f'metrics/{k}'}, phase='val')

        self.mlog.reset_meter(self.episode_num + 1, mode='val')

@ex.main
def run_cfg(cfg, uuid):
    if cfg['eval_kwargs']['exp_path'] is not None:
        # Process exp path
        exp_paths = [cfg['eval_kwargs']['exp_path']]

        # Set up config with the first exp only
        metadata_dir = get_subdir(exp_paths[0], 'metadata')
        config_path = os.path.join(metadata_dir, 'config.json')

        # Load config
        with open(config_path) as config:
            config_data = json.load(config)

        # Update configs
        config_data['uuid'] += '_benchmark' + uuid
        config_data['cfg']['saving']['log_dir'] += '/benchmark'
        config_data['cfg']['saving']['visdom_log_file'] = os.path.join(config_data['cfg']['saving']['log_dir'], 'visdom_logs.json')
        config_data['cfg']['learner']['test'] = True

        if cfg['eval_kwargs']['overwrite_configs']:
            config_data['cfg'] = update_dict_deepcopy(config_data['cfg'], cfg)

        set_seed(config_data['cfg']['training']['seed'])

        # Get checkpoints
        ckpt_paths = []
        for exp_path in exp_paths:
            ckpts_dir = get_subdir(exp_path, 'checkpoints')
            ckpt_path = os.path.join(ckpts_dir, 'ckpt-latest.dat')
            ckpt_paths.append(ckpt_path)
    else:
        config_data = { 'cfg': cfg, 'uuid': uuid }
        ckpt_paths = [None]
        exp_paths = [LOG_DIR]

    if 'eval_kwargs' in cfg and 'debug' in cfg['eval_kwargs']:
        if cfg['eval_kwargs']['debug']:
            config_data['cfg']['saving']['logging_type'] = 'visdom'
            config_data['cfg']['saving']['save_eval_videos'] = True
        else:
            config_data['cfg']['saving']['save_eval_videos'] = False

    print(pprint.pformat(config_data))
    print('Loaded:', config_data['uuid'])
    agent = HabitatAgent(ckpt_path=ckpt_paths[0], config_data=config_data)

    if cfg['eval_kwargs']['challenge']:
        challenge = habitat.Challenge()
        challenge.submit(agent)
    else:
        benchmark = habitat.Benchmark(config_file=cfg['eval_kwargs']['benchmark_config'], config_dir='/')
        metrics = benchmark.evaluate(agent, cfg['eval_kwargs']['benchmark_episodes'])
        agent.finish_benchmark(metrics)
        benchmark._env.close()

        everything = update_dict_deepcopy(metrics, config_data)
        patience, unstuck_dist = config_data['cfg']['learner']['backout']['patience'], config_data['cfg']['learner']['backout']['unstuck_dist']
        write_location = os.path.join(exp_paths[0], f'benchmark_data_p{patience}_d{unstuck_dist}.json')
        with open(write_location, 'w') as outfile:
            json.dump(everything, outfile)


if __name__ == "__main__":
    ex.run_commandline()
