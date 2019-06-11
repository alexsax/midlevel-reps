# train_rl.py
# Authors: Sasha Sax (1,3), Bradley Emi (2), Jeffrey Zhang (1) -- UC Berkeley, FAIR, Stanford VL
# Desc: Train or test an agent using PPO.
# Usage:
#    python -m scripts.train_rl DIRECTORY_TO_SAVE_RESULTS run_training with uuid=EXP_UUID [CFG1 ...] [cfg.SUB_CFG1.PROPERTY1 ...]
# Notes: 
#     (i) must be run from parent directory (top-level of git)
#     (ii) currently, a visdom instance MUST be used or the script will fail. Defaults to localhost.

import sys
import shutil
import copy
import glob
from gym import logger
from gym import spaces
import gym
import logging
import numpy as np
import os
import pprint
import random
import runpy
import sacred
import subprocess
import time
import torch
import torchvision.utils


from evkit.env.wrappers import ProcessObservationWrapper
from evkit.env import EnvFactory
from evkit.models.architectures import AtariNet, TaskonomyFeaturesOnlyNet
from evkit.models.taskonomy_network import TaskonomyNetwork
from evkit.models.actor_critic_module import NaivelyRecurrentACModule
from evkit.preprocess.transforms import rescale_centercrop_resize, rescale, grayscale_rescale, cross_modal_transform, identity_transform, rescale_centercrop_resize_collated, map_pool_collated, map_pool, taskonomy_features_transform, image_to_input_collated, taskonomy_multi_features_transform
from evkit.preprocess.baseline_transforms import blind, pixels_as_state
from evkit.preprocess import TransformFactory
import evkit.rl.algo
from evkit.rl.policy import Policy, PolicyWithBase, BackoutPolicy
from evkit.rl.storage import RolloutSensorDictStorage, RolloutSensorDictReplayBuffer, StackedSensorDictStorage
from evkit.saving.checkpoints import checkpoint_name, save_checkpoint, last_archived_run
from evkit.saving.observers import FileStorageObserverWithExUuid
from evkit.utils.misc import Bunch, cfg_to_md, compute_weight_norm, is_interactive, remove_whitespace, update_dict_deepcopy
import evkit.utils.logging
from evkit.utils.random import set_seed
import tnt.torchnet as tnt

# Set up experiment using SACRED
ex = sacred.Experiment(name="RL Training", interactive=is_interactive())
LOG_DIR = sys.argv[1].strip()
sys.argv.pop(1)
runpy.run_module('configs.core', init_globals=globals())
runpy.run_module('configs.image_architectures', init_globals=globals())
runpy.run_module('configs.habitat', init_globals=globals())
runpy.run_module('configs.gibson', init_globals=globals())
runpy.run_module('configs.doom', init_globals=globals())

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()
                        
def log_input_images(obs_unpacked, mlog, num_stack, key_names=['map'], meter_name='debug/input_images', step_num=0):
    # Plots the observations from the first process
    stacked = []
    for key_name in key_names:
        if key_name not in obs_unpacked:
            logger.debug(key_name, "not found")
            continue
        obs = obs_unpacked[key_name][0]
        obs = (obs + 1.0) / 2.0
        obs_chunked = list(torch.chunk(obs, num_stack, dim=0))
        key_stacked = torchvision.utils.make_grid(obs_chunked, nrow=num_stack, padding=2)
        stacked.append(key_stacked)
    stacked = torch.cat(stacked, dim=1)
    mlog.update_meter(stacked, meters={meter_name})
    mlog.reset_meter(step_num, meterlist={meter_name})


@ex.main
def run_training(cfg, uuid):
    try:
        logger.info("Running with configuration:\n" + pprint.pformat(cfg))
        torch.set_num_threads(1)
        set_seed(cfg['training']['seed'])

        # get new output_dir name (use for checkpoints)
        old_log_dir = cfg['saving']['log_dir']
        changed_log_dir = False
        existing_log_paths = []
        if os.path.exists(old_log_dir) and cfg['saving']['autofix_log_dir']:
            LOG_DIR, existing_log_paths = evkit.utils.logging.unused_dir_name(old_log_dir)
            os.makedirs(LOG_DIR, exist_ok=False)
            cfg['saving']['log_dir'] = LOG_DIR
            cfg['saving']['results_log_file'] = os.path.join(LOG_DIR, 'result_log.pkl')
            cfg['saving']['reward_log_file'] = os.path.join(LOG_DIR, 'rewards.pkl')
            cfg['saving']['visdom_log_file'] = os.path.join(LOG_DIR, 'visdom_logs.json')
            changed_log_dir = True

        # Load checkpoint, config, agent
        agent = None
        if cfg['training']['resumable']:
            if cfg['saving']['checkpoint']:
                prev_run_path = cfg['saving']['checkpoint']
                ckpt_fpath = os.path.join(prev_run_path, 'checkpoints', 'ckpt-latest.dat')
                if cfg['saving']['checkpoint_configs']:  # update configs with values from ckpt
                    prev_run_metadata_paths = [ os.path.join(prev_run_path, f)
                                               for f in os.listdir(prev_run_path)
                                               if f.endswith('metadata')]
                    prev_run_config_path = os.path.join(prev_run_metadata_paths[0], 'config.json')
                    with open(prev_run_config_path) as f:
                        config = json.load(f)  # keys are ['cfg', 'uuid', 'seed']
                    cfg = update_dict_deepcopy(cfg, config['cfg'])
                    uuid = config['uuid']
                    logger.warning("Reusing config from {}".format(prev_run_config_path))
                if ckpt_fpath is not None and os.path.exists(ckpt_fpath):
                    checkpoint_obj = torch.load(ckpt_fpath)
                    start_epoch = checkpoint_obj['epoch']
                    logger.info("Loaded learner (epoch {}) from {}".format(start_epoch, ckpt_fpath))
                    agent = checkpoint_obj['agent']
                    actor_critic = agent.actor_critic
                else:
                    logger.warning("No checkpoint found at {}".format(ckpt_fpath))

        # Make environment
        simulator, scenario = cfg['env']['env_name'].split('_')
        if cfg['env']['transform_fn_pre_aggregation'] is None:
            cfg['env']['transform_fn_pre_aggregation'] = "None" 
        envs = EnvFactory.vectorized(
                        cfg['env']['env_name'],
                        cfg['training']['seed'],
                        cfg['env']['num_processes'],
                        cfg['saving']['log_dir'],
                        cfg['env']['add_timestep'],
                        env_specific_kwargs  = cfg['env']['env_specific_kwargs'],
                        num_val_processes    = cfg['env']['num_val_processes'],
                        preprocessing_fn     = eval(cfg['env']['transform_fn_pre_aggregation']),
                        addl_repeat_count    = cfg['env']['additional_repeat_count'],
                        sensors              = cfg['env']['sensors'],
                        vis_interval         = cfg['saving']['vis_interval'],
                        visdom_server        = cfg['saving']['visdom_server'],
                        visdom_port          = cfg['saving']['visdom_port'],
                        visdom_log_file      = cfg['saving']['visdom_log_file'],
                        visdom_name          = uuid)
        if 'transform_fn_post_aggregation' in cfg['env'] and cfg['env']['transform_fn_post_aggregation'] is not None:
            transform, space = eval(cfg['env']['transform_fn_post_aggregation'])(envs.observation_space)
            envs = ProcessObservationWrapper(envs, transform, space)
        is_habitat_env = (simulator == 'Habitat')
        action_space = envs.action_space
        observation_space = envs.observation_space
        retained_obs_shape = { k: v.shape
                               for k, v in observation_space.spaces.items()
                               if k in cfg['env']['sensors']}
        logger.info(f"Action space: {action_space}")
        logger.info(f"Observation space: {observation_space}")
        logger.info("Retaining: {}".format(set(observation_space.spaces.keys()).intersection(cfg['env']['sensors'].keys())))

        # Finish setting up the agent
        if agent == None:
            perception_model = eval(cfg['learner']['perception_network'])(
                              cfg['learner']['num_stack'],
                              **cfg['learner']['perception_network_kwargs'])
            base = NaivelyRecurrentACModule(
                              perception_unit=perception_model,
                              use_gru=cfg['learner']['recurrent_policy'],
                              internal_state_size=cfg['learner']['internal_state_size'])
            actor_critic = PolicyWithBase(
                              base, action_space,
                              num_stack=cfg['learner']['num_stack'],
                              takeover=None)
            if cfg['learner']['use_replay']:
                agent = evkit.rl.algo.PPOReplay(actor_critic,
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
                agent = evkit.rl.algo.PPO(actor_critic,
                                  cfg['learner']['clip_param'],
                                  cfg['learner']['ppo_epoch'],
                                  cfg['learner']['num_mini_batch'],
                                  cfg['learner']['value_loss_coef'],
                                  cfg['learner']['entropy_coef'],
                                  lr=cfg['learner']['lr'],
                                  eps=cfg['learner']['eps'],
                                  max_grad_norm=cfg['learner']['max_grad_norm'])
            start_epoch = 0

        
        # Machinery for storing rollouts
        num_train_processes = cfg['env']['num_processes'] - cfg['env']['num_val_processes']
        num_val_processes = cfg['env']['num_val_processes']
        assert cfg['env']['num_val_processes'] < cfg['env']['num_processes'], "Can't train without some training processes!"
        current_obs = StackedSensorDictStorage(cfg['env']['num_processes'], cfg['learner']['num_stack'], retained_obs_shape)
        current_train_obs = StackedSensorDictStorage(num_train_processes, cfg['learner']['num_stack'], retained_obs_shape)
        logger.debug(f'Stacked obs shape {current_obs.obs_shape}')

        if cfg['learner']['use_replay']:
            rollouts = RolloutSensorDictReplayBuffer(
                                cfg['learner']['num_steps'],
                                num_train_processes,
                                current_obs.obs_shape,
                                action_space,
                                cfg['learner']['internal_state_size'],
                                actor_critic,
                                cfg['learner']['use_gae'],
                                cfg['learner']['gamma'],
                                cfg['learner']['tau'],
                                cfg['learner']['replay_buffer_size'])
        else:
            rollouts = RolloutSensorDictStorage(
                                cfg['learner']['num_steps'],
                                num_train_processes,
                                current_obs.obs_shape,
                                action_space,
                                cfg['learner']['internal_state_size'])

        # Set up logging
        if cfg['saving']['logging_type'] == 'visdom':
            mlog = tnt.logger.VisdomMeterLogger(
                                title=uuid, env=uuid,
                                server=cfg['saving']['visdom_server'],
                                port=cfg['saving']['visdom_port'],
                                log_to_filename=cfg['saving']['visdom_log_file'])
        elif cfg['saving']['logging_type'] == 'tensorboard':
            mlog = tnt.logger.TensorboardMeterLogger(
                                env=uuid,
                                log_dir=cfg['saving']['log_dir'],
                                plotstylecombined=True)
        else:
            raise NotImplementedError("Unknown logger type: ({cfg['saving']['logging_type']})")

        # Add metrics and logging to TB/Visdom
        loggable_metrics = ['metrics/rewards',
                            'diagnostics/dist_perplexity',
                            'diagnostics/lengths',
                            'diagnostics/max_importance_weight',
                            'diagnostics/value',
                            'losses/action_loss',
                            'losses/dist_entropy',
                            'losses/value_loss']
        core_metrics     = ['metrics/rewards', 'diagnostics/lengths']
        debug_metrics    = ['debug/input_images']
        if 'habitat' in cfg['env']['env_name'].lower():
            for metric in ['metrics/spl', 'metrics/success']:
                loggable_metrics.append(metric)
                core_metrics.append(metric)
        for meter in loggable_metrics:
            mlog.add_meter(meter, tnt.meter.ValueSummaryMeter())
        for debug_meter in debug_metrics:
            mlog.add_meter(debug_meter, tnt.meter.SingletonMeter(), ptype='image')
        mlog.add_meter('config', tnt.meter.SingletonMeter(), ptype='text')
        mlog.update_meter(cfg_to_md(cfg, uuid), meters={'config'}, phase='train')
        
        # File loggers
        flog = tnt.logger.FileLogger(cfg['saving']['results_log_file'], overwrite=True)
        reward_only_flog = tnt.logger.FileLogger(cfg['saving']['reward_log_file'], overwrite=True)

        # replay data to mlog, move metadata file
        if changed_log_dir:
            evkit.utils.logging.replay_logs(existing_log_paths, mlog)
            evkit.utils.logging.move_metadata_file(old_log_dir, cfg['saving']['log_dir'], uuid)

        ##########
        # LEARN! #
        ##########
        if cfg['training']['cuda']:
            current_train_obs = current_train_obs.cuda()
            current_obs = current_obs.cuda()
            rollouts.cuda()
            actor_critic.cuda()

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([cfg['env']['num_processes'], 1])
        episode_lengths = torch.zeros([cfg['env']['num_processes'], 1])

        # First observation
        obs = envs.reset()
        current_obs.insert(obs) 
        mask_done = torch.FloatTensor([[0.0] for _ in range(cfg['env']['num_processes'])]).pin_memory()       
        states = torch.zeros(cfg['env']['num_processes'], cfg['learner']['internal_state_size']).pin_memory()

        # Main loop
        start_time = time.time()
        num_updates = int(cfg['training']['num_frames']) // ( cfg['learner']['num_steps'] * cfg['env']['num_processes'] )
        logger.info(f"Running until num updates == {num_updates}")
        for j in range(start_epoch, num_updates, 1):
            for step in range(cfg['learner']['num_steps']):
                obs_unpacked = {k: current_obs.peek()[k].peek() for k in current_obs.peek()}
                if j == start_epoch and step < 10:
                    log_input_images(obs_unpacked, mlog, num_stack=cfg['learner']['num_stack'], 
                                         key_names=['rgb_filled', 'map'], meter_name='debug/input_images', step_num=step)
 
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, states = actor_critic.act(
                        obs_unpacked,
                        states.cuda(),
                        mask_done.cuda())    
                cpu_actions = list(action.squeeze(1).cpu().numpy())
                obs, reward, done, info = envs.step(cpu_actions)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

                # Handle terminated episodes; logging values and computing the "done" mask
                episode_rewards += reward
                episode_lengths += (1 + cfg['env']['additional_repeat_count'])
                mask_done = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                for i, (r, l, done_) in enumerate(zip(episode_rewards, episode_lengths, done)):  # Logging loop
                    if done_:
                        phase = 'train' if i < num_train_processes else 'val'
                        mlog.update_meter(r, meters={'metrics/rewards'}, phase=phase)
                        mlog.update_meter(l, meters={'diagnostics/lengths'}, phase=phase)
                        if 'habitat' in cfg['env']['env_name'].lower():
                            mlog.update_meter(info[i]["spl"], meters={'metrics/spl'}, phase=phase)
                            mlog.update_meter(np.ceil(info[i]["spl"]), meters={'metrics/success'}, phase=phase)
                episode_rewards *= mask_done
                episode_lengths *= mask_done

                # Insert the new observation into RolloutStorage
                if cfg['training']['cuda']:
                    mask_done = mask_done.cuda()
                for k in obs:
                    if k in current_train_obs.sensor_names:
                        current_train_obs[k].insert(obs[k][:num_train_processes], mask_done[:num_train_processes])
                current_obs.insert(obs, mask_done)
                rollouts.insert(current_train_obs.peek(),
                                states[:num_train_processes],
                                action[:num_train_processes],
                                action_log_prob[:num_train_processes],
                                value[:num_train_processes],
                                reward[:num_train_processes],
                                mask_done[:num_train_processes])
                mlog.update_meter(value[:num_train_processes].mean().item(), meters={'diagnostics/value'}, phase='train')

            # Training update
            if not cfg['learner']['test']:
                if not cfg['learner']['use_replay']:
                    # Moderate compute saving optimization (if no replay buffer):
                    #     Estimate future-discounted returns only once
                    with torch.no_grad():
                        next_value = actor_critic.get_value(rollouts.observations.at(-1),
                                                            rollouts.states[-1],
                                                            rollouts.masks[-1]).detach()
                    rollouts.compute_returns(next_value, cfg['learner']['use_gae'], cfg['learner']['gamma'], cfg['learner']['tau'])
                value_loss, action_loss, dist_entropy, max_importance_weight, info = agent.update(rollouts)
                rollouts.after_update()  # For the next iter: initial obs <- current observation
                
                # Update meters with latest training info
                mlog.update_meter(dist_entropy,          meters={'losses/dist_entropy'})
                mlog.update_meter(np.exp(dist_entropy),  meters={'diagnostics/dist_perplexity'})
                mlog.update_meter(value_loss,            meters={'losses/value_loss'})
                mlog.update_meter(action_loss,           meters={'losses/action_loss'})
                mlog.update_meter(max_importance_weight, meters={'diagnostics/max_importance_weight'})

            # Main logging
            if (j) % cfg['saving']['log_interval'] == 0:
                n_steps_since_logging = cfg['saving']['log_interval'] * num_train_processes * cfg['learner']['num_steps']
                total_num_steps = (j + 1) * num_train_processes * cfg['learner']['num_steps']
                logger.info("Update {}, num timesteps {}, FPS {}".format(
                            j + 1,
                            total_num_steps,
                            int(n_steps_since_logging / (time.time() - start_time))
                        ))
                for metric in core_metrics:   # Log to stdout
                    for mode in ['train', 'val']:
                        if metric in core_metrics or mode == 'train':
                            mlog.print_meter(mode, total_num_steps, meterlist={metric})
                for mode in ['train', 'val']: # Log to files
                    results = mlog.peek_meter(phase=mode)                    
                    reward_only_flog.log(mode, {metric: results[metric] for metric in core_metrics})
                    if mode == 'train':
                        results['step_num'] = j + 1
                        flog.log('all_results', results)

                    mlog.reset_meter(total_num_steps, mode=mode)
                start_time = time.time()

            # Save checkpoint
            if j % cfg['saving']['save_interval'] == 0:
                save_dir_absolute = os.path.join(cfg['saving']['log_dir'], cfg['saving']['save_dir'])
                save_checkpoint(
                    { 'agent': agent, 'epoch': j },
                    save_dir_absolute, j)
    # Clean up (either after ending normally or early [e.g. from a KeyboardInterrupt])
    finally:
        try:
            if isinstance(envs, list):
                [env.close() for env in envs]
            else:
                envs.close()
            logger.info("Killed envs.")
        except UnboundLocalError:
            logger.info("No envs to kill!")


if is_interactive() and __name__ == '__main__':
    assert LOG_DIR, 'log dir cannot be empty'
    os.makedirs(LOG_DIR, exist_ok=True)
    subprocess.call("rm -rf {}/*".format(LOG_DIR), shell=True)
    ex.observers.append(FileStorageObserverWithExUuid.create(LOG_DIR))
    ex.run_commandline(
        'run_config with \
            uuid="gibson_random" \
            cfg.env.num_processes=1\
            '.format())
elif __name__ == '__main__':
    assert LOG_DIR, 'log dir cannot be empty'
    os.makedirs(LOG_DIR, exist_ok=True)
    subprocess.call("rm -rf {}/*".format(LOG_DIR), shell=True)
    ex.observers.append(FileStorageObserverWithExUuid.create(LOG_DIR))
    try:
        ex.run_commandline()
    except FileNotFoundError as e:
        logger.error(f'File not found! Are you trying to test an experiment with the uuid: {e}?')
        raise e
else: 
    logger.info(__name__)
