import os

import gym
import numpy as np
from gym.spaces.box import Box
from gym import wrappers, logger

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from .wrappers import VisdomMonitor, ProcessObservationWrapper, SkipWrapper, SensorEnvWrapper
from .distributed_factory import DistributedEnv

# from evkit.env.gibson.gibsonenv import GibsonEnv, DummyGibsonEnv
from evkit.env.habitat.habitatenv import make_habitat_vector_env

try:
    from evkit.env.vizdoom import *
except ImportError:
    pass

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

import torch


DEFAULT_SENSOR_NAME = "DEFAULT"

class EnvFactory(object):
    @staticmethod
    def vectorized(env_id, seed, num_processes, log_dir, add_timestep,
             sensors={DEFAULT_SENSOR_NAME: None},
             addl_repeat_count=0, preprocessing_fn=None,
             env_specific_kwargs={},
             vis_interval=20,
             visdom_name='main',
             visdom_log_file=None,
             visdom_server='localhost',
             visdom_port='8097',
             num_val_processes=0,
             gae_gamma=None):
        '''Returns vectorized environment. Either the simulator implements this (habitat) or
           'vectorized' uses the call_to_run helper
        '''
        simulator, scenario = env_id.split('_')
        if simulator.lower() in ['habitat']: # These simulators internally handle vectorization/distribution
            env = make_habitat_vector_env(
                            num_processes=num_processes,
                            preprocessing_fn=preprocessing_fn,
                            log_dir=log_dir,
                            num_val_processes=num_val_processes,
                            vis_interval=vis_interval,
                            visdom_name=visdom_name,
                            visdom_log_file=visdom_log_file,
                            visdom_server=visdom_server,
                            visdom_port=visdom_port,
                            seed=seed,
                            **env_specific_kwargs)         
        else: # These simulators must be manually vectorized
            envs = [ EnvFactory.call_to_run(env_id, seed, 
                            rank, log_dir, add_timestep,
                            sensors=sensors,
                            addl_repeat_count=addl_repeat_count,
                            preprocessing_fn=preprocessing_fn,
                            env_specific_kwargs=env_specific_kwargs,
                            vis_interval=vis_interval,
                            visdom_name=visdom_name,
                            visdom_log_file=visdom_log_file,
                            visdom_server=visdom_server,
                            visdom_port=visdom_port,
                            num_val_processes=num_val_processes,
                            num_processes=num_processes)
                     for rank in range(num_processes) ]
            if num_processes == 1:
                env = DummyVecEnv(envs)
            else:
                env = DistributedEnv.new(envs,
                            gae_gamma=gae_gamma,
                            distribution_method=DistributedEnv.distribution_schemes.vectorize)
        return env

        
    
    @staticmethod
    def call_to_run(env_id, seed, rank, log_dir, add_timestep,
             sensors={DEFAULT_SENSOR_NAME: None},
             addl_repeat_count=0, preprocessing_fn=None,
             gibson_config=None,
             blank_sensor=False,
             start_locations_file=None,
             target_dim=16,
             blind=False,
             env_specific_kwargs=None,
             vis_interval=20,
             visdom_name='main',
             visdom_log_file=None,
             visdom_server='localhost',
             visdom_port='8097',
             num_val_processes=0,
             num_processes=1):
        '''Returns a function which can be called to instantiate a new environment.
            
            
            Args:
                env_id: Name of the ID to make
                seed:   random seed for environment
                rank:   environment number (i of k)
                log_dir:    directory to log to
                add_timestep:   ???
                sensors: A configuration of sensor names -> specs (for now, just none)
                preprocessing_fn(env): function which returns (transform, obs_shape)
                    transform(obs): a function that is run on every obs
                    obs_shape: the final shape of transform(obs)
                gibson_config: If using gibson, which config to use
                visdon_name: If using visdom, what to name the visdom environment
                visdom_log_file: Where to store visdom logging entries. This allows replaying
                    training back to visdom. If this is set to none, then disable visdom logging. 
                visdom_server: visdom server ip (http:// is automatically appended)
                visdom_port: Which port the visdom server is listening on
            
            Returns:
                A callable function (no parameters) which instantiates an enviroment.
        '''
        simulator, scenario = env_id.split('_')
        if env_specific_kwargs is None:
            env_specific_kwargs = {}
        def _thunk():
            preprocessing_fn_implemented_inside_env = False
            logging_implemented_inside_env = False
            already_distributed = False
            if env_id.startswith("dm"):
                _, domain, task = env_id.split('.')
                env = dm_control2gym.make(domain_name=domain, task_name=task)
            elif env_id.startswith("Gibson"):
                env = GibsonEnv(env_id=env_id,
                                gibson_config=gibson_config,
                                blind=blind,
                                blank_sensor=blank_sensor,
                                start_locations_file=start_locations_file,
                                target_dim=target_dim, 
                                **env_specific_kwargs)
            elif env_id.startswith("DummyGibson"):
                env = DummyGibsonEnv(env_id=env_id,
                                     gibson_config=gibson_config,
                                     blind=blind,
                                     blank_sensor=blank_sensor,
                                     start_locations_file=start_locations_file,
                                     target_dim=target_dim,
                                     **env_specific_kwargs)
            elif env_id.startswith("Doom"):
                env_specific_kwargs['repeat_count'] = addl_repeat_count + 1
                num_train_processes = num_processes - num_val_processes
                # 1 (train only), 2 test only
                env_specific_kwargs['randomize_textures'] = 1 if rank < num_train_processes else 2
                vizdoom_class = eval(scenario.split('.')[0])
                env = vizdoom_class(**env_specific_kwargs)
            elif env_id.startswith("Habitat"):
                env = make_habitat_vector_env(
                                num_processes=rank,
                                target_dim=target_dim,
                                preprocessing_fn=preprocessing_fn,
                                log_dir=log_dir,
                                num_val_processes=num_val_processes,
                                visdom_name=visdom_name,
                                visdom_log_file=visdom_log_file,
                                visdom_server=visdom_server,
                                visdom_port=visdom_port,
                                seed=seed,
                                **env_specific_kwargs)
                already_distributed = True
                preprocessing_fn_implemented_inside_env = True
                logging_implemented_inside_env = True
            else:
                env = gym.make(env_id)

            if already_distributed: # Env is now responsible for logging, preprocessing, repeat_count
                return env

            is_atari = hasattr(gym.envs, 'atari') and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(env_id)


            if add_timestep:
                raise NotImplementedError("AddTimestep not implemented for SensorDict")
                obs_shape = env.observation_space.shape
                if add_timestep and len(obs_shape) == 1 \
                    and str(env).find('TimeLimit') > -1:
                    env = AddTimestep(env)

            if not (logging_implemented_inside_env or log_dir is None):
                os.makedirs(os.path.join(log_dir, visdom_name), exist_ok=True)
                print("Visdom log file", visdom_log_file)
                first_val_process = num_processes - num_val_processes
                if (rank == 0 or rank == first_val_process) and visdom_log_file is not None:
                    env = VisdomMonitor(env,
                                   directory=os.path.join(log_dir, visdom_name),
                                   video_callable=lambda x: x % vis_interval == 0,
                                   uid=str(rank),
                                   server=visdom_server,
                                   port=visdom_port,
                                   visdom_log_file=visdom_log_file,
                                   visdom_env=visdom_name)
                else:
                    print("Not using visdom")
                    env = wrappers.Monitor(env,
                                   directory=os.path.join(log_dir, visdom_name),
                                   uid=str(rank))

            if is_atari:
                env = wrap_deepmind(env)
            if addl_repeat_count > 0:
                if not hasattr(env, 'repeat_count') and not hasattr(env.unwrapped, 'repeat_count'):
                    env = SkipWrapper(repeat_count)(env)

            if sensors is not None:
                if hasattr(env, 'is_embodied') or hasattr(env.unwrapped, 'is_embodied'):
                    pass
                else:
                    assert len(sensors) == 1, 'Can only handle one sensor'
                    sensor_name = list(sensors.keys())[0]
                    env = SensorEnvWrapper(env, name=sensor_name)
                
            if not (preprocessing_fn_implemented_inside_env or preprocessing_fn is None):
                transform, space = preprocessing_fn(env.observation_space)
                env = ProcessObservationWrapper(env, transform, space)
            env.seed(seed + rank)
            return env

        return _thunk

class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

    
