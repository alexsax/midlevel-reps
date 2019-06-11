# Doom configs
#   This should be sourced by the training script,
#   which must save a sacred experiment in the variable "ex"
#   For descriptions of all fields, see configs/core.py

@ex.named_config
def cfg_doom_navigation():
    uuid = 'doom_visualnavigation'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently only PPO
        'clip_param': 0.1,           # Clip param for trust region in PPO
        'entropy_coef': 0.01,        # Weighting of the entropy term in PPO
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'gamma': 0.99,               # Gamma to use if env.observation_space.shape = 1
        'internal_state_size': 512,  # If using a recurrent policy, what state size to use
        'lr': 0.0001,                # Learning rate for algorithm
        'num_steps': 200,            # Length of each rollout
        'num_mini_batch': 16,        # Size of PPO minibatch
        'num_stack': 4,              # Frames that each cell (CNN) can see
        'max_grad_norm': 0.5,        # Clip grads
        'ppo_epoch': 4,              # Number of times PPO goes over the buffer
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 0.0001,   # Weighting of value_loss in PPO
        'perception_network': 'AtariNet',
        'test':False,
        'use_replay':False,
        'replay_buffer_size': 1000,
        'on_policy_epoch': 4,
        'off_policy_epoch': 0,
    }
    image_dim = 84
    cfg['env'] = {
        'add_timestep': False,       # Add timestep to the observation
        'env_name': 'Doom_VizdoomMultiGoalWithClutterEnv.room-v0',
        "env_specific_args": {
#             "episode_timeout": 1000,
            "episode_timeout": 100,
            "n_clutter_objects": 8,
            "n_goal_objects": 1
          },
        'sensors': {
            'rgb_filled': None,
            'taskonomy': None,
            'map': None,
            'target': None
        },
        'transform_fn_pre_aggregation': None,
        'transform_fn_post_aggregation': None,
        'num_processes': 1,
        'additional_repeat_count': 3,
    }
    
    cfg['saving'] = {
        'port': 8097,
        'log_dir': LOG_DIR,
        'log_interval': 1,
        'save_interval': 100,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis': False,
        'vis_interval': 200,
        'launcher_script': None,
        'visdom_server': 'localhost',
        'visdom_port': '8097',
        'checkpoint': None,
        'checkpoint_configs': False,  # copy the metadata of the checkpoint. YMMV.
    }
    
    cfg['training'] = {
        'cuda': True,
        'seed': random.randint(0,1000),
        'num_frames': 5e6,
        'resumable': True,
    }


@ex.named_config
def scratch_doom():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'doom_scratch'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
             'n_map_channels': 0,
             'use_target': False,
        }
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            "episode_timeout": 1000,
            "n_clutter_objects": 8,
            "n_goal_objects": 1
        },        
        'transform_fn_pre_aggregation': """
            TransformFactory.splitting(
                {
                    'color': {
                        'rgb_filled':rescale_centercrop_resize((3,84,84)) }
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': None,
    }


    
    
    
    
    

@ex.named_config
def cfg_doom_exploration():
    uuid = 'doom_myopicexploration'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently only PPO
        'clip_param': 0.1,           # Clip param for trust region in PPO
        'entropy_coef': 0.01,        # Weighting of the entropy term in PPO
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'gamma': 0.99,               # Gamma to use if env.observation_space.shape = 1
        'internal_state_size': 512,  # If using a recurrent policy, what state size to use
        'lr': 0.0001,                # Learning rate for algorithm
        'num_steps': 200,            # Length of each rollout
        'num_mini_batch': 16,        # Size of PPO minibatch
        'num_stack': 4,              # Frames that each cell (CNN) can see
        'max_grad_norm': 0.5,        # Clip grads
        'ppo_epoch': 4,              # Number of times PPO goes over the buffer
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 0.0001,   # Weighting of value_loss in PPO
        'perception_network': 'AtariNet',
        'test':False,
        'use_replay':False,
        'replay_buffer_size': 1000,
        'on_policy_epoch': 4,
        'off_policy_epoch': 0,
    }
    image_dim = 84
    cfg['env'] = {
        'add_timestep': False,       # Add timestep to the observation
        'env_name': 'Doom_VizdoomExplorationEnv.room-v0',
        "env_specific_args": {
            "episode_timeout": 2000,
          },
        'sensors': {
            'rgb_filled': None,
            'taskonomy': None,
            'map': None,
            'occupancy': None
        },
        'transform_fn_pre_aggregation': None,
        'transform_fn_post_aggregation': None,
        'num_processes': 1,
        'additional_repeat_count': 3,
    }
    
    cfg['saving'] = {
        'port': 8097,
        'log_dir': LOG_DIR,
        'log_interval': 1,
        'save_interval': 100,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis': False,
        'vis_interval': 200,
        'launcher_script': None,
        'visdom_server': 'localhost',
        'visdom_port': '8097',
        'checkpoint': None,
        'checkpoint_configs': False,  # copy the metadata of the checkpoint. YMMV.
    }
    
    cfg['training'] = {
        'cuda': True,
        'seed': random.randint(0,1000),
        'num_frames': 5e5,
        'resumable': True,
    }


@ex.named_config
def scratch_doom_exploration():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'doom_scratch_exploration'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
             'n_map_channels': 1,
             'use_target': False,
        }
    }
    cfg['env'] = {
        'env_specific_kwargs': {
        },        
        'transform_fn_pre_aggregation': """
            TransformFactory.splitting(
                {
                    'color': {
                        'rgb_filled':rescale_centercrop_resize((3,84,84)) },
                    'occupancy': {
                         'map': rescale_centercrop_resize((1,84,84))}
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': None,
    }


