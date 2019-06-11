# Gibson configs
#   This should be sourced by the training script,
#   which must save a sacred experiment in the variable "ex"
#   For descriptions of all fields, see configs/core.py

try:
    from gibson.data.datasets import get_model_path
except ImportError:
    pass
    
@ex.named_config
def cfg_exploration():
    uuid = 'gibson_exploration'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently only PPO
        'clip_param': 0.1,           # Clip param for trust region in PPO
        'entropy_coef': 1e-4,        # Weighting of the entropy term in PPO
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'gamma': 0.99,               # Gamma to use if env.observation_space.shape = 1
        'internal_state_size': 512,  # If using a recurrent policy, what state size to use
        'lr': 1e-4,                  # Learning rate for algorithm
        'num_steps': 512,            # Length of each rollout
        'num_mini_batch': 8,         # Size of PPO minibatch
        'num_stack': 4,              # Frames that each cell (CNN) can see
        'max_grad_norm': 0.5,        # Clip grads
        'ppo_epoch': 8,              # Number of times PPO goes over the buffer
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 1e-3,     # Weighting of value_loss in PPO
        'perception_network': 'AtariNet', # The classname of an architecture to use
        'perception_network_kwargs': {},  # kwargs to pass to `perception_network`
        'test':False,
        'use_replay':True,
        'replay_buffer_size':10000,
        'on_policy_epoch':8,
        'off_policy_epoch':8,
    }
    image_dim = 84
    cfg['env'] = {
        'add_timestep': False,       # Add timestep to the observation
        'env_name': 'Gibson_HuskyVisualExplorationEnv',
        'env_specific_kwargs': {
            'target_dim': 16,        # 2D Tile the target vector to size of the representation
            'gibson_config':'/root/perception_module/evkit/env/gibson/husky_visual_explore_train_noX.yaml',
            'start_locations_file': os.path.join(get_model_path('Beechwood'), 'first_floor_poses.csv'),
            'blind':False,            
            'blank_sensor':True,
        },
        'sensors': {
            'rgb_filled': None,
            'features': None,
            'taskonomy': None,
            'map': None,
            'target': None
        },
        'transform_fn_pre_aggregation': None,
        'transform_fn_post_aggregation': None,
        'num_processes': 1,
        'num_val_processes': 0,
        'additional_repeat_count': 0,
    }
    
    cfg['saving'] = {
        'checkpoint': None,
        'checkpoint_configs': False,  # copy the metadata of the checkpoint. YMMV.
        'log_dir': LOG_DIR,
        'log_interval': 10,
        'save_interval': 100,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis_interval': 200,
        'visdom_server': 'localhost',
        'visdom_port': '8097',
    }
    
    cfg['training'] = {
        'cuda': True,
        'seed': random.randint(0,1000),
        'num_frames': 5e5,
        'resumable': True,
    }    
       
@ex.named_config
def cfg_navigation():
    uuid = 'gibson_visualnavigation'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently only PPO
        'clip_param': 0.1,           # Clip param for trust region in PPO
        'entropy_coef': 1e-4,        # Weighting of the entropy term in PPO
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'gamma': 0.99,               # Gamma to use if env.observation_space.shape = 1
        'internal_state_size': 512,  # If using a recurrent policy, what state size to use
        'lr': 1e-4,                  # Learning rate for algorithm
        'num_steps': 512,            # Length of each rollout
        'num_mini_batch': 8,         # Size of PPO minibatch
        'num_stack': 4,              # Frames that each cell (CNN) can see
        'max_grad_norm': 0.5,        # Clip grads
        'ppo_epoch': 8,              # Number of times PPO goes over the buffer
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 1e-3,     # Weighting of value_loss in PPO
        'perception_network': 'AtariNet', # The classname of an architecture to use
        'perception_network_kwargs': {},  # kwargs to pass to `perception_network`
        'test':False,
        'use_replay':True,
        'replay_buffer_size':10000,
        'on_policy_epoch':8,
        'off_policy_epoch':8,
    }
    image_dim = 84
    cfg['env'] = {
        'add_timestep': False,       # Add timestep to the observation
        'env_name': 'Gibson_HuskyVisualNavigateEnv',
        'env_specific_kwargs': {
            'blind':False,
            'blank_sensor':True,
            'gibson_config':'/root/perception_module/evkit/env/gibson/husky_visual_navigate.yaml',
            'start_locations_file': os.path.join(get_model_path('Beechwood'), 'first_floor_poses.csv'),
        },
        'sensors': {
            'rgb_filled': None,
            'features': None,
            'taskonomy': None,
            'map': None,
            'target': None
        },
        'transform_fn_pre_aggregation': None,
        'transform_fn_post_aggregation': None,
        'num_processes': 1,
        'num_val_processes': 0,
        'repeat_count': 0,

    }
    
    cfg['saving'] = {
        'checkpoint': None,
        'checkpoint_configs': False,  # copy the metadata of the checkpoint. YMMV.
        'log_dir': LOG_DIR,
        'log_interval': 1,
        'save_interval': 100,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis_interval': 200,
        'visdom_server': 'localhost',
        'visdom_port': '8097',
    }
    
    cfg['training'] = {
        'cuda': True,
        'seed': random.randint(0,1000),
        'num_frames': 5e5,
        'resumable': True,
    }

@ex.named_config
def cfg_planning():
    uuid = 'gibson_coordinatenavigation'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently only PPO
        'clip_param': 0.1,           # Clip param for trust region in PPO
        'entropy_coef': 1e-4,        # Weighting of the entropy term in PPO
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'gamma': 0.99,               # Gamma to use if env.observation_space.shape = 1
        'internal_state_size': 512,  # If using a recurrent policy, what state size to use
        'lr': 1e-4,                  # Learning rate for algorithm
        'num_steps': 512,            # Length of each rollout
        'num_mini_batch': 8,         # Size of PPO minibatch
        'num_stack': 4,              # Frames that each cell (CNN) can see
        'max_grad_norm': 0.5,        # Clip grads
        'ppo_epoch': 8,              # Number of times PPO goes over the buffer
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 1e-3,     # Weighting of value_loss in PPO
        'perception_network': 'AtariNet', # The classname of an architecture to use
        'perception_network_kwargs': {},  # kwargs to pass to `perception_network`
        'test':False,
        'use_replay':True,
        'replay_buffer_size':10000,
        'on_policy_epoch':8,
        'off_policy_epoch':8,
    }
    image_dim = 84
    cfg['env'] = {
        'add_timestep': False,       # Add timestep to the observation
        'env_name': 'Gibson_HuskyCoordinateNavigateEnv',
        'env_specific_kwargs': {
            'blind':False,
            'blank_sensor':True,
            'start_locations_file': os.path.join(get_model_path('Beechwood'), 'first_floor_poses.csv'),
            'gibson_config':'/root/perception_module/evkit/env/gibson/husky_coordinate_navigate.yaml',
            'target_dim': 16         # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'sensors': {
            'rgb_filled': None,
            'features': None,
            'taskonomy': None,
            'map': None,
            'target': None
        },
        'transform_fn_pre_aggregation': None,
        'transform_fn_post_aggregation': None,
        'num_processes': 1,
        'num_val_processes': 0,
        'repeat_count': 0,
    }
    
    cfg['saving'] = {
        'checkpoint': None,
        'checkpoint_configs': False,  # copy the metadata of the checkpoint. YMMV.
        'log_dir': LOG_DIR,
        'log_interval': 10,
        'save_interval': 100,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis_interval': 200,
        'visdom_server': 'localhost',
        'visdom_port': '8097',
    }
    
    cfg['training'] = {
        'cuda': True,
        'seed': random.randint(0,1000),
        'num_frames': 5e5,
        'resumable': True,
    }