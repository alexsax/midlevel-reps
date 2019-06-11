# Base config values
#   This should be sourced by the training script,
#   which must save a sacred experiment in the variable "ex"

@ex.config
def cfg_base():
    uuid = 'basic'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently only supports PPO
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
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 1e-3,     # Weighting of value_loss in PPO
        'perception_network': 'AtariNet', # The classname of an architecture to use
        'perception_network_kwargs': {},  # kwargs to pass to `perception_network`
        'test':False,                # Whether to only test the agent (not implemented)
        'use_replay':True,           # Use PPO with replay buffer
        'replay_buffer_size':10000,  # Per-process replay buffer size
        'ppo_epoch': 8,              # Number of times PPO goes over the rollouts (if no replay)
        'on_policy_epoch':8,         # Number of times PPO updates from the current rollouts (if replay)
        'off_policy_epoch':8,        # Number of times PPO updates from buffer samples (if replay)
    }
    image_dim = 84
    cfg['env'] = {
        'add_timestep': False,       # Add timestep to the observation
        'env_name': 'Gibson_HuskyVisualExplorationEnv',  # format: SIMULATOR_SCENARIOCONSTRUCTOR
        'env_specific_kwargs': {},   # kwargs to be passed to SCENARIOCONSTRUCTOR above
        'sensors': {},               # Dict of available sensors. Values are not currently used.
        'num_processes': 1,          # Total number of process environments.
        'num_val_processes': 0,      # How many processes to use for val environments
        'additional_repeat_count': 0,                   # How many times to repeat each agent action
        'transform_fn_pre_aggregation': None,           # Transform to apply within each env process
        'transform_fn_post_aggregation': None,          # Transform to apply after aggregation
    }
    
    cfg['saving'] = {
        'autofix_log_dir': False,      # finds one not taken
        'checkpoint': None,            # folder containing subdirectory checkpoints/ckpt-latest.dat
        'checkpoint_configs': False,   # copy the metadata of the checkpoint. YMMV
        'log_dir': LOG_DIR,            # Base directory for all saving and logging
        'log_interval': 10,            # Log stats to terminal and TB/Visdom every k
        'logging_type': 'tensorboard', # One of ['tensorboard', 'visdom']
        'save_interval': 100,          # Save checkpoint every k
        'save_dir': 'checkpoints',     # Subdir in which to save checkpoints
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis_interval': 200,           # Save video every k episodes
        'visdom_port': '8097',         # Port on which visdom listens
        'visdom_server': 'localhost',  # Without an active visdom_server, script will hang or crash
    }
    
    cfg['training'] = {
        'cuda': True,                  # Whether to use CUDA
        'num_frames': 5e5,             # For how many frames to train
        'resumable': True,             # Must be enabled in order to load from checkpoints
        'seed': 42,                    # Initial seed for torch, numpy, random, etc. 
    }