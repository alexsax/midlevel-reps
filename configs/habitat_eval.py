# Habitat configs
#   This should be sourced by the evaluate script,
#   which must save a sacred experiment in the variable "ex"
from evkit.utils.misc import remove_whitespace
import os

@ex.config
def cfg_base():
    cfg = {}
    uuid = ""
    config_file = os.path.join(os.getcwd(), 'habitat-api/configs/tasks/pointnav_gibson_val.yaml')
    cfg['eval_kwargs'] = {
        'exp_path': '/mnt/logdir/keypoints3d_encoding_restart1',
        'weights_only_path': None,
        'challenge': True,  # True for challenge.submit() False for benchmark.evaluate()
        'debug': False,  # forces visdom and logs videos
        'overwrite_configs': True,  # for experiments that are not up to date with latest configs, upload this run's cfg into the experiments
        'benchmark_episodes': 10, # up to 994
        'benchmark_config': config_file,
    }

@ex.named_config
def weights_only():
    cfg = {}
    cfg['eval_kwargs'] = {
        'exp_path': None,
        'weights_only_path': '/mnt/eval_runs/curvature_encoding_moresteps_collate5/checkpoints/weights_and_more-latest.dat',
    }


@ex.named_config
def cfg_overwrite():
    cfg = {}
    uuid = "_overwrite"
    cfg['learner'] = {
        'taskonomy_encoder': '/mnt/models/keypoints3d_encoder.dat', # 'None' for random projection, 'pixels_as_state' for pixels as state
        'perception_network': 'features_only',
        'encoder_type': 'taskonomy', # 'taskonomy' for regular encoder or 'atari' for student nets
        'backout': {
            'use_backout': True,
            'patience': 80,
            'unstuck_dist': 0.3,
            'randomize_actions': True,
            'backout_type': 'hardcoded',  # hardcoded, trained
            'backout_ckpt_path': '/mnt/logdir/curvature_encoding_moresteps_collate/checkpoints/ckpt-latest.dat',  # note this one is ckpt path since we do not need configs
            'num_takeover_steps': 8,
        },
        'validator': {
            'use_validator': True,
            'validator_type': 'jerk'
        }
    }
    image_dim = 84
    cfg['env'] = {
        'sensors': {
            'features': None,
            'taskonomy': None,
            'map': None,
            'target': None,
            'global_pos': None,
        },
        'collate_env_obs': False,
        'env_gpus': [0],
        'transform_fn': "TransformFactory.independent({{'taskonomy':taskonomy_features_transform('{taskonomy_encoder}', encoder_type='{encoder_type}'), 'map':image_to_input_pool((3,{image_dim},{image_dim})), 'target':identity_transform(), 'global_pos':identity_transform()}}, keep_unnamed=False)".format(encoder_type=cfg['learner']['encoder_type'], taskonomy_encoder=cfg['learner']['taskonomy_encoder'], image_dim=image_dim),
        'use_target': True,
        'use_map': True,
        'habitat_map_kwargs': {
            'map_building_size': 22, # in meters
            'map_max_pool': False,
            'use_cuda': False,
            'history_size': None,
        },
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_features_transform('{taskonomy_encoder}'),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            taskonomy_encoder='/mnt/models/normal_encoder.dat'),
    }
    cfg['training'] = {
        'seed': 42
    }
    del image_dim

