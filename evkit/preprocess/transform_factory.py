from collections import defaultdict
import numpy as np
import skimage
import torchvision as vision
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing.dummy as mp
import multiprocessing
from gym import spaces

from evkit.sensors import SensorPack

class TransformFactory(object):
# TransformFactory.independent(
#     {
#         {
#             'taskonomy': taskonomy_features_transform('{taskonomy_encoder}', encoder_type='{encoder_type}'),
#             'rgb_filled':rescale_centercrop_resize((3, {image_dim}, {image_dim})),
#             'map':rescale_centercrop_resize((1,84,84))
#         }
#     }
# )
    @staticmethod
    def independent(names_to_transforms, multithread=False, keep_unnamed=True):
        def processing_fn(obs_space):
            ''' Obs_space is expected to be a 1-layer deep spaces.Dict '''
            transforms = {}
            sensor_space = {}
            transform_names = set(names_to_transforms.keys())
            obs_space_names = set(obs_space.spaces.keys())
            assert transform_names.issubset(obs_space_names), \
                "Trying to transform observations that are not present ({})".format(
                transform_names - obs_space_names)
            for name in obs_space_names:
                if name in names_to_transforms:
                    transform = names_to_transforms[name]
                    transforms[name], sensor_space[name] = transform(obs_space.spaces[name])
                elif keep_unnamed:
                    sensor_space[name] = obs_space.spaces[name]
                else:
                    print(f'Did not transform {name}, removing from obs')
            
            def _independent_tranform_thunk(obs):
                results = {}
                if multithread:
                    pool = mp.pool(min(mp.cpu_count(), len(sensor_shapes)))
                    pool.map()
                else:
                    for name, transform in transforms.items():
                        try:
                            results[name] = transform(obs[name])
                        except Exception as e:
                            print(f'Problem applying preproces transform to {name}.', e)
                            raise e
                for name, val in obs.items():
                    if name not in results and keep_unnamed:
                        results[name] = val
                return SensorPack(results)

            return _independent_tranform_thunk, spaces.Dict(sensor_space)
        return processing_fn

    @staticmethod
    def splitting(names_to_transforms, multithread=False, keep_unnamed=True):
        def processing_fn(obs_space):
            ''' Obs_space is expected to be a 1-layer deep spaces.Dict '''
            old_name_to_new_name_to_transform = defaultdict(dict)
            sensor_space = {}
            transform_names = set(names_to_transforms.keys())
            obs_space_names = set(obs_space.spaces.keys())
            assert transform_names.issubset(obs_space_names), \
                "Trying to transform observations that are not present ({})".format(
                transform_names - obs_space_names)
            for old_name in obs_space_names:
                if old_name in names_to_transforms:
                    assert hasattr(names_to_transforms, 'items'), 'each sensor must map to a dict of transfors'
                    for new_name, transform_maker in names_to_transforms[old_name].items():
                        transform, sensor_space[new_name] = transform_maker(obs_space.spaces[old_name])
                        old_name_to_new_name_to_transform[old_name][new_name] = transform
                elif keep_unnamed:
                    sensor_space[old_name] = obs_space.spaces[old_name]
            
            def _transform_thunk(obs):
                results = {}
                transforms_to_run = []
                for old_name, new_names_to_transform in old_name_to_new_name_to_transform.items():
                        for new_name, transform in new_names_to_transform.items():
                            transforms_to_run.append((old_name, new_name, transform))
                if multithread:
                    pool = mp.Pool(min(multiprocessing.cpu_count(), len(transforms_to_run)))
                    # raise NotImplementedError("'multithread' not yet implemented for TransformFactory.splitting")    
                    res = pool.map(lambda t_o: t_o[0](t_o[1]),
                                   zip([t for _, _, t in transforms_to_run],[obs[old_name] for old_name, _, _ in transforms_to_run]))
                    for transformed, (old_name, new_name, _) in zip(res, transforms_to_run):
                        results[new_name] = transformed
                else:
                    for old_name, new_names_to_transform in old_name_to_new_name_to_transform.items():
                        for new_name, transform in new_names_to_transform.items():
                            results[new_name] = transform(obs[old_name])
                if keep_unnamed:
                    for name, val in obs.items():
                        if name not in results and name not in old_name_to_new_name_to_transform:
                            results[name] = val
                return SensorPack(results)

            return _transform_thunk, spaces.Dict(sensor_space)
        return processing_fn
            
    
    