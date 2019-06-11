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
from evkit.models.taskonomy_network import TaskonomyDecoder, TaskonomyEncoder, TaskonomyNetwork

RESCALE_0_1_NEG1_POS1 = vision.transforms.Normalize([0.5,0.5,0.5], [0.5, 0.5, 0.5])
RESCALE_NEG1_POS1_0_1 = vision.transforms.Normalize([-1.,-1.,-1.], [2., 2., 2.])
RESCALE_0_255_NEG1_POS1 = vision.transforms.Normalize([127.5,127.5,127.5], [255, 255, 255])

class Pipeline(object):
    
    def __init__(self, env_or_pipeline):
        pass
    
    def forward(self):
        pass

# Remember to import these into whomever does the eval(preprocessing_fn) - habitatenv and evaluate_habitat
def identity_transform():
    def _thunk(obs_space):
        return lambda x: x, obs_space
    return _thunk    

def fill_like(output_size, fill_value=0.0, dtype=torch.float32):
    def _thunk(obs_space):
        tensor = torch.ones((1,), dtype=dtype)
        def _process(x):
            return tensor.new_full(output_size, fill_value).numpy()
        return _process, spaces.Box(-1, 1, output_size, tensor.numpy().dtype)
    return _thunk


def rescale_centercrop_resize(output_size, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)

            obs_space: Should be form WxHxC
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    def _rescale_centercrop_resize_thunk(obs_space):
        obs_shape = obs_space.shape
        obs_min_wh = min(obs_shape[:2])
        output_wh = output_size[-2:]  # The out
        processed_env_shape = output_size

        pipeline = vision.transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.CenterCrop([obs_min_wh, obs_min_wh]),
            vision.transforms.Resize(output_wh),    
            vision.transforms.ToTensor(),
            RESCALE_0_1_NEG1_POS1,
        ])

        return pipeline, spaces.Box(-1, 1, output_size, dtype)
    return _rescale_centercrop_resize_thunk


def rescale_centercrop_resize_collated(output_size, dtype=np.float32):
    # WARNING: I will leave this here in case previous models use it, but this is semantically not correct - we do not do any processing
    ''' rescale_centercrop_resize

        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)

            obs_space: Should be form WxHxC
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    def _rescale_centercrop_resize_thunk(obs_space):
        obs_shape = obs_space.shape
        obs_min_wh = min(obs_shape[:2])
        output_wh = output_size[-2:]  # The out
        processed_env_shape = output_size

        pipeline = vision.transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.CenterCrop([obs_min_wh, obs_min_wh]),
            vision.transforms.Resize(output_wh),
            vision.transforms.ToTensor(),
            RESCALE_0_1_NEG1_POS1,
        ])

        def iterative_pipeline(pipeline):
            def runner(x):
                if isinstance(x, torch.Tensor):  # for training
                    x = torch.cuda.FloatTensor(x.cuda())
                else:  # for testing
                    x = torch.cuda.FloatTensor(x).cuda()

                x = x.permute(0, 3, 1, 2) / 255.0 #.view(1, 3, 256, 256)
                x = 2.0 * x - 1.0
                return x
                # if isinstance(x, torch.Tensor):  # for training
                #     _, h,w,c = x.shape
                #     iterative_ret = [pipeline(x_.view(c,h,w).to(torch.uint8)) for x_ in x]
                # elif isinstance(x, np.ndarray):  # for testing
                #     iterative_ret = [pipeline(x_) for x_ in x]
                # else:
                #     assert False, f'transform does not like {type(x)}'
                # return torch.stack(iterative_ret)
            return runner

        return iterative_pipeline(pipeline), spaces.Box(-1, 1, output_size, dtype)
    return _rescale_centercrop_resize_thunk


def rescale():
    ''' Rescales observations to a new values

        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    def _rescale_thunk(obs_space):
        obs_shape = obs_space.shape
        np_pipeline = vision.transforms.Compose([
            vision.transforms.ToTensor(),
            RESCALE_0_1_NEG1_POS1,
        ])
        def pipeline(im):
            if isinstance(im, np.ndarray):
                return np_pipeline(im)
            else:
                return RESCALE_0_255_NEG1_POS1(im)
        return pipeline, spaces.Box(-1.0, 1.0, obs_space.shape, np.float32)
    return _rescale_thunk


def grayscale_rescale():
    ''' Rescales observations to a new values

        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    def _grayscale_rescale_thunk(obs_space):
        pipeline = vision.transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.Grayscale(),
            vision.transforms.ToTensor(),
            vision.transforms.Normalize([0.5], [0.5])
        ])
        obs_shape = obs_space.shape
        return pipeline, spaces.Box(-1.0, 1.0,
                             (1, obs_shape[0], obs_shape[1]),
                             dtype=np.float32)
    return _grayscale_rescale_thunk



def cross_modal_transform(eval_to_get_net, output_shape=(3,84,84), dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    _rescale_thunk = rescale_centercrop_resize((3, 256, 256))
    output_size = output_shape[-1]
    output_shape = output_shape
    net = eval_to_get_net
    resize_fn = vision.transforms.Compose([
            vision.transforms.ToPILImage(),
            vision.transforms.Resize(output_size),    
            vision.transforms.ToTensor(),
            RESCALE_0_1_NEG1_POS1,
        ])

    def encode(x):
        with torch.no_grad():
            return net(x)
    
    def _transform_thunk(obs_space):
        rescale, _ = _rescale_thunk(obs_space)
        def pipeline(x):
            with torch.no_grad():
                if isinstance(x, torch.Tensor):
                    x = torch.cuda.FloatTensor(x.cuda())
                else:  # for testing
                    x = torch.cuda.FloatTensor(x).cuda()
                x = encode(x)
                y = (x + 1.) / 2
                z = torch.stack([resize_fn(y_.cpu()) for y_ in y])
                return z
        return pipeline, spaces.Box(-1, 1, output_shape, dtype)
    
    return _transform_thunk


def image_to_input_collated(output_size, dtype=np.float32):
    def _thunk(obs_space):
        def runner(x):
            # input: n x h x w x c
            # output: n x c x h x w  and normalized, ready to pass into net.forward
            assert x.shape[2] == x.shape[1], 'Input image must be square, of the form: N,H,W,C'
            if isinstance(x, torch.Tensor):  # for training
                x = torch.cuda.FloatTensor(x.cuda())
            else:  # for testing
                x = torch.cuda.FloatTensor(x.copy()).cuda()

            x = x.permute(0, 3, 1, 2) / 255.0 #.view(1, 3, 256, 256)
            x = 2.0 * x - 1.0
            return x

        return runner, spaces.Box(-1, 1, output_size, dtype)
    return _thunk


def map_pool(output_size, dtype=np.float32):
    def _thunk(obs_space):
        def runner(x):
            with torch.no_grad():
                # input: h x w x c
                # output: c x h x w  and normalized, ready to pass into net.forward
                assert x.shape[0] == x.shape[1], 'we are only using square data, data format: N,H,W,C'
                if isinstance(x, torch.Tensor):  # for training
                    x = torch.cuda.FloatTensor(x.cuda())
                else:  # for testing
                    x = torch.cuda.FloatTensor(x.copy()).cuda()

                x.unsqueeze_(0)

                x = x.permute(0, 3, 1, 2) / 255.0 #.view(1, 3, 256, 256)
                x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
                x = torch.rot90(x, k=2, dims=(2,3)) # Face north (this could be computed using a different world2agent transform)
                x = 2.0 * x - 1.0

                x.squeeze_(0)
#                 print(x.shape)
                return x.cpu()

        return runner, spaces.Box(-1, 1, output_size, dtype)
    return _thunk


def map_pool_collated(output_size, dtype=np.float32):
    def _thunk(obs_space):
        def runner(x):
            with torch.no_grad():
                # input: n x h x w x c
                # output: n x c x h x w  and normalized, ready to pass into net.forward
                assert x.shape[2] == x.shape[1], 'we are only using square data, data format: N,H,W,C'
                if isinstance(x, torch.Tensor):  # for training
                    x = torch.cuda.FloatTensor(x.cuda())
                else:  # for testing
                    x = torch.cuda.FloatTensor(x.copy()).cuda()

                x = x.permute(0, 3, 1, 2) / 255.0 #.view(1, 3, 256, 256)
                x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
                x = torch.rot90(x, k=2, dims=(2,3)) # Face north (this could be computed using a different world2agent transform)
                x = 2.0 * x - 1.0
                return x

        return runner, spaces.Box(-1, 1, output_size, dtype)
    return _thunk




def taskonomy_features_transform(task_path, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    # print(task_path)
    net = TaskonomyEncoder().cuda()
    net.eval()
    checkpoint = torch.load(task_path)
    net.load_state_dict(checkpoint['state_dict'])
    def encode(x):
        with torch.no_grad():
            return net(x)

    def _taskonomy_features_transform_thunk(obs_space):
        def pipeline(x):
            x = torch.Tensor(x).cuda()
            x = encode(x)
            return x.cpu()
        return pipeline, spaces.Box(-1, 1, (8, 16, 16), dtype)
    
    return _taskonomy_features_transform_thunk

def _load_encoder(encoder_path):
    net = TaskonomyEncoder() #.cuda()
    net.eval()
    checkpoint = torch.load(encoder_path)
    net.load_state_dict(checkpoint['state_dict'])
    for p in net.parameters():
        p.requires_grad = False
    # net = Compose(nn.GroupNorm(32, 32, affine=False), net)
    return net

def _load_encoders_seq(encoder_paths):
    experts = [_load_encoder(encoder_path) for encoder_path in encoder_paths]
    experts = [e.cuda() for e in experts]
    return experts

def _load_encoders_parallel(encoder_paths, n_processes=None):
    # This is not working right now for some reason
    n_processes = len(encoder_paths) if n_processes is None else min(len(encoder_paths), n_processes)
    n_parallel = min(multiprocessing.cpu_count(), n_processes)
    pool = multiprocessing.Pool(min(n_parallel, n_processes))
    experts = pool.map(_load_encoder, encoder_paths)
    pool.close()
    pool.join()
    experts = [e.cuda() for e in experts]
    return experts

def taskonomy_multi_features_transform(task_paths, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    nets = _load_encoders_seq(task_paths)
    def encode(x):
        with torch.no_grad():
            return torch.cat([net(x) for net in nets], dim=1)

    def _taskonomy_features_transform_thunk(obs_space):
        def pipeline(x):
            x = torch.Tensor(x).cuda()
            x = encode(x)
            return x.cpu()
        return pipeline, spaces.Box(-1, 1, (8*len(nets), 16, 16), dtype)
    
    return _taskonomy_features_transform_thunk












def taskonomy_features_transform_collated(task_path, dtype=np.float32):
    ''' rescale_centercrop_resize
    
        Args:
            output_size: A tuple CxWxH
            dtype: of the output (must be np, not torch)
            
        Returns:
            a function which returns takes 'env' and returns transform, output_size, dtype
    '''
    # print(task_path)
    net = TaskonomyEncoder().cuda()
    net.eval()
    checkpoint = torch.load(task_path)
    net.load_state_dict(checkpoint['state_dict'])
    def encode(x):
        with torch.no_grad():
            x = torch.Tensor(x).cuda()
            if isinstance(x, torch.Tensor):  # for training
                x = torch.cuda.FloatTensor(x.cuda())
            else:  # for testing
                x = torch.cuda.FloatTensor(x).cuda()

            x = x.permute(0, 3, 1, 2) / 255.0 #.view(1, 3, 256, 256)
            x = 2.0 * x - 1.0
            return net(x)

    def _taskonomy_features_transform_thunk(obs_space):
        pipeline = lambda x: encode(x).cpu()
        return pipeline, spaces.Box(-1, 1, (8, 16, 16), dtype)
    
    return _taskonomy_features_transform_thunk


# def taskonomy_features_transform_collated(task_path, encoder_type='taskonomy', dtype=np.float32):
#     ''' rescale_centercrop_resize
    
#         Args:
#             output_size: A tuple CxWxH
#             dtype: of the output (must be np, not torch)
            
#         Returns:
#             a function which returns takes 'env' and returns transform, output_size, dtype
#     '''
#     _rescale_thunk = rescale_centercrop_resize((3, 256, 256))
#     _pixels_as_state_thunk = pixels_as_state((8, 16, 16))  # doubt this works... because we need to reshape and that's not impl at collate
#     if task_path != 'pixels_as_state' and task_path != 'blind':
#         if encoder_type == 'taskonomy':
#             net = TaskonomyEncoder(normalize_outputs=False)  # Note this change! We do not normalize the encoder on default now
#         if task_path != 'None':
#             checkpoint = torch.load(task_path)
#             if any([isinstance(v, nn.Module) for v in checkpoint.values()]):
#                 net = [v for v in checkpoint.values() if isinstance(v, nn.Module)][0]
#             elif 'state_dict' in checkpoint.keys():
#                 net.load_state_dict(checkpoint['state_dict'])
#             else:
#                 assert False, f'Cannot read task_path {task_path}, no nn.Module or state_dict found. Encoder_type is {encoder_type}'
#         net = net.cuda()
#         net.eval()

#     def encode(x):
#         if task_path == 'pixels_as_state' or task_path == 'blind':
#             return x
#         with torch.no_grad():
#             return net(x)
    
#     def _taskonomy_features_transform_thunk(obs_space):
#         rescale, _ = _rescale_thunk(obs_space)
#         pixels_as_state, _ = _pixels_as_state_thunk(obs_space)
#         def pipeline(x):
#             with torch.no_grad():
#                 if isinstance(x, torch.Tensor):  # for training
#                     x = torch.cuda.FloatTensor(x.cuda())
#                 else:  # for testing
#                     x = torch.cuda.FloatTensor(x).cuda()

#                 x = x.permute(0, 3, 1, 2) / 255.0 #.view(1, 3, 256, 256)
#                 x = 2.0 * x - 1.0
#                 x = encode(x)
#                 return x
#         def pixels_as_state_pipeline(x):
#             return pixels_as_state(x).cpu()
#         def blind_pipeline(x):
#             batch_size = x.shape[0]
#             return torch.zeros((batch_size, 8, 16, 16))
#         if task_path == 'blind':
#             return blind_pipeline, spaces.Box(-1, 1, (8, 16, 16), dtype)
#         elif task_path == 'pixels_as_state':
#             return pixels_as_state_pipeline, spaces.Box(-1, 1, (8, 16, 16), dtype)
#         else:
#             return pipeline, spaces.Box(-1, 1, (8, 16, 16), dtype)
    
#     return _taskonomy_features_transform_thunk

def taskonomy_features_transforms_collated(task_paths, encoder_type='taskonomy', dtype=np.float32):
    # handles multiple taskonomy encoders at once
    num_tasks = 0
    if task_paths != 'pixels_as_state' and task_paths != 'blind':
        task_path_list = [tp.strip() for tp in task_paths.split(',')]
        num_tasks = len(task_path_list)
        assert num_tasks > 0, 'at least need one path'
        if encoder_type == 'taskonomy':
            nets = [TaskonomyEncoder(normalize_outputs=False) for _ in range(num_tasks)]
        else:
            assert False, f'do not recongize encoder type {encoder_type}'
        for i, task_path in enumerate(task_path_list):
            checkpoint = torch.load(task_path)
            net_in_ckpt = [v for v in checkpoint.values() if isinstance(v, nn.Module)]
            if len(net_in_ckpt) > 0:
                nets[i] = net_in_ckpt[0]
            elif 'state_dict' in checkpoint.keys():
                nets[i].load_state_dict(checkpoint['state_dict'])
            else:
                assert False, f'Cannot read task_path {task_path}, no nn.Module or state_dict found. Encoder_type is {encoder_type}'
            nets[i] = nets[i].cuda()
            nets[i].eval()

    def encode(x):
        if task_paths == 'pixels_as_state' or task_paths == 'blind':
            return x
        with torch.no_grad():
            feats = []
            for net in nets:
                feats.append(net(x))
            return torch.cat(feats, dim=1)

    def _taskonomy_features_transform_thunk(obs_space):
        def pipeline(x):
            with torch.no_grad():
                if isinstance(x, torch.Tensor):  # for training
                    x = torch.cuda.FloatTensor(x.cuda())
                else:  # for testing
                    x = torch.cuda.FloatTensor(x).cuda()

                x = x.permute(0, 3, 1, 2) / 255.0 #.view(1, 3, 256, 256)
                x = 2.0 * x - 1.0
                x = encode(x)
                return x
        def pixels_as_state_pipeline(x):
            return pixels_as_state(x).cpu()
        if task_path == 'pixels_as_state':
            return pixels_as_state_pipeline, spaces.Box(-1, 1, (8, 16, 16), dtype)
        else:
            return pipeline, spaces.Box(-1, 1, (8 * num_tasks, 16, 16), dtype)

    return _taskonomy_features_transform_thunk

