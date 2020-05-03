from .taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork, TASKONOMY_PRETRAINED_URLS, TASKS_TO_CHANNELS
import multiprocessing.dummy as mp
import torch

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def representation_transform(img, feature_task='normal', device=default_device):
    '''
    Transforms an RGB image into a feature driven by some vision task
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    '''
    return VisualPrior.to_representation(img, feature_tasks=[feature_task], device=device)

def multi_representation_transform(img, feature_tasks=['normal'], device=default_device):
    '''
    Transforms an RGB image into a features driven by some vision tasks
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    '''
    return VisualPrior.to_representation(img, feature_tasks, device)

def max_coverage_featureset_transform(img, k=4, device=default_device):
    '''
    Transforms an RGB image into a features driven by some vision tasks.
    The tasks are chosen according to the Max-Coverage Min-Distance Featureset
    From the paper:
        Mid-Level Visual Representations Improve Generalization and Sample Efficiency 
            for Learning Visuomotor Policies.
        Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
        Arxiv preprint 2018.

    This function expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8*k, 16, 16)
    '''
    return VisualPrior.max_coverage_transform(img, k, device)

def feature_readout(img, feature_task='normal', device=default_device):
    '''
    Transforms an RGB image into a feature driven by some vision task, 
    then returns the result of a readout of the feature.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    '''
    return VisualPrior.to_predicted_label(img, feature_tasks=[feature_task], device=device)

def multi_feature_readout(img, feature_tasks=['normal'], device=default_device):
    '''
    Transforms an RGB image into a features driven by some vision tasks
    then returns the readouts of the features.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    '''
    return VisualPrior.to_predicted_label(img, feature_tasks, device)



class VisualPrior(object):

    max_coverate_featuresets = [
        ['autoencoding'],
        ['segment_unsup2d', 'segment_unsup25d'],
        ['edge_texture', 'reshading', 'curvature'],
        ['normal', 'keypoints2d', 'segment_unsup2d', 'segment_semantic'],
    ]
    model_dir = None
    viable_feature_tasks = [
             'autoencoding',
             'colorization',
             'curvature',
             'denoising',
             'edge_texture',
             'edge_occlusion',
             'egomotion', 
             'fixated_pose', 
             'jigsaw',
             'keypoints2d',
             'keypoints3d',
             'nonfixated_pose',
             'point_matching', 
             'reshading',
             'depth_zbuffer',
             'depth_euclidean',
             'normal',
             'room_layout',
             'segment_unsup25d',
             'segment_unsup2d',
             'segment_semantic',
             'class_object',
             'class_scene',
             'inpainting',
             'vanishing_point']


    @classmethod
    def to_representation(cls, img, feature_tasks=['normal'], device=default_device):
        '''
            Transforms an RGB image into a feature driven by some vision task(s)
            Expects inputs:
                shape  (batch_size, 3, 256, 256)
                values [-1,1]
            Outputs:
                shape  (batch_size, 8, 16, 16)

            This funciton is technically unsupported and there are absolutely no guarantees. 
        '''
        VisualPriorRepresentation._load_unloaded_nets(feature_tasks)
        for t in feature_tasks:
            VisualPriorRepresentation.feature_task_to_net[t] = VisualPriorRepresentation.feature_task_to_net[t].to(device)
        nets = [VisualPriorRepresentation.feature_task_to_net[t] for t in feature_tasks]
        with torch.no_grad():
            return torch.cat([net(img) for net in nets], dim=1)

    @classmethod
    def to_predicted_label(cls, img, feature_tasks=['normal'], device=default_device):
        '''
            Transforms an RGB image into a predicted label for some task.
            Expects inputs:
                shape  (batch_size, 3, 256, 256)
                values [-1,1]
            Outputs:
                shape  (batch_size, C, 256, 256)
                values [-1,1]

            This funciton is technically unsupported and there are absolutely no guarantees. 
        '''
        VisualPriorPredictedLabel._load_unloaded_nets(feature_tasks)
        for t in feature_tasks:
            VisualPriorPredictedLabel.feature_task_to_net[t] = VisualPriorPredictedLabel.feature_task_to_net[t].to(device)
        nets = [VisualPriorPredictedLabel.feature_task_to_net[t] for t in feature_tasks]
        with torch.no_grad():
            return torch.cat([net(img) for net in nets], dim=1)
    
    @classmethod
    def max_coverage_transform(cls, img, k=4, device=default_device):
        assert k > 0, 'Number of features to use for the max_coverage_transform must be > 0'
        if k > 4:
            raise NotImplementedError("max_coverage_transform featureset not implemented for k > 4")
        return cls.to_representation(img, feature_tasks=max_coverate_featuresets[k - 1], device=device)

    @classmethod
    def set_model_dir(model_dir):
        cls.model_dir = model_dir

    
class VisualPriorRepresentation(object):
    '''
        Handles loading networks that transform images into encoded features.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, 8, 16, 16)
    '''
    feature_task_to_net = {}

    @classmethod
    def _load_unloaded_nets(cls, feature_tasks, model_dir=None):
        net_paths_to_load = []
        feature_tasks_to_load = []
        for feature_task in feature_tasks:
            if feature_task not in cls.feature_task_to_net:
                net_paths_to_load.append(TASKONOMY_PRETRAINED_URLS[feature_task + '_encoder'])
                feature_tasks_to_load.append(feature_task)
        nets = cls._load_networks(net_paths_to_load)
        for feature_task, net in zip(feature_tasks_to_load, nets):
            cls.feature_task_to_net[feature_task] = net

    @classmethod
    def _load_networks(cls, network_paths, model_dir=None):
        return [cls._load_encoder(url, model_dir) for url in network_paths]

    @classmethod
    def _load_encoder(cls, url, model_dir=None, progress=True):
        net = TaskonomyEncoder() #.cuda()
        net.eval()
        checkpoint = torch.utils.model_zoo.load_url(url, model_dir=model_dir, progress=progress)
        net.load_state_dict(checkpoint['state_dict'])
        for p in net.parameters():
            p.requires_grad = False
        # net = Compose(nn.GroupNorm(32, 32, affine=False), net)
        return net
             


class VisualPriorPredictedLabel(object):
    '''
        Handles loading networks that transform images into transformed images.
        Expects inputs:
            shape  (batch_size, 3, 256, 256)
            values [-1,1]
        Outputs:
            shape  (batch_size, C, 256, 256)
            values [-1,1]
            
        This class is technically unsupported and there are absolutely no guarantees. 
    '''
    feature_task_to_net = {}
    
    @classmethod
    def _load_unloaded_nets(cls, feature_tasks, model_dir=None):
        net_paths_to_load = []
        feature_tasks_to_load = []
        for feature_task in feature_tasks:
            if feature_task not in cls.feature_task_to_net:
                if feature_task not in TASKS_TO_CHANNELS:
                    raise NotImplementedError('Task {} not implemented in VisualPriorPredictedLabel'.format(feature_task))
                net_paths_to_load.append((TASKS_TO_CHANNELS[feature_task],
                                          TASKONOMY_PRETRAINED_URLS[feature_task + '_encoder'],
                                          TASKONOMY_PRETRAINED_URLS[feature_task + '_decoder']))
                feature_tasks_to_load.append(feature_task)
        nets = cls._load_networks(net_paths_to_load)
        for feature_task, net in zip(feature_tasks_to_load, nets):
            cls.feature_task_to_net[feature_task] = net

    @classmethod
    def _load_networks(cls, network_paths, model_dir=None, progress=True):
        nets = []
        for out_channels, encoder_path, decoder_path in network_paths:
            nets.append(TaskonomyNetwork(
                    out_channels=out_channels,
                    load_encoder_path=encoder_path,
                    load_decoder_path=decoder_path,
                    model_dir=model_dir,
                    progress=progress))
        return nets