import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


task_mapping = {
 'autoencoder': 'autoencoding',
 'colorization': 'colorization',
 'curvature': 'curvature',
 'denoise': 'denoising',
 'edge2d':'edge_texture',
 'edge3d': 'edge_occlusion',
 'ego_motion': 'egomotion', 
 'fix_pose': 'fixated_pose', 
 'jigsaw': 'jigsaw',
 'keypoint2d': 'keypoints2d',
 'keypoint3d': 'keypoints3d',
 'non_fixated_pose': 'nonfixated_pose',
 'point_match': 'point_matching', 
 'reshade': 'reshading',
 'rgb2depth': 'depth_zbuffer',
 'rgb2mist': 'depth_euclidean',
 'rgb2sfnorm': 'normal',
 'room_layout': 'room_layout',
 'segment25d': 'segment_unsup25d',
 'segment2d': 'segment_unsup2d',
 'segmentsemantic': 'segment_semantic',
 'class_1000': 'class_object',
 'class_places': 'class_scene',
 'inpainting_whole': 'inpainting',
 'vanishing_point': 'vanishing_point'
}


CHANNELS_TO_TASKS = {
    1: ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', ],
    2: ['curvature', 'principal_curvature'],
    3: ['autoencoding', 'denoising', 'normal', 'inpainting', 'rgb', 'normals'],
    128: ['segment_unsup2d', 'segment_unsup25d'],
    1000: ['class_object'],
    None: ['segment_semantic']
}

PIX_TO_PIX_TASKS = ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', 'curvature', 'autoencoding', 'denoising', 'normal', 'inpainting', 'segment_unsup2d', 'segment_unsup25d', 'segment_semantic', ]
FEED_FORWARD_TASKS = ['class_object', 'class_scene', 'room_layout', 'vanishing_point']
SINGLE_IMAGE_TASKS = PIX_TO_PIX_TASKS + FEED_FORWARD_TASKS
SIAMESE_TASKS = ['fix_pose', 'jigsaw', 'ego_motion', 'point_match', 'non_fixated_pose']


TASKS_TO_CHANNELS = {}
for n, tasks in CHANNELS_TO_TASKS.items():
    for task in tasks:
        TASKS_TO_CHANNELS[task] = n

LIST_OF_OLD_TASKS = sorted(list(task_mapping.keys()))
LIST_OF_TASKS = sorted(list(task_mapping.values()))


class TaskonomyNetwork(nn.Module):
    
    def __init__(self, out_channels=3, eval_only=True, load_encoder_path=None, load_decoder_path=None):
        ''' 
            out_channels = None for decoder only
        '''
        super().__init__()
        self.encoder = TaskonomyEncoder(eval_only=True)
        self.encoder.normalize_outputs = False

        self.decoder = None
        if out_channels is not None:
            self.decoder = TaskonomyDecoder(out_channels=out_channels, eval_only=True)
        
        if load_encoder_path is not None:
            self.load_encoder(load_encoder_path)
        
        if load_decoder_path is not None:
            self.load_decoder(load_decoder_path)


    def load_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return self.encoder.load_state_dict(checkpoint['state_dict'])

    def load_decoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return self.decoder.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        y = self.encoder(x)
        return self.decoder(y)
        

class Scissor(torch.nn.Module):
    # Remove the first row and column of our data
    # To deal with asymmetry in ConvTranpose layers
    # if used correctly, this removes 0's
    def forward(self, x):
        _, _, h, _ = x.shape
        x = x[:,:,1:h,1:h]
        return x

class TaskonomyDecoder(nn.Module):
    """
    Note regarding DeConvolution Layer:
    - TF uses padding = 'same': `o = i * stride` (e.g. 128 -> 64 if stride = 2)
    - Using the equation relating output_size, input_size, stride, padding, kernel_size, we get 2p = 1
    - See https://stackoverflow.com/questions/50683039/conv2d-transpose-output-shape-using-formula
    - This means we need to add asymmetric padding of (1,0,1,0) prior to deconv
    - PyTorch ConvTranspose2d does not support asymmetric padding, so we need to pad ourselves
    - But since we pad ourselves it goes into the input size and since stride = 2, we get an extra row/column of zeros
    - e.g. This is because it is putting a row/col between each row/col of the input (our padding is treated as input)
    - That's fine, if we remove that row and column, we get the proper outputs we are looking for
    - See https://github.com/vdumoulin/conv_arithmetic to visualize deconvs
    """

    def __init__(self, out_channels=3, eval_only=True):
        super(TaskonomyDecoder, self).__init__()
        self.conv2 = self._make_layer(8, 1024)
        self.conv3 = self._make_layer(1024, 1024)
        self.conv4 = self._make_layer(1024, 512)
        self.conv5 = self._make_layer(512, 256)
        self.conv6 = self._make_layer(256, 256)
        self.conv7 = self._make_layer(256, 128)

        self.deconv8 = self._make_layer(128, 64, stride=2, deconv=True)
        self.conv9 = self._make_layer(64, 64)

        self.deconv10 = self._make_layer(64, 32, stride=2, deconv=True)
        self.conv11 = self._make_layer(32, 32)

        self.deconv12 = self._make_layer(32, 16, stride=2, deconv=True)
        self.conv13 = self._make_layer(16, 32)

        self.deconv14 = self._make_layer(32, 16, stride=2, deconv=True)
        self.decoder_output = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, bias=True, padding=1),
            nn.Tanh()
        )

        self.eval_only = eval_only
        if self.eval_only:
            self.eval()

        for p in self.parameters():
            p.requires_grad = False

    def _make_layer(self, in_channels, out_channels, stride=1, deconv=False):
        if deconv:
            pad = nn.ZeroPad2d((1,0,1,0))  # Pad first row and column
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=0, bias=False)
            scissor = Scissor()  # Remove first row and column
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # pad = 'SAME'

        bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=True)
        lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        if deconv:
            layer = nn.Sequential(pad, conv, scissor, bn, lrelu)
        else:
            layer = nn.Sequential(conv, bn, lrelu)
        return layer

    def forward(self, x):
        # Input x: N x 256 x 256 x 3
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.deconv8(x)
        x = self.conv9(x)

        x = self.deconv10(x)
        x = self.conv11(x)

        x = self.deconv12(x)
        x = self.conv13(x)

        x = self.deconv14(x)
        x = self.decoder_output(x)
        # add gaussian-noise?
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False, padding=1)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = F.pad(out, pad=(1,1,1,1), mode='constant', value=0)  # other modes are reflect, replicate
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TaskonomyEncoder(nn.Module):

    def __init__(self, normalize_outputs=True, eval_only=True, train_penultimate=False, train=False):
        self.inplanes = 64
        super(TaskonomyEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        block = Bottleneck
        layers = [3,4,6,3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.compress1 = nn.Conv2d(2048, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.compress_bn = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.groupnorm = nn.GroupNorm(8, 8, affine=False)
        self.normalize_outputs = normalize_outputs
        self.eval_only = eval_only
        if self.eval_only:
            self.eval()
        for p in self.parameters():
            p.requires_grad = False

        if train_penultimate:
            for name, param in self.named_parameters():
                if 'compress' in name:  # last layers: compress1.weight, compress_bn.weight, compress_bn.bias
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if train:
            for p in self.parameters():
                p.requires_grad = True


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []

        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers.append(block(self.inplanes, planes, downsample=downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))

        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.MaxPool2d( kernel_size=1, stride=stride ),
            )
        layers.append(block(self.inplanes, planes, stride, downsample))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, pad=(3,3,3,3), mode='constant', value=0)
        #  other modes are reflect, replicate, constant

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = F.pad(x, (0,1,0,1), 'constant', 0)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.compress1(x)
        x = self.compress_bn(x)
        x = self.relu1(x)
        if self.normalize_outputs:
            x = self.groupnorm(x)
        return x

    def train(self, val):
        if val and self.eval_only:
            warnings.warn("Ignoring 'train()' in TaskonomyEncoder since 'eval_only' was set during initialization.", RuntimeWarning)
        else:
            return super().train(val)
