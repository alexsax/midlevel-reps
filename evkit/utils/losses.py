from feature_selector.models.taskonomydecoder import TaskonomyDecoder
from feature_selector.utils import SINGLE_IMAGE_TASKS, TASKS_TO_CHANNELS, FEED_FORWARD_TASKS
import torch
import torch.nn.functional as F

def heteroscedastic_normal(mean_and_scales, target, weight=None, eps=1e-2):
    mu, scales = mean_and_scales
    loss = (mu - target)**2 / (scales**2 + eps) + torch.log(scales**2 + eps)
#     return torch.sum(weight * loss) / torch.sum(weight) if weight is not None else loss.mean() 
    return torch.mean(weight * loss) / weight.mean() if weight is not None else loss.mean() 

def heteroscedastic_double_exponential(mean_and_scales, target, weight=None, eps=5e-2):
    mu, scales = mean_and_scales
    loss = torch.abs(mu - target) / (scales + eps) + torch.log(2.0 * (scales + eps))
    return torch.mean(weight * loss) / weight.mean() if weight is not None else loss.mean() 

def weighted_mse_loss(inputs, target, weight=None):
    if weight is not None:
#         sq = (inputs - target) ** 2
#         weightsq = torch.sum(weight * sq)
        return torch.mean(weight * (inputs - target) ** 2)/torch.mean(weight)
    else:
        return F.mse_loss(inputs, target)

def weighted_l1_loss(inputs, target, weight=None):
    if weight is not None:
        return torch.mean(weight * torch.abs(inputs - target))/torch.mean(weight)
    return F.l1_loss(inputs, target)

def perceptual_l1_loss(decoder_path, bake_decodings):
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    print(f'Loaded decoder from {decoder_path} for perceptual loss')
    def runner(inputs, target, weight=None, cache={}):
        # the last arguments are so we can 'cache' and pass the decodings outside
        inputs_decoded = decoder(inputs)
        targets_decoded = target if bake_decodings else decoder(target)
        cache['inputs_decoded'] = inputs_decoded
        cache['targets_decoded'] = targets_decoded

        if weight is not None:
            return torch.mean(weight * torch.abs(inputs_decoded - targets_decoded))/torch.mean(weight)
        return F.l1_loss(inputs_decoded, targets_decoded)
    return runner


def perceptual_cross_entropy_loss(decoder_path, bake_decodings):
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    print(f'Loaded decoder from {decoder_path} for perceptual loss')
    def runner(inputs, target, weight=None, cache={}):
        # the last arguments are so we can 'cache' and pass the decodings outside
        inputs_decoded = decoder(inputs)
        targets_decoded = target if bake_decodings else decoder(target)
        cache['inputs_decoded'] = inputs_decoded
        cache['targets_decoded'] = targets_decoded

        batch_size, _ = targets_decoded.shape
        return -1. * torch.sum(torch.softmax(targets_decoded, dim=1) * F.log_softmax(inputs_decoded, dim=1)) / batch_size
    return runner
