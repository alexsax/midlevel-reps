import numpy as np
from   skimage.transform import resize
import skimage
import torchvision.utils as tvutils
import torch


def rescale_for_display( batch, rescale=True, normalize=False ):
    '''
        Prepares network output for display by optionally rescaling from [-1,1],
        and by setting some pixels to the min/max of 0/1. This prevents matplotlib
        from rescaling the images. 
    '''
    if rescale:
        display_batch = [ rescale_image( im.copy(), new_scale=[0, 1], current_scale=[-1, 1] ) 
                         for im in batch ]
    else:
        display_batch = batch.copy()
    if not normalize:
        for im in display_batch:
            im[0,0,0] = 1.0  # Adjust some values so that matplotlib doesn't rescale
            im[0,1,0] = 0.0  # Now adjust the min
    return display_batch



def rescale_image(im, new_scale=[-1.,1.], current_scale=None, no_clip=False):
    """
    Rescales an image pixel values to target_scale
    
    Args:
        img: A np.float_32 array, assumed between [0,1]
        new_scale: [min,max] 
        current_scale: If not supplied, it is assumed to be in:
            [0, 1]: if dtype=float
            [0, 2^16]: if dtype=uint
            [0, 255]: if dtype=ubyte
    Returns:
        rescaled_image
    """
    # im = im.astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= (max_val - min_val) 
    min_val, max_val = new_scale
    im *= (max_val - min_val)
    im += min_val
    im = skimage.img_as_float(im)

    return im 


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    By kchen @ https://github.com/kchen92/joint-representation/blob/24b30ca6963d2ec99618af379c1e05e1f7026710/lib/data/input_pipeline_feed_dict.py
    """
    if type(im) == PIL.PngImagePlugin.PngImageFile:
        interps = [PIL.Image.NEAREST, PIL.Image.BILINEAR]
        return skimage.util.img_as_float(im.resize(new_dims, interps[interp_order]))
        
    if all( new_dims[i] == im.shape[i] for i in range( len( new_dims ) ) ):
        resized_im = im #return im.astype(np.float32)
    elif im.shape[-1] == 1 or im.shape[-1] == 3:
        #     # skimage is fast but only understands {1,3} channel images
        resized_im = resize(im, new_dims, order=interp_order, preserve_range=True)
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    # resized_im = resized_im.astype(np.float32)
    return resized_im

def resize_rescale_image(img, new_dims, new_scale, interp_order=1, current_scale=None, no_clip=False):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float( img )
    img = resize_image( img, new_dims, interp_order )
    img = rescale_image( img, new_scale, current_scale=current_scale, no_clip=no_clip )

    return img



def pack_images(x, prediction, label, mask=None):
    uncertainty = None
    if isinstance(prediction, tuple):
        prediction, uncertainty = prediction

    if len(label.shape) == 4 and label.shape[1] == 2:
        zeros = torch.zeros(label.shape[0], 1, label.shape[2], label.shape[3]).to(label.device)
        label = torch.cat([label, zeros], dim=1)
        prediction = torch.cat([prediction, zeros], dim=1)
        if uncertainty is not None:
            uncertainty = torch.cat([uncertainty, zeros], dim=1)
        if mask is not None:
            mask = torch.cat([mask, mask[:,0].unsqueeze(1)], dim=1)

    if len(x.shape) == 4 and x.shape[1] == 2:
        zeros = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        x = torch.cat([x, zeros], dim=1)
    to_cat = []
    
    if x.shape[1] <= 3:
        to_cat.append(x)
    shape_with_three_channels = list(x.shape)
    shape_with_three_channels[1] = 3
    to_cat.append(prediction.expand(shape_with_three_channels))
    if uncertainty is not None:
        print(uncertainty.min(), uncertainty.max())
        uncertainty = 2*uncertainty - 1.0
        uncertainty = uncertainty.clamp(min=-1.0, max=1.0)
        to_cat.append(uncertainty.expand(shape_with_three_channels))
    to_cat.append(label.expand(shape_with_three_channels))
    if mask is not None:
        to_cat.append(mask.expand(shape_with_three_channels))
#     print([p.shape for p in to_cat])
    im_samples = torch.cat(to_cat, dim=3)
    im_samples = tvutils.make_grid(im_samples.detach().cpu(), nrow=1, padding=2)
    return im_samples


def maybe_entriple(x, is_mask=False):
    if x.shape[1] == 2:
        if is_mask:
            x = torch.cat([x, x[:,0].unsqueeze(1)], dim=1)
        else:
            zeros = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
            x = torch.cat([x, zeros], dim=1)
    shape_with_three_channels = list(x.shape)
    shape_with_three_channels[1] = 3
    return x.expand(shape_with_three_channels)

def pack_chained_images(x, predictions, labels, mask=None):
    x = maybe_entriple(x)
    if mask is not None:
        mask = maybe_entriple(mask, is_mask=True)
    tripled_predictions, uncertainties = [], []
    for p in predictions:
        if isinstance(p, tuple):
            p, u = p
            uncertainties.append(maybe_entriple(u))
        else:
            uncertainties.append(None)
        tripled_predictions.append(maybe_entriple(p))
    predictions = tripled_predictions
    labels = [maybe_entriple(l) for l in labels]

    to_cat = []
    if x.shape[1] <= 3:
        to_cat.append(x)
    for pred, uncert, label in zip(predictions, uncertainties, labels):
        to_cat.append(label)
        to_cat.append(pred)
        if uncert is not None:
            print(uncert.min(), uncert.max())
            uncert = 2*uncert - 1.0
            uncert = uncert.clamp(min=-1.0, max=1.0)
            to_cat.append(uncert)
    if mask is not None:
        to_cat.append(mask)
#     print([p.shape for p in to_cat])
    im_samples = torch.cat(to_cat, dim=3)
    im_samples = tvutils.make_grid(im_samples.detach().cpu(), nrow=1, padding=2)
    return im_samples