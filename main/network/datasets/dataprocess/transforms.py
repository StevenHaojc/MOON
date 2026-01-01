import random
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy import ndimage
from timm.models.layers import to_3tuple


def load_nii_file(nii_image):
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def random_intensity(image, factor, p=0.5):
    shift = scale = factor
    assert (shift >0) and (scale >0)
    if random.random() > p:
        return image
    shift_factor = np.random.uniform(shift, shift, size=[image.shape[0],1,1,1]).astype('float32') # [-0.1,+0.1]
    scale_factor = np.random.uniform(1.0 - scale, 1.0 + scale, size=[image.shape[0],1,1,1]).astype('float32') # [0.9,1.1)
    return image * scale_factor + shift_factor
    
def random_crop(image, crop_shape):
    crop_shape = to_3tuple(crop_shape)
    _, z_shape, y_shape, x_shape = image.shape
    z_min = np.random.randint(0, z_shape - crop_shape[0])
    y_min = np.random.randint(0, y_shape - crop_shape[1])
    x_min = np.random.randint(0, x_shape - crop_shape[2])
    image = image[..., z_min:z_min+crop_shape[0], y_min:y_min+crop_shape[1], x_min:x_min+crop_shape[2]]
    return image

def center_crop(image, target_shape=(10, 80, 80)):
    target_shape = to_3tuple(target_shape)
    b, z_shape, y_shape, x_shape = image.shape
    z_min = z_shape // 2 - target_shape[0] // 2
    y_min = y_shape // 2 - target_shape[1] // 2
    x_min = x_shape // 2 - target_shape[2] // 2
    image = image[:, z_min:z_min+target_shape[0], y_min:y_min+target_shape[1], x_min:x_min+target_shape[2]]
    return image

def randomflip_z(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, ::-1, ...]

def randomflip_x(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[..., ::-1]

def randomflip_y(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, :, ::-1, ...]

def random_flip(image, mode='x', p=0.5):
    if mode == 'x':
        image = randomflip_x(image, p=p)
    elif mode == 'y':
        image = randomflip_y(image, p=p)
    elif mode == 'z':
        image = randomflip_z(image, p=p)
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')
    return image

def rotate(image, angle=10):
    angle = random.randint(-10, 10)
    r_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
    if r_image.shape != image.shape:
        r_image = center_crop(r_image, target_shape=image.shape[1:])
    return r_image

def random_channel_cutout(image, cutout_num=2, mode='zeros', p=0.5):
    if random.random() > p:
        return image
    else:
        c, z_shape, y_shape, x_shape = image.shape
        # cutout_num = random.randint(1, cutout_num) # random cutout_num
        cutout_channels = random.sample(range(c), cutout_num)
        image[cutout_channels, :, :, :] = np.zeros(shape=(z_shape, y_shape, x_shape), dtype=image.dtype)
        if mode != 'zeros':
            if mode == 'avr':
                re_image = image.copy()
                re_image = np.delete(re_image, cutout_channels, axis=0)
                image[cutout_channels, :, :, :] = np.repeat(np.expand_dims(np.mean(re_image, axis=0), axis=0), cutout_num, axis=0)
        return image