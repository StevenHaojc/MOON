from typing import List, Tuple, Union
import SimpleITK as sitk
import numpy as np
from timm.models.layers import to_3tuple
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union


def resample_base(
    sitk_im: sitk.Image,
    origin: Union[List, Tuple, np.ndarray],
    direction: Union[List, Tuple],
    spacing: Union[List, Tuple, np.ndarray],
    size: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    the base resample function, can be used to resample a small patch out of the original image
    or to resample to sample patch back to the original image, and of course, to resize a volume

    :param sitk_im: input image
    :param origin: the origin of the resampled volume
    :param direction: the direction of the resampled volume
    :param spacing: the spacing of the resampled volume
    :param size: the output size of the resampled volume
    :param interpolator: interpolation method, can be 'nearest' and linear, defaults to 'nearest'
    :param pad_value: value for voxels extroplated
    :return: the resampled SimpleITK image object
    """
    size = [int(s) for s in size]
    SITK_INTERPOLATOR_DICT = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "gaussian": sitk.sitkGaussian,
        "label_gaussian": sitk.sitkLabelGaussian,
        "bspline": sitk.sitkBSpline,
        "hamming_sinc": sitk.sitkHammingWindowedSinc,
        "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
        "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
        "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
    }
    if float(sitk.__version__[:3]) >= 2.2:
        SITK_INTERPOLATOR_DICT.update(
            {
                "bspline1": sitk.sitkBSpline1,
                "bspline2": sitk.sitkBSpline2,
                "bspline3": sitk.sitkBSpline3,
                "bspline4": sitk.sitkBSpline4,
                "bspline5": sitk.sitkBSpline5,
            }
        )

    assert (
        interpolator in SITK_INTERPOLATOR_DICT.keys()
    ), "`interpolator` should be one of {}".format(SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(size)
    resample_filter.SetOutputSpacing(np.array(spacing).tolist())
    resample_filter.SetOutputOrigin(np.array(origin).tolist())
    resample_filter.SetOutputDirection(direction)
    resample_filter.SetOutputPixelType(sitk_im.GetPixelID())
    resample_filter.SetDefaultPixelValue(pad_value)
    resample_filter.SetInterpolator(sitk_interpolator)

    # the simpleitk resample_image_filter works like align_corners=False in pytorch
    # and will result in padded slices at the end of the output image. to avoid this,
    # while keep the correct output origin, we need to do something
    in_direction = sitk_im.GetDirection()
    out_direction = np.array(direction).reshape(-1)
    if np.allclose(out_direction, in_direction, atol=1e-2) and np.isclose(
        np.abs(out_direction).sum(), 3, atol=1e-2
    ):
        in_spacing = np.array(sitk_im.GetSpacing())
        out_spacing = np.array(spacing)
        tmp_origin = np.array(origin) - (in_spacing - out_spacing) / 2
        resample_filter.SetOutputOrigin(tmp_origin.tolist())

    img = resample_filter.Execute(sitk_im)
    img.SetOrigin(np.array(origin).tolist())

    return img

def resample_itkimage_withsize(
    itkimage: sitk.Image,
    new_size: Union[List, Tuple, np.ndarray],
    interpolator: str = "nearest",
    pad_value: Union[int, float] = -1024,
) -> sitk.Image:
    """
    Image resize with size by sitk resampleImageFilter.

    :param itkimage: input itk image or itk volume.
    :param new_size: the target size of the resampled image, such as [120, 80, 80].
    :param interpolator: for mask used nearest, for image linear is an option.
    :param pad_value: the value for the pixel which is out of image.
    :return: resampled itk image.
    """

    # get resize factor
    origin_size = np.array(itkimage.GetSize())

    new_size = np.array(new_size)

    factor = origin_size / new_size

    # get new spacing
    origin_spcaing = itkimage.GetSpacing()
    new_spacing = factor * origin_spcaing

    itkimg_resampled = resample_base(
        itkimage,
        itkimage.GetOrigin(),
        itkimage.GetDirection(),
        new_spacing,
        new_size,
        interpolator,
        pad_value,
    )

    return itkimg_resampled


def normalized_apply_window(img_array):
    window_center = 50  
    window_width = 350 

    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    windowed_img = np.clip(img_array, lower_bound, upper_bound)
    normalized_img = (windowed_img - np.min(windowed_img)) / (np.max(windowed_img) - np.min(windowed_img))

    return normalized_img


def read_and_process_CTdara(nii_path,target_shape):

    
    itk_image = sitk.ReadImage(nii_path)
    # image_array = sitk.GetArrayFromImage(itk_image)
    # print(image_array.shape)
    itk_resampled_image = resample_itkimage_withsize(itk_image, target_shape, "linear", pad_value=-1024)
    np_resampled_image = sitk.GetArrayFromImage(itk_resampled_image)
    np_normalized_image = normalized_apply_window(np_resampled_image)
    np_normalized_image = np.expand_dims(np_normalized_image, axis=0)
    # print(np_normalized_image.shape)
    return np_normalized_image



if __name__ == '__main__':
    nii_path = '/mnt/workspace/ESC/Data/esophagus/V0_0013_0000.nii.gz'
    target_shape = (36,60,90)
    process_img = read_and_process_CTdara(nii_path,target_shape)

    min_val = np.amin(process_img)
    max_val = np.amax(process_img)

    print(f"Pixel value range: {min_val} to {max_val}")