from typing import List, Tuple, Union
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F

def resample_base(sitk_im: sitk.Image,
                 origin: Union[List, Tuple, np.ndarray],
                 direction: Union[List, Tuple],
                 spacing: Union[List, Tuple, np.ndarray], 
                 size: Union[List, Tuple, np.ndarray],
                 interpolator: str = "nearest",
                 pad_value: Union[int, float] = -1024) -> sitk.Image:
    
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
        SITK_INTERPOLATOR_DICT.update({
            "bspline1": sitk.sitkBSpline1,
            "bspline2": sitk.sitkBSpline2, 
            "bspline3": sitk.sitkBSpline3,
            "bspline4": sitk.sitkBSpline4,
            "bspline5": sitk.sitkBSpline5,
        })

    assert interpolator in SITK_INTERPOLATOR_DICT, f"Interpolator must be one of {list(SITK_INTERPOLATOR_DICT.keys())}"
    sitk_interpolator = SITK_INTERPOLATOR_DICT[interpolator]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(size)
    resample_filter.SetOutputSpacing(np.array(spacing).tolist())
    resample_filter.SetOutputOrigin(np.array(origin).tolist())
    resample_filter.SetOutputDirection(direction)
    resample_filter.SetOutputPixelType(sitk_im.GetPixelID())
    resample_filter.SetDefaultPixelValue(pad_value)
    resample_filter.SetInterpolator(sitk_interpolator)

    in_direction = sitk_im.GetDirection()
    out_direction = np.array(direction).reshape(-1)
    if np.allclose(out_direction, in_direction, atol=1e-2) and np.isclose(np.abs(out_direction).sum(), 3, atol=1e-2):
        in_spacing = np.array(sitk_im.GetSpacing())
        out_spacing = np.array(spacing)
        tmp_origin = np.array(origin) - (in_spacing - out_spacing) / 2
        resample_filter.SetOutputOrigin(tmp_origin.tolist())

    img = resample_filter.Execute(sitk_im)
    img.SetOrigin(np.array(origin).tolist())

    return img

def resample_itkimage_withsize(itkimage: sitk.Image,
                             new_size: Union[List, Tuple, np.ndarray],
                             interpolator: str = "nearest",
                             pad_value: Union[int, float] = -1024) -> sitk.Image:

    origin_size = np.array(itkimage.GetSize())
    new_size = np.array(new_size)
    factor = origin_size / new_size
    new_spacing = factor * itkimage.GetSpacing()

    return resample_base(itkimage,
                        itkimage.GetOrigin(),
                        itkimage.GetDirection(), 
                        new_spacing,
                        new_size,
                        interpolator,
                        pad_value)

def normalized_apply_window(img_array):
    window_center = 50
    window_width = 350
    
    lower_bound = window_center - window_width / 2 
    upper_bound = window_center + window_width / 2
    
    windowed_img = np.clip(img_array, lower_bound, upper_bound)
    return (windowed_img - windowed_img.min()) / (windowed_img.max() - windowed_img.min() + 1e-8)

def read_and_process_CTdara(nii_path, target_shape):
    itk_image = sitk.ReadImage(nii_path)
    resampled_image = resample_itkimage_withsize(itk_image, target_shape, "linear", -1024)
    np_image = sitk.GetArrayFromImage(resampled_image)
    np_normalized = normalized_apply_window(np_image)
    np_resampled = sitk.GetArrayFromImage(resampled_image)
    return np_normalized[np.newaxis], np_resampled

def prediction2label(pred):
    pred_tran = (pred > 0.5).cumprod(dim=1)
    sums = pred_tran.sum(dim=1)
    return torch.where(sums == 0, torch.zeros_like(sums), sums - 1)

if __name__ == '__main__':
    nii_path = '/mnt/workspace/ESC/Data/esophagus/V0_0013_0000.nii.gz'
    target_shape = (36, 60, 90)
    process_img, resampled_img = read_and_process_CTdara(nii_path, target_shape)
    print(f"Processed image range: {process_img.min():.3f} to {process_img.max():.3f}")
    print(f"Resampled image range: {resampled_img.min():.3f} to {resampled_img.max():.3f}")