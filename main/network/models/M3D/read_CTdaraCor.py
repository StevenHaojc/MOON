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
        "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc
    }

    if float(sitk.__version__[:3]) >= 2.2:
        SITK_INTERPOLATOR_DICT.update({
            "bspline1": sitk.sitkBSpline1,
            "bspline2": sitk.sitkBSpline2,
            "bspline3": sitk.sitkBSpline3, 
            "bspline4": sitk.sitkBSpline4,
            "bspline5": sitk.sitkBSpline5
        })

    assert interpolator in SITK_INTERPOLATOR_DICT, f"Interpolator must be one of {list(SITK_INTERPOLATOR_DICT.keys())}"

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(size)
    resample_filter.SetOutputSpacing(np.array(spacing).tolist())
    resample_filter.SetOutputOrigin(np.array(origin).tolist())
    resample_filter.SetOutputDirection(direction) 
    resample_filter.SetOutputPixelType(sitk_im.GetPixelID())
    resample_filter.SetDefaultPixelValue(pad_value)
    resample_filter.SetInterpolator(SITK_INTERPOLATOR_DICT[interpolator])

    in_direction = sitk_im.GetDirection()
    out_direction = np.array(direction).reshape(-1)
    if np.allclose(out_direction, in_direction, atol=1e-2) and \
       np.isclose(np.abs(out_direction).sum(), 3, atol=1e-2):
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

def normalized_apply_window(img_array: np.ndarray) -> np.ndarray:
    window_center = 50
    window_width = 350
    
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    
    windowed_img = np.clip(img_array, lower_bound, upper_bound)
    return (windowed_img - windowed_img.min()) / (windowed_img.max() - windowed_img.min() + 1e-8)

def resample_image(itk_image: sitk.Image, 
                  new_spacing: List[float] = [1.0, 1.0, 1.0],
                  is_seg: bool = False) -> np.ndarray:
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    new_spacing = np.array(new_spacing)
    
    resample_factor = original_spacing / new_spacing
    new_size = [int(sz) for sz in original_size * resample_factor]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_seg else sitk.sitkLinear)
    
    return sitk.GetArrayFromImage(resampler.Execute(itk_image))

def read_and_process_CTdara(nii_path: str,
                           target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    itk_image = sitk.ReadImage(nii_path)
    
    itk_resampled_image = resample_itkimage_withsize(itk_image, target_shape, "linear", -1024)
    resampled_image = resample_image(itk_image)
    
    np_resampled_image = sitk.GetArrayFromImage(itk_resampled_image)
    
    resampled_image = normalized_apply_window(resampled_image)
    np_normalized_image = normalized_apply_window(np_resampled_image)
    np_normalized_image = np.expand_dims(np_normalized_image, axis=0)
    
    return np_normalized_image, resampled_image

def prediction2label(pred: torch.Tensor) -> torch.Tensor:
    """Convert ordinal predictions to class labels"""
    pred_tran = (pred > 0.5).cumprod(dim=1) 
    sums = pred_tran.sum(dim=1)
    return torch.where(sums == 0, torch.zeros_like(sums), sums - 1)

if __name__ == '__main__':
    nii_path = '/mnt/workspace/ESC/Data/esophagus/V0_0013_0000.nii.gz'
    target_shape = (36, 60, 90)
    process_img, resampled_img = read_and_process_CTdara(nii_path, target_shape)
    print(f"Processed image range: [{process_img.min():.3f}, {process_img.max():.3f}]")
    print(f"Resampled image range: [{resampled_img.min():.3f}, {resampled_img.max():.3f}]")