import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import shutil
# import cv2

def load_nifti_itk(file_name):

    reader = sitk.ImageFileReader()
    reader.SetFileName(file_name)
    image = reader.Execute()

    spacing = np.array(list(image.GetSpacing()))

    origin = image.GetOrigin()
    direction = image.GetDirection()

    nda = sitk.GetArrayFromImage(image)
    img_array = nda.transpose((2, 1, 0))[::-1, ::-1, :]

    return img_array, [spacing, origin, direction]

def convert_num_to_nii(image, info):
    spacing, origin, direction = info
    change_image = image[::-1, ::-1, :]
    image_array = np.swapaxes(change_image, 2, 0)
    imgitk = sitk.GetImageFromArray(image_array)
    imgitk.SetSpacing((spacing[0], spacing[1], spacing[2]))
    imgitk.SetOrigin(origin)
    imgitk.SetDirection(direction)
    return imgitk


def mask_merge(mask):
    masknew = np.zeros_like(mask)
    masknew[mask >0] = 1
    return masknew
def mask_convert(mask_path):
    mask, info = load_nifti_itk(mask_path)
    mask = mask_merge(mask)
    S_noz_index = np.nonzero(mask)
    S_nonz_index_R = np.unique(S_noz_index[2])
    z_min_id = min(S_nonz_index_R)
    z_max_id = max(S_nonz_index_R)
    matrix = np.zeros_like(mask)
    for i in S_nonz_index_R:
        DI = abs(i - z_min_id)*info[0][2]/10
        matrix[:,:,i]=np.array(mask[:,:,i])
        
        if DI <= 5:
            matrix[:,:,i][matrix[:,:,i]>0] = 2
        else:
            matrix[:,:,i][matrix[:,:,i]>0] = 1
    return  matrix,info


imgp =  '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/data/esophagus'
mask_path = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/mask/esophagus'
savep = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/mask/lesion5cm'

for i in os.listdir(mask_path):
    inp = os.path.join(mask_path,i)
    save_path_mask = os.path.join(savep,i)
    try:
        mask,info = mask_convert(inp)
        save_mask = convert_num_to_nii(mask,info)
        sitk.WriteImage(save_mask, save_path_mask)
        print(i)

    except:
        print('erro',i)


