from os.path import join
import SimpleITK as sitk
import numpy as np
import os
import csv
from multiprocessing import Pool, cpu_count

# Your original functions (load_nifti_itk, convert_num_to_nii, crop_liver) go here
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


def merge_lab(lab):
    newlab = np.zeros_like(lab)
    newlab[lab == 2] = 1
    return newlab


def convert_num_to_nii(image, info):
    spacing, origin, direction = info
    change_image = image[::-1, ::-1, :]
    image_array = np.swapaxes(change_image, 2, 0)
    imgitk = sitk.GetImageFromArray(image_array)
    imgitk.SetSpacing((spacing[0], spacing[1], spacing[2]))
    imgitk.SetOrigin(origin)
    imgitk.SetDirection(direction)
    return imgitk

def crop_liver(img,lab):
    x = lab
    tempL = np.nonzero(x)
    pad = [5,5,1]
    bbox = [[max(0, np.min(tempL[0])-pad[0]), min(x.shape[0], np.max(tempL[0])+pad[0])], \
            [max(0, np.min(tempL[1])-pad[1]), min(x.shape[1], np.max(tempL[1])+pad[1])], \
            [max(0, np.min(tempL[2])-pad[2]), min(x.shape[2], np.max(tempL[2])+pad[2])]]
    imgn = img[bbox[0][0]:bbox[0][1],
               bbox[1][0]:bbox[1][1],
               bbox[2][0]:bbox[2][1]]
    labn = lab[bbox[0][0]:bbox[0][1],
               bbox[1][0]:bbox[1][1],
               bbox[2][0]:bbox[2][1]]
    return imgn,labn, bbox

    
def process_file(mf2):
    try:
        mf3 = mf2.split(".")[1]
        mf4 = mf2.split(".")[0]
        if mf3 == 'nii':
            print(mf2, 'exist')
            img_pn = join(img_p, mf4 + '_0000.nii.gz')
            lab_pn = join(lab_p, mf4 + '.nii.gz')
            save_imgpn = join(save_imgp, mf4 + '_0000.nii.gz')
            save_labpn = join(save_labp, mf4 + '.nii.gz')
            imgn1, info = load_nifti_itk(img_pn)
            labn1, _ = load_nifti_itk(lab_pn)
            labn1 = merge_lab(labn1)
            imgn2, labn2, bbox = crop_liver(imgn1, labn1)
            xmin, xmax, ymin, ymax, zmin, zmax = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], bbox[2][0], bbox[2][1]
            save_image = convert_num_to_nii(imgn2, info)
            save_mask = convert_num_to_nii(labn2, info)
            sitk.WriteImage(save_image, save_imgpn)
            # sitk.WriteImage(save_mask, save_labpn)
            return mf4, xmin, xmax, ymin, ymax, zmin, zmax
    except Exception as e:
        print(f"{mf2} error: {e}")
        return None

# Define your paths
# (img_p, lab_p, save_imgp, save_labp, output_csv) go here
img_p = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/data/esophagus'
lab_p = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/mask/lesion5cm'
save_imgp = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/data/esc5cm'
save_labp = lab_p
output_csv = '/mnt/workspace/workgroup/LCL/Test/ESC_NC_Screening/Newdata/mask/output5cm.csv'
if __name__ == '__main__':
    files_to_process = [f for f in os.listdir(lab_p) if f.endswith('.nii.gz')]
    
    # Set up your multiprocessing pool
    pool = Pool(processes=cpu_count())

    # Process files in parallel
    results = pool.map(process_file, files_to_process)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Write results to CSV
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_path', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])  # Write header
        for result in results:
            if result:  # Check if result is not None
                writer.writerow(result)

    print("Processing complete.")
