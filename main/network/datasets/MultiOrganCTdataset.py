import torch
from torch.utils.data import Dataset
import numpy as np
from functools import partial
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler
try:
    from datasets.MultiOrganCTdataset import *
    from datasets.dataprocess.transforms import *
    from datasets.dataprocess.read_CTdara import read_and_process_CTdara
except:
    from .MultiOrganCTdataset import *
    from .dataprocess.transforms import *
    from .dataprocess.read_CTdara import read_and_process_CTdara
import imageio

def save_middle_slice(tensor, index, organ_name, savepath):
    # Check if the tensor is a PyTorch tensor and move it to the CPU if necessary
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.cpu().numpy()
    else:
        # If it's already a NumPy array, use it directly
        tensor_np = tensor

    # Get the middle index along the third dimension (depth)
    middle_idx = tensor_np.shape[1] // 2
    print(middle_idx,tensor_np.shape)
    # Select the middle slice and squeeze to remove extra dimensions
    middle_slice = tensor_np[0, middle_idx, :, :].squeeze()

    # Normalize the slice to be in the range [0, 255] for PNG format
    middle_slice = ((middle_slice - middle_slice.min()) * (255 / (middle_slice.max() - middle_slice.min()))).astype('uint8')

    # Save the 2D slice as a PNG file
    imageio.imwrite(f'{savepath}/{organ_name}_slice_{index}.png', middle_slice)

def process_file(file_path):
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  
                parts = line.strip().split(' ', 2) 
                if len(parts) == 3:
                    result.append(parts)
    return np.array(result)
class MultiOrganCTdataset(Dataset):
    def __init__(self, args, dataset_type='train',is_training=True):
        self.args = args
        self.is_training = is_training
        self.dataset_type = dataset_type 
        self.target_shapes = args.target_shapes 
        
        img_list = []
        lab_list = []
        txt_list = []
        organ_list = args.organ_list
        
        if dataset_type == 'train':
            # anno = np.loadtxt(args.train_anno_file, dtype=np.str_)
            anno = process_file(args.train_anno_file)
            # import pdb; pdb.set_trace()
        elif dataset_type == 'valid':
            anno = process_file(args.val_anno_file)
            # anno = np.loadtxt(args.val_anno_file, dtype=np.str_)
        else: 
            anno = process_file(args.test_anno_file)
            # anno = np.loadtxt(args.test_anno_file, dtype=np.str_)

        for item in anno:
            organ_img_list = []
            for organ in organ_list:
                organ_img_list.append(f'{args.data_dir}/{organ}/{item[0]}.nii.gz')
            img_list.append(organ_img_list)
            # import pdb; pdb.set_trace()
            lab_list.append(int(item[1]))
            # if self.is_training:
            txt_list.append(item[2])

        self.img_list = img_list
        self.lab_list = lab_list
        # if self.is_training:
        self.txt_list = txt_list
    
    def __getitem__(self, index):
        args = self.args
        organ_imgs = []
        for organ, img_path in zip(args.organ_list, self.img_list[index]):
            target_shape = self.target_shapes[organ]

            processed_img = read_and_process_CTdara(img_path, target_shape)
            organ_imgs.append(processed_img)
        
        # esophagus_img, liver_img, spleen_img = organ_imgs  ########   clip
        esophagus_img, liver_img, spleen_img, full_img = organ_imgs
        

        if self.is_training:
            esophagus_img = self.transforms(esophagus_img, args.train_transform_list)
            liver_img = self.transforms(liver_img, args.train_transform_list)
            spleen_img = self.transforms(spleen_img, args.train_transform_list)
            full_img = self.transforms(full_img, args.train_transform_list) ######## clip
            

        # savepath = '/mnt/workspace/ESC/PNG'  # Define the path where you want to save the images
        # save_middle_slice(esophagus_img, index, 'esophagus', savepath)
        # save_middle_slice(liver_img, index, 'liver', savepath)
        # save_middle_slice(spleen_img, index, 'spleen', savepath)
        # print(esophagus_img.shape, liver_img.shape, spleen_img.shape)

        label = self.lab_list[index]
        
        # if self.is_training:
        txt = self.txt_list[index]
        return esophagus_img, liver_img, spleen_img,full_img, label, txt  ###### clip
        # else:
            # return esophagus_img, liver_img, spleen_img,full_img, label
    def transforms(self, mp_image, transform_list):
        args = self.args
        # if 'channel_cutout' in transform_list:
        #     mp_image = random_channel_cutout(mp_image, cutout_num=args.cutcnum, p=args.cutcprob, mode=args.cutcmode)
        # if 'center_crop' in transform_list:
        #     mp_image = center_crop(mp_image, args.crop_size)
        # if 'random_crop' in transform_list:
        #     mp_image = random_crop(mp_image, args.crop_size)
        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)
        if 'random_intensity' in transform_list:
            mp_image = random_intensity(mp_image, 0.1, p=0.25)
        return mp_image

        

    def __len__(self):
        return len(self.img_list)





def create_loader(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
):

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader
