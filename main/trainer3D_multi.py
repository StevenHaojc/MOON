import torch
import os
import argparse
import json
import sys
from os.path import abspath, join, dirname
import logging

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.insert(0, root_dir)
from network.Utils_MultiOrgan_multi_gpu import run_train_main_multi_gpu
from network.Utils_MultiOrgan_one_gpu import run_train_main_one_gpu

def main(args):
    if len(args.gpus) > 1:
        run_train_main_multi_gpu(args)
    else:
        run_train_main_one_gpu(args)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train neural network model')
    
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in Proto')
    parser.add_argument('--gpus', default='6,7', type=str, help='GPU device IDs to use (e.g., "0,1,2")')
    parser.add_argument('--model_use', type=str, default= 'Base')
    parser.add_argument('--fusion_method', type=str, default= 'concat')

    parser.add_argument('--organ_list', nargs='+', default=['esophagus', 'liver', 'spleen', 'full'])
    parser.add_argument('--target_shapes', type=str, default={'esophagus': (40,40,100),
                                                              'liver': (256, 196, 36),
                                                              'spleen': (152, 196, 24),
                                                              'full': (256, 256, 32)})
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--embed_dims', type=int, default=[64, 128, 320, 512])
    parser.add_argument('--depths', type=int, default=[3, 4, 8, 3])
    parser.add_argument('--typeloss', type=str, default= 'ordinal')
    parser.add_argument('--pretrained_model_path', default='main/network/models/M3D/Uniformer/pretrained_weights/uniformer_base_k400_8x8.pth', type=str)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--optimizer_use', type=str, default= 'AdamW')

    parser.add_argument('--angle', default=45, type=int)
    parser.add_argument('--flip_prob', default=0.5, type=float)
    parser.add_argument('--train_transform_list', type=str, default='["z_flip", "x_flip", "y_flip", "rotation", "random_intensity"]')
    parser.add_argument('--data_dir', type=str, default="/home/lichunli.lcl/LCL/ESCode/Data/")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])

    parser.add_argument("--anno_root", type=str, default="main/txtfile/")
    parser.add_argument("--train_anno_file", type=str, default="train_fold{fold}_size_descp_ratio2.txt")
    parser.add_argument("--val_anno_file", type=str, default="val_fold{fold}_size_descp_ratio2.txt")

    parser.add_argument("--results_root", type=str, default="results/Cls3/ThreeOrgan/")
    parser.add_argument("--output_results", type=str, default="fold{fold}/")
    parser.add_argument("--checkpoint_path", type=str, default="fold{fold}/model_checkpoint.pth")
    parser.add_argument("--early_stop_patience", type=int, default=200)

    args = parser.parse_args()

    fold_str = f"{args.fold}"
    modelname_str = args.model_use
    latefusion_str = args.fusion_method

    args.train_anno_file = os.path.join(args.anno_root, args.train_anno_file.replace("{fold}", fold_str))
    args.val_anno_file = os.path.join(args.anno_root, args.val_anno_file.replace("{fold}", fold_str))

    args.output_results = os.path.join(args.results_root, modelname_str, latefusion_str, args.output_results.replace("{fold}", fold_str))
    args.checkpoint_path = os.path.join(args.results_root, modelname_str, latefusion_str, args.checkpoint_path.replace("{fold}", fold_str))

    args_dict = vars(args)
    json_string = json.dumps(args_dict, indent=4)

    json_file_path = os.path.join(args.output_results, f"{modelname_str}_fold{fold_str}_config.json")
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    with open(json_file_path, 'w') as outfile:
        outfile.write(json_string)

    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    options = parse_arguments()
    main(options)