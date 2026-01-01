import torch
import os
import argparse
import json
import sys
from os.path import abspath, join, dirname
import logging

# Add project root directory to sys.path
root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.insert(0, root_dir)
from network.Utils_MultiInfer import run_infer_main
from network.Utils_MultiInferTSNE import run_infer_main_tsne

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural network inference model')
    
    parser.add_argument('--gpus', default='0,1', type=str, help='GPU device IDs to use (e.g., "0,1,2")')
    parser.add_argument('--model_use', type=str, default= 'base')
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

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--data_dir', type=str, default="../Data")

    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])

    # Path settings for data annotations
    parser.add_argument("--dataset_type", type=str, default="test")
    parser.add_argument("--anno_root", type=str, default="main/txtfile/")
    parser.add_argument("--test_anno_file", type=str, default="test_lab_size_descp_ratio2.txt")
    parser.add_argument("--train_anno_file", type=str, default="train_fold{fold}_size_descp_ratio2.txt")
    parser.add_argument("--val_anno_file", type=str, default="val_fold{fold}_size_descp_ratio2.txt")

    # Model and results paths
    parser.add_argument("--results_root", type=str, default="results/Cls3/ThreeOrgan")
    parser.add_argument("--results_dir", type=str, default="fold{fold}/")
    parser.add_argument("--model_path", type=str, default="best_model_Val_loss.pth")

    args = parser.parse_args()

    # Update paths based on fold number
    fold_str = f"{args.fold}"
    modelname_str = args.model_use
    latefusion_str = args.fusion_method

    args.train_anno_file = os.path.join(args.anno_root, args.train_anno_file.replace("{fold}", fold_str))
    args.val_anno_file = os.path.join(args.anno_root, args.val_anno_file.replace("{fold}", fold_str))
    args.test_anno_file = os.path.join(args.anno_root, args.test_anno_file)
    
    args.results_dir = os.path.join(args.results_root, modelname_str, latefusion_str, args.results_dir.replace("{fold}", fold_str))
    args.model_path = os.path.join(args.results_dir, args.model_path)

    # Save configuration to JSON
    args_dict = vars(args)
    json_string = json.dumps(args_dict, indent=4)
    json_file_path = os.path.join(args.results_dir, f"{modelname_str}_{latefusion_str}_fold{fold_str}_testconfig.json")
    
    with open(json_file_path, 'w') as outfile:
        outfile.write(json_string)

    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    options = parse_arguments()
    run_infer_main(options)