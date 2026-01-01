import argparse
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging
from .models.M3D.Uniformer.MultiOrganBase import MultiOrganBase
from .models.M3D.Uniformer.MultiOrganInter import MultiOrganInter
from .models.M3D.Uniformer.MultiOrganCross1 import MultiOrganCross1 
from .models.M3D.Uniformer.MultiOrganCross2 import MultiOrganCross2
from .models.M3D.Uniformer.MultiOrganCross3 import MultiOrganCross3
from .datasets.MultiOrganCTdataset import MultiOrganCTdataset, create_loader
from .loss_func.regression_loss import ordinal_regression, prediction2label
from .metrics.metrics_ordin import compute_metrics

def parse_file(file_path):
    """Parse annotation file with space-separated columns"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 2)
            if len(parts) != 3:
                print(f"Skipping malformed line: {line}")
                continue
            data.append(parts)
    return data

def create_datasets_and_loaders(args, dataset_type, is_training):
    """Create dataset and data loader"""
    dataset = MultiOrganCTdataset(args, dataset_type=dataset_type, is_training=is_training)
    loader = create_loader(dataset, 
                         batch_size=args.batch_size,
                         is_training=is_training,
                         num_workers=args.workers, 
                         pin_memory=True)
    return loader

def get_model(args):
    """Create model based on args"""
    model_map = {
        'Base': MultiOrganBase,
        'Inter': MultiOrganInter,
        'Cross1': MultiOrganCross1,
        'Cross2': MultiOrganCross2, 
        'Cross3': MultiOrganCross3
    }
    
    model_class = model_map.get(args.model_use)
    if not model_class:
        raise ValueError(f"Invalid model_use: {args.model_use}")
        
    return model_class(args, pretrained_paths=None)

def testdata(args, device):
    """Test data and collect predictions"""
    # Initialize model
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create test loader
    test_loader = create_datasets_and_loaders(args, dataset_type=args.dataset_type, is_training=False)
    
    # Initialize prediction storage
    all_cls = []
    all_score = []
    all_labels = []
    
    # Test loop
    progress_bar = tqdm(test_loader, desc='inferring')
    with torch.no_grad():
        for batch in progress_bar:
            # Unpack batch
            tensor_elements = batch[:-1]
            non_tensor_elements = batch[-1:]
            
            # Process tensors
            tensor_elements = [x.to(device).float() for x in tensor_elements]
            esophagus_img, liver_img, spleen_img, full_img, labels = tensor_elements
            txt = non_tensor_elements[0] if non_tensor_elements else None
            
            # Forward pass
            _, _, _, outputs = model(esophagus_img, liver_img, spleen_img, full_img, txt)
            
            # Get predictions
            output_cls = prediction2label(outputs).cpu()
            
            # Store results
            all_cls.extend(output_cls.numpy())
            all_score.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_score, all_cls, all_labels

def write_result2csv(score_info, cls_info, lab_info, args):
    """Write predictions to CSV"""
    # Load annotations
    anno_files = {
        'train': args.train_anno_file,
        'valid': args.val_anno_file,
        'test': args.test_anno_file
    }
    anno_info = np.array(parse_file(anno_files[args.dataset_type]))
    
    # Create dataframe
    df = pd.DataFrame({
        'ID': anno_info[:, 0],
        'True_Label': lab_info,
        'Predicted_Label': cls_info,
        'Score_Class_0': [score[0] for score in score_info],
        'Score_Class_1': [score[1] for score in score_info],
        'Score_Class_2': [score[2] for score in score_info],
        'Score_Class_3': [score[3] for score in score_info]
    })
    
    # Save results
    csv_path = os.path.join(args.results_dir, f"{args.dataset_type}.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info(f"Predictions saved to '{csv_path}'")

def run_infer_main(args):
    """Main inference function"""
    logging.info("Starting inference...")
    
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run inference
    score_dat, cls_dat, lab_dat = testdata(args, device)
    write_result2csv(score_dat, cls_dat, lab_dat, args)
    
    logging.info("Inference completed!")
