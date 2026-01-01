import argparse
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .models.M3D.Uniformer.MultiOrganBase import MultiOrganBase
from .models.M3D.Uniformer.MultiOrganInter import MultiOrganInter
from .models.M3D.Uniformer.MultiOrganCross1 import MultiOrganCross1
from .models.M3D.Uniformer.MultiOrganCross2 import MultiOrganCross2 
from .models.M3D.Uniformer.MultiOrganCross3 import MultiOrganCross3
from .datasets.MultiOrganCTdataset import MultiOrganCTdataset, create_loader
from .loss_func.regression_loss import prediction2label

def create_datasets_and_loaders(args, dataset_type, is_training):
    """Create dataset and dataloader"""
    dataset = MultiOrganCTdataset(args, dataset_type=dataset_type, is_training=is_training)
    loader = create_loader(dataset, 
                         batch_size=args.batch_size,
                         is_training=is_training,
                         num_workers=args.workers,
                         pin_memory=True)
    return loader

def get_model(args):
    """Get model based on args"""
    model_map = {
        'Base': MultiOrganBase,
        'Inter': MultiOrganInter, 
        'Cross1': MultiOrganCross1,
        'Cross2': MultiOrganCross2,
        'Cross3': MultiOrganCross3,
        'Inter_add': MultiOrganInter_add,
        'Inter_cat': MultiOrganInter_cat,
        'Inter_cross': MultiOrganInter_cross,
        'Inter_cross_v2': MultiOrganInter_cross_v2,
        'Inter_cross_v3': MultiOrganInter_cross_v3
    }
    
    model_class = model_map.get(args.model_use)
    if not model_class:
        raise ValueError(f"Invalid model_use: {args.model_use}")
        
    return model_class(args, pretrained_paths=None)

def extract_features_and_labels(model, loader, device):
    """Extract features and labels from model"""
    features = []
    true_labels = []
    pred_labels = [] 
    scores = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting features'):
            # Unpack batch
            tensor_elements = batch[:-1]
            non_tensor_elements = batch[-1:]
            
            # Convert tensors
            tensor_elements = [x.to(device).float() for x in tensor_elements]
            esophagus_img, liver_img, spleen_img, full_img, labels = tensor_elements
            txt = non_tensor_elements[0] if non_tensor_elements else None
            
            # Get model outputs
            model.typeloss = 'tsne'
            esophagus_out, liver_out, spleen_out, multi_organ_out = model(
                esophagus_img, liver_img, spleen_img, full_img, txt)
            
            pred = prediction2label(multi_organ_out)
            
            # Store results
            features.append(multi_organ_out.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            scores.extend(multi_organ_out.cpu().numpy())
    
    return np.vstack(features), np.array(true_labels), np.array(pred_labels), np.array(scores)

def plot_tsne(features, true_labels, pred_labels, save_path):
    """Plot t-SNE visualization"""
    # t-SNE parameters
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=20,
        early_exaggeration=12,
        learning_rate='auto',
        n_iter=1000
    )
    features_tsne = tsne.fit_transform(features)
    
    # Plot settings
    plt.rcParams.update({'font.size': 25})
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Colors for different grades
    color_list = ['#B1D0E0', '#FFCD6B', '#59C1BD', '#FF6B6B']
    unique_labels = np.sort(np.unique(true_labels))
    color_map = {label: color_list[i] for i, label in enumerate(unique_labels)}
    
    # Plot points
    for label in unique_labels:
        mask = true_labels == label
        ax1.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                   c=color_map[label], label=f'Grade {label}',
                   s=80, alpha=0.8)
    
    # Labels and formatting
    ax1.set_xlabel('t-SNE Component 1', fontsize=25)
    ax1.set_ylabel('t-SNE Component 2', fontsize=25)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_features(args, device):
    """Main feature visualization function"""
    # Initialize model
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Create data loader
    loader = create_datasets_and_loaders(args, args.dataset_type, False)
    
    # Extract features
    features, true_labels, pred_labels, scores = extract_features_and_labels(
        model, loader, device)
    
    # Create visualization
    save_path = os.path.join(args.results_dir, f'tsne_{args.dataset_type}_20.pdf')
    plot_tsne(features, true_labels, pred_labels, save_path)
    logging.info(f"t-SNE visualization saved to {save_path}")

def run_infer_main_tsne(args):
    """Main t-SNE visualization pipeline"""
    logging.info("Starting t-SNE visualization...")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    visualize_features(args, device)
    
    logging.info("t-SNE visualization completed!")