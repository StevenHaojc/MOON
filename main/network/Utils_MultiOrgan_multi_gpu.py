import os
import time
import csv
import sys
import torch
import logging
from torch.optim import Adam, AdamW, SGD
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from .models.M3D.Uniformer.MultiOrganBase import MultiOrganBase
from .models.M3D.Uniformer.MultiOrganInter import MultiOrganInter
from .models.M3D.Uniformer.MultiOrganCross1 import MultiOrganCross1
from .models.M3D.Uniformer.MultiOrganCross2 import MultiOrganCross2
from .models.M3D.Uniformer.MultiOrganCross3 import MultiOrganCross3

from .datasets.MultiOrganCTdataset import MultiOrganCTdataset, create_loader
from .loss_func.regression_loss import ordinal_regression, prediction2label, CCALoss
from .metrics.metrics_ordin import compute_metrics_multi_gpu
from .models.M3D.Uniformer import *

def initialize_model(args, device):
    """Initialize model, optimizer, criterion and scheduler"""
    pretrained_paths = {
        'esophagus': args.pretrained_model_path,
        'liver': args.pretrained_model_path, 
        'spleen': args.pretrained_model_path
    }

    # Initialize model based on args
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
        
    model = model_class(args, pretrained_paths)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Initialize optimizer
    optimizer_map = {
        'Adam': Adam,
        'SGD': SGD,
        'AdamW': AdamW
    }
    optimizer_class = optimizer_map.get(args.optimizer_use)
    if not optimizer_class:
        raise ValueError(f"Invalid optimizer_use: {args.optimizer_use}")
        
    optimizer = optimizer_class(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Loss and scheduler
    criterion = 0.9*ordinal_regression + 0.1*CCALoss(args.num_classes)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return model, optimizer, criterion, scheduler

def compute_head_outputs(x, y, z, model, args):
    """Compute head outputs based on fusion method"""
    if args.fusion_method == 'sum':
        headx = torch.mm(x, model.module.multi_organ_head.fc_x.weight.t()) + model.module.multi_organ_head.fc_x.bias
        heady = torch.mm(y, model.module.multi_organ_head.fc_y.weight.t()) + model.module.multi_organ_head.fc_y.bias
        headz = torch.mm(z, model.module.multi_organ_head.fc_z.weight.t()) + model.module.multi_organ_head.fc_z.bias
    
    elif args.fusion_method == 'concat':
        fc_weights = model.module.multi_organ_head.fc_out.weight
        fc_bias = model.module.multi_organ_head.fc_out.bias
        segment_size = fc_weights.size(1) // 3
        
        def compute_head(input_tensor, weight_slice):
            return torch.mm(input_tensor, weight_slice.t()) + fc_bias / 3
            
        headx = compute_head(x, fc_weights[:, :segment_size])
        heady = compute_head(y, fc_weights[:, segment_size:segment_size*2])
        headz = compute_head(z, fc_weights[:, segment_size*2:])
        
    else:  # film, lrf, caf1, caf2
        if args.fusion_method == 'caf2':
            headx, heady, headz = x, y, z
        else:
            headx = heady = headz = outputs
            
    return xyz_to_sigmoid(headx, heady, headz)

def train_one_epoch(args, epoch, model, loader, optimizer, criterion, writer, device, scaler):
    """Train for one epoch"""
    model.train()
    all_labels = []
    all_predictions = []
    modal_predictions = {k:[] for k in ['x','y','z']}
    total_losses = {k:0.0 for k in ['all','x','y','z']}
    
    progress_bar = tqdm(loader, desc='Training', leave=False)
    
    for batch in progress_bar:
        esophagus_img, liver_img, spleen_img, full_img, labels, txt = [x.to(device).float() for x in batch]
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        with autocast():
            x,y,z,outputs = model(esophagus_img, liver_img, spleen_img, full_img, txt)
            headx, heady, headz = compute_head_outputs(x, y, z, model, args)
            
            # Compute losses
            loss = criterion(outputs, labels)
            loss_x = criterion(headx, labels) 
            loss_y = criterion(heady, labels)
            loss_z = criterion(headz, labels)
            
            # Get predictions
            predictions = prediction2label(outputs)
            modal_preds = {
                'x': calculate_xyz_cls(headx),
                'y': calculate_xyz_cls(heady), 
                'z': calculate_xyz_cls(headz)
            }
            
            # Store predictions
            all_predictions.append(predictions)
            all_labels.append(labels)
            for k,v in modal_preds.items():
                modal_predictions[k].append(v)
                
            # Update losses
            total_losses['all'] += loss.item()
            total_losses['x'] += loss_x.item()
            total_losses['y'] += loss_y.item() 
            total_losses['z'] += loss_z.item()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        progress_bar.set_description(f"Train loss: {loss.item():.4f}")
        
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = {
        'all': compute_metrics_multi_gpu(all_predictions, all_labels),
        'x': compute_metrics_multi_gpu(torch.cat(modal_predictions['x']), all_labels),
        'y': compute_metrics_multi_gpu(torch.cat(modal_predictions['y']), all_labels),
        'z': compute_metrics_multi_gpu(torch.cat(modal_predictions['z']), all_labels)
    }
    
    # Log metrics
    for k in total_losses:
        total_losses[k] /= len(loader)
        writer.add_scalar(f'Loss_tr/{k}', total_losses[k], epoch)
    
    writer.add_scalar('Accuracy_tr/all', metrics['all']['acc'], epoch)
    writer.add_scalar('Accuracy_tr/g1', metrics['all']['acc1'], epoch) 
    writer.add_scalar('Accuracy_tr/g2', metrics['all']['acc2'], epoch)
    
    for k in ['x','y','z']:
        writer.add_scalar(f'Accuracy_tr/{k}', metrics[k]['acc'], epoch)
        
    return total_losses['all'], metrics['all'], metrics['x'], metrics['y'], metrics['z']

def validate(args, epoch, model, loader, criterion, writer, device):
    """Validation function"""
    model.eval()
    with torch.no_grad():
        return train_one_epoch(args, epoch, model, loader, None, criterion, writer, device, None)

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, args, logger):
    """Main training loop"""
    writer = SummaryWriter(log_dir=args.output_results)
    metrics_file = os.path.join(args.output_results, 'metrics.csv')
    
    best_metrics = {metric: (-np.inf if metric != 'Val_loss' else np.inf)
                   for metric in ['Val_acc_a', 'f1', 'precision', 'kappa', 'recall', 'Val_loss']}
    best_merge_acc = -np.inf
    
    scaler = GradScaler()
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Train and validate
        train_loss, train_metrics, train_x, train_y, train_z = train_one_epoch(
            args, epoch, model, train_loader, optimizer, criterion, writer, device, scaler)
            
        val_loss, val_metrics, val_x, val_y, val_z = validate(
            args, epoch, model, val_loader, criterion, writer, device)
            
        scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - start_time
        metrics = {
            'Epoch': epoch + 1,
            'Time': epoch_time,
            'Train_acc_a': train_metrics['acc'],
            'Train_acc_g1': train_metrics['acc1'],
            'Train_acc_g2': train_metrics['acc2'],
            'Trx': train_x['acc'],
            'Try': train_y['acc'], 
            'Trz': train_z['acc'],
            'Tvx': val_x['acc'],
            'Tvy': val_y['acc'],
            'Tvz': val_z['acc'],
            'Val_loss': val_loss,
            **val_metrics
        }
        
        # Save metrics
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
            
        # Save best models
        merge_acc = val_metrics['acc1'] + val_metrics['acc2']
        if merge_acc > best_merge_acc:
            best_merge_acc = merge_acc
            save_model(model, os.path.join(args.output_results, 'best_model_merge.pth'))
            
        for metric in best_metrics:
            curr_value = metrics[metric]
            if ((metric != 'Val_loss' and curr_value > best_metrics[metric]) or
                (metric == 'Val_loss' and curr_value < best_metrics[metric])):
                best_metrics[metric] = curr_value
                save_model(model, os.path.join(args.output_results, f'best_model_{metric}.pth'))
                
    writer.close()
    return best_metrics

def run_train_main_multi_gpu(args):
    """Main training entry point"""
    logging.info("Starting training...")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, optimizer, criterion, scheduler = initialize_model(args, device)
    train_loader = create_datasets_and_loaders(args, 'train', True)
    val_loader = create_datasets_and_loaders(args, 'valid', False)
    
    train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, args, logging)
    
    logging.info("Training completed!")