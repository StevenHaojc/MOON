import os
import time
import csv
import sys
import torch
import logging
import torch.optim as optim
from torch.optim import Adam, AdamW, SGD
import numpy as np
from os.path import join, abspath, dirname
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from .datasets.MultiOrganCTdataset import MultiOrganCTdataset, create_loader
from .loss_func.regression_loss import ordinal_regression, prediction2label, CCALoss 
from .models.M3D.Uniformer.MultiOrganBase import MultiOrganBase
from .models.M3D.Uniformer.MultiOrganInter import MultiOrganInter
from .models.M3D.Uniformer.MultiOrganCross1 import MultiOrganCross1
from .models.M3D.Uniformer.MultiOrganCross2 import MultiOrganCross2
from .models.M3D.Uniformer.MultiOrganCross3 import MultiOrganCross3
from .metrics.metrics_ordin import compute_metrics_one_gpu
from .models.M3D.Uniformer import *

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.insert(0, root_dir)

def initialize_model(args, device):
    pretrained_paths = {
        'esophagus': args.pretrained_model_path,
        'liver': args.pretrained_model_path,
        'spleen': args.pretrained_model_path
    }

    model_map = {
        'Base': MultiOrganBase,
        'Inter': MultiOrganInter,
        'Cross1': MultiOrganCross1,
        'Cross2': MultiOrganCross2,
        'Cross3': MultiOrganCross3
    }
    
    if args.model_use not in model_map:
        raise ValueError(f"Invalid model_use: {args.model_use}")
    
    model = model_map[args.model_use](args, pretrained_paths=pretrained_paths)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer_map = {
        'Adam': Adam,
        'SGD': SGD, 
        'AdamW': AdamW
    }
    
    if args.optimizer_use not in optimizer_map:
        raise ValueError(f"Invalid optimizer_use: {args.optimizer_use}")
        
    optimizer = optimizer_map[args.optimizer_use](
        model.parameters(), lr=1e-4, weight_decay=1e-5)

    criterion = 0.9*ordinal_regression + 0.1*CCALoss(args.num_classes)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    return model, optimizer, criterion, scheduler

def create_datasets_and_loaders(args, dataset_type, is_training):
    datasets = MultiOrganCTdataset(args, dataset_type=dataset_type, is_training=is_training)
    loaders = create_loader(datasets, batch_size=args.batch_size, is_training=is_training, 
                          num_workers=args.workers, pin_memory=True)
    return loaders

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0.0
    progress_bar = tqdm(enumerate(loader), total=len(loader), leave=False)
    
    for i, batch_data in progress_bar:
        tensor_elements = batch_data[:-1]
        non_tensor_elements = batch_data[-1:]

        tensor_elements = [x.to(device).float() for x in tensor_elements]
        esophagus_img, liver_img, spleen_img, full_img, labels = tensor_elements
        txt = non_tensor_elements[0] if non_tensor_elements else None
        labels = labels.to(device).long()

        optimizer.zero_grad()
    
        if scaler:
            with autocast():
                _,_,_,outputs = model(esophagus_img, liver_img, spleen_img, full_img, txt)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(esophagus_img, liver_img, spleen_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(loader, desc='Validating', leave=False)

    for batch_data in progress_bar:
        tensor_elements = batch_data[:-1]
        non_tensor_elements = batch_data[-1:]

        tensor_elements = [x.to(device).float() for x in tensor_elements]
        esophagus_img, liver_img, spleen_img, full_img, labels = tensor_elements
        txt = non_tensor_elements[0] if non_tensor_elements else None
        labels = labels.to(device).long()

        with autocast(enabled=True):
            _,_,_,outputs = model(esophagus_img, liver_img, spleen_img, full_img, txt)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        predictions = prediction2label(outputs)
        all_predictions.append(predictions)
        all_labels.append(labels)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    return total_loss / len(loader), compute_metrics_one_gpu(all_predictions, all_labels)

def save_model(model, path):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, path)

def train(model, train_loaders, val_loaders, optimizer, criterion, scheduler, device, args, logger):
    writer = SummaryWriter(log_dir=args.output_results)
    best_metrics = {metric: (-np.inf if metric != 'Val_loss' else np.inf)
                   for metric in ['acc', 'f1', 'precision', 'kappa', 'recall', 'Val_loss']}
    scaler = GradScaler()

    metrics_csv_path = join(args.output_results, 'valid_metrics.csv')
    fieldnames = ['Epoch', 'Train_loss', 'Val_loss', 'Time', 'acc', 'f1', 'precision', 'kappa', 'recall']

    with open(metrics_csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in tqdm(range(args.num_epochs), desc='Training'):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loaders, optimizer, criterion, device, scaler)
        val_loss, val_metrics = validate(model, val_loaders, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - start_time
        val_metrics.update({
            'Epoch': epoch + 1,
            'Train_loss': train_loss,
            'Val_loss': val_loss,
            'Time': epoch_duration
        })

        with open(metrics_csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(val_metrics)

        for metric, value in val_metrics.items():
            if metric in best_metrics:
                if (metric != 'Val_loss' and value > best_metrics[metric]) or \
                   (metric == 'Val_loss' and value < best_metrics[metric]):
                    best_metrics[metric] = value
                    save_model(model, join(args.output_results, f'best_model_{metric}.pth'))

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_metrics['acc'], epoch)

    writer.close()
    return best_metrics

def run_train_main_one_gpu(args):
    logging.info("Starting training...")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, optimizer, criterion, scheduler = initialize_model(args, device)
    train_loader = create_datasets_and_loaders(args, 'train', True)
    val_loader = create_datasets_and_loaders(args, 'valid', False)
    
    train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, args, logging)
    logging.info("Training completed!")