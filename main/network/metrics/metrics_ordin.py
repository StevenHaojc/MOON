import torch
import numpy as np
from sklearn import metrics
from collections import OrderedDict
from sklearn.metrics import accuracy_score, roc_auc_score

def convert_labels_for_one_vs_rest(true_labels, threshold):
    """
    Convert multi-class labels to binary labels for one-vs-rest evaluation.
    
    Args:
        true_labels: Original multi-class labels
        threshold: Threshold value for binary conversion
        
    Returns:
        Binary labels (0 if label < threshold, 1 otherwise)
    """
    return [0 if label < threshold else 1 for label in true_labels]

def evaluate_results(true_labels, predicted_labels, thresholds):
    """
    Evaluate predictions for multiple one-vs-rest scenarios.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        thresholds: List of threshold values for one-vs-rest evaluation
        
    Returns:
        Dictionary containing accuracy scores for each threshold
    """
    results = {}
    for threshold in thresholds:
        converted_labels = convert_labels_for_one_vs_rest(true_labels, threshold)
        accuracy = accuracy_score(converted_labels, predicted_labels >= threshold)
        results[f'acc_{threshold}'] = accuracy
    return results

def compute_metrics_multi_gpu(y_pred, y_true):
    """
    Compute comprehensive metrics including one-vs-rest evaluations.
    
    Args:
        y_pred: Model predictions 
        y_true: Ground truth labels
        
    Returns:
        OrderedDict containing various performance metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Compute basic metrics
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    
    # Compute one-vs-rest accuracies
    one_vs_rest = evaluate_results(y_true, y_pred, thresholds=[1, 2])

    return OrderedDict([
        ('acc', acc),
        ('acc1', one_vs_rest['acc_1']),
        ('acc2', one_vs_rest['acc_2']),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('kappa', kappa),
    ])

def compute_metrics_one_gpu(y_pred, y_true):
    """
    Compute basic performance metrics for single GPU setup.
    
    Args:
        y_pred: Model predictions
        y_true: Ground truth labels
        
    Returns:
        OrderedDict containing performance metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    return OrderedDict([
        ('acc', metrics.accuracy_score(y_true, y_pred)),
        ('f1', metrics.f1_score(y_true, y_pred, average='macro')),
        ('recall', metrics.recall_score(y_true, y_pred, average='macro')),
        ('precision', metrics.precision_score(y_true, y_pred, average='macro')),
        ('kappa', metrics.cohen_kappa_score(y_true, y_pred)),
    ])

def compute_metrics(y_pred, y_true):
    """
    Basic metrics computation wrapper.
    Identical to compute_metrics_one_gpu for consistency.
    """
    return compute_metrics_one_gpu(y_pred, y_true)

# Example usage:
if __name__ == "__main__":
    # Example data
    y_true = torch.tensor([0, 1, 2, 3, 1, 2])
    y_pred = torch.tensor([0, 1, 1, 3, 1, 2])
    
    # Compute metrics
    metrics_multi = compute_metrics_multi_gpu(y_pred, y_true)
    metrics_single = compute_metrics_one_gpu(y_pred, y_true)
    
    print("Multi-GPU metrics:", metrics_multi)
    print("Single-GPU metrics:", metrics_single)