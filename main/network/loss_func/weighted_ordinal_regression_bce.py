import torch
import torch.nn.functional as F


def weighted_ordinal_regression_bce(predictions, targets, class_weights=None):
    """
    Perform weighted ordinal regression using binary cross-entropy loss for each pair of
    consecutive classes, with additional emphasis on higher-level categories.
    
    Arguments:
        predictions (torch.Tensor): A tensor containing the predicted probabilities
            for each class. Shape: [batch_size, num_labels - 1]
        targets (torch.Tensor): A tensor containing the target labels. Shape: [batch_size]
        class_weights (torch.Tensor): A tensor containing the weights for each class.
    
    Returns:
        torch.Tensor: The weighted average binary cross-entropy loss over all pairwise classifications.
    """
    # Ensure targets is a long tensor since it will be used as index
    targets = targets.long()
    
    # Default class weights if not provided

    if class_weights is None:
        class_weights = torch.ones(predictions.size(1), device=predictions.device)
        
    # Create a binary matrix for pairwise comparisons
    # num_labels = predictions.size(1) + 1
    num_labels = predictions.size(1) + 1

    pairwise_targets = torch.arange(num_labels, device=targets.device).unsqueeze(0) <= targets.unsqueeze(1)
    pairwise_targets = pairwise_targets[:, 1:].float()  # Exclude the first column since it's always True
    
    # Calculate binary cross-entropy loss for each pairwise comparison
    loss = F.binary_cross_entropy_with_logits(predictions, pairwise_targets, reduction='none')

    # Weight the loss for each classification pair based on class weights
    for i in range(num_labels - 1):
        loss[:, i] *= class_weights[i]

    # Take the mean over all pairwise comparisons
    return loss.mean()


def prediction2label(logits):

    """
    Convert model logits (predictions for each class pair) into final class labels.
    
    Arguments:
        logits (torch.Tensor): A tensor containing the predicted logits
            for each class pair. Shape: [batch_size, num_labels - 1]
    
    Returns:
        torch.Tensor: The predicted class labels. Shape: [batch_size]
    """

    probabilities = torch.sigmoid(logits)
    print(probabilities)

    binary_decisions = (probabilities > 0.5).int()

    num_labels = logits.size(1) + 1
    class_labels = torch.full((logits.size(0),), num_labels - 1, dtype=torch.int64, device=logits.device)

    for i in range(logits.size(1) - 1, -1, -1):
        class_labels -= binary_decisions[:, i]
    print(class_labels)

    return class_labels
