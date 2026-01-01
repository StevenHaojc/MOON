import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OhemClassLoss(nn.Module):
    def __init__(self, keep_rate):
        super(OhemClassLoss, self).__init__()
        self.rate = keep_rate

    def forward(self, pred, target):
        batch_size = pred.size(0) 
        ohem_cls_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=-1)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*self.rate))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss


def rand_bbox(size, lam):
    W = size[4]
    H = size[3]
    D = size[2]
    # cut_rat = np.sqrt(1. - lam)
    cut_rat = np.cbrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cut_d = np.int(D * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(D)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)
    
    return bbx1, bby1, bbx2, bby2, bbz1, bbz2


def cutmix(data, target, alpha, use_cuda=True):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    
    bbx1, bby1, bbz1, bbx2, bby2, bbz2 = rand_bbox(data.size(), lam)
    data[:, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = data[indices, :, bbz1:bbz2, bby1:bby2, bbx1:bbx22]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1) / (data.size()[-1] * data.size()[-2] * data.size()[-3]))

    targets = [target, shuffled_target, lam]
    return data, targets
    
# loss 
def cutmix_criterion(criterion, pred, target):
    target1, target2, lam = target[0], target[1], target[2]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(pred, target1) + (1 - lam) * criterion(pred, target2)


def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1. - lam)
    targets = [target, shuffled_target, lam]

    return data, targets


def mixup_criterion(criterion, pred, target):
    target1, target2, lam = target[0], target[1], target[2]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(pred, target1) + (1. - lam) * criterion(pred, target2)