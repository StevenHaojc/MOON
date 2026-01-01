# -*- coding: utf-8 -*-
# @Author  : wama
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from sklearn import metrics



def ordinal_regression(predictions, targets):
    """
    顺序回归，编码方式如https://arxiv.org/pdf/0704.1028.pdf中所示
    """

    # 假设目标是一个张量，你可以使用.item()方法获取它的整数值
    # if isinstance(targets, torch.Tensor) and targets.numel() == 1:
    #     targets = target.item()
    # elif isinstance(targets, (float, np.float32, np.float64)):
    #     # 如果它是一个浮点数（对于索引来说这很不寻常），将其转换为整数
    #     targets = int(targets)

    # 创建具有与预测相同形状的修改后的目标张量 [batch_size, num_labels]
    modified_target = torch.zeros_like(predictions)
    # 填充顺序目标函数，即 0 -> [1,0,0,...]
    if targets.numel() == 1:
        modified_target[0, 0:targets + 1] = 1
    else:
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1

    return nn.MSELoss(reduction='mean')(predictions, modified_target)





def prediction2label(pred):
    """
    将一组有序预测转换为类别标签。
    
    :param pred: 一个2D NumPy数组，每一行包含有序预测。
    :return: 一个1D NumPy数组，包含类别标签。
    """
    pred_tran = (pred > 0.5).cumprod(axis=1)
    # 对每一行求和，找到最后一个1出现的索引
    sums = pred_tran.sum(axis=1)
    # 如果没有1，则类别标签为0，否则为最后一个1的索引减1。
    # 使用PyTorch操作来得到标签，不需要转换为NumPy数组
    labels = torch.where(sums == 0, torch.zeros_like(sums), sums - 1)
    return labels


def prototype_loss(prototype_representations, targets, prototypes):
    """
    计算原型学习损失。
    
    参数:
    - prototype_representations: 输入特征的原型表示。
    - targets: 真实类别标签。
    - prototypes: 原型向量。
    
    返回值:
    - loss: 计算出的损失值。
    """
    # 确保标签是正确的类型
    targets = targets.long()
    
    # 计算每个样本与其对应原型的距离
    positive_distances = torch.norm(prototype_representations - prototypes[targets], dim=1)
    
    # 计算每个样本与最不相似原型的距离
    negative_distances = torch.cdist(prototype_representations, prototypes)
    negative_distances[torch.arange(negative_distances.size(0)), targets] = float('inf')  # 忽略正类别
    negative_distances, _ = negative_distances.min(dim=1)
    
    # 使用合适的损失公式（如三元组损失、对比损失等）
    loss = F.relu(positive_distances - negative_distances + 1)  # 这里假设使用了1作为边界
    
    return loss.mean()



# class GCCALoss(nn.Module):
#     def __init__(self, outdim_size):
#         super(GCCALoss, self).__init__()
#         self.outdim_size = outdim_size

#     def forward(self, H1, H2, H3):
#         """
#         H1, H2, H3 are of the same shape [batch_size, features_dim]
#         """
#         r = 1e-4
#         eps = 1e-12

#         H1, H2, H3 = [H - H.mean(dim=0) for H in [H1, H2, H3]]
#         H1, H2, H3 = [H / (H.std(dim=0) + eps) for H in [H1, H2, H3]]

#         C1 = torch.mm(H1.T, H1) + r * torch.eye(H1.size(1), device=H1.device)
#         C2 = torch.mm(H2.T, H2) + r * torch.eye(H2.size(1), device=H2.device)
#         C3 = torch.mm(H3.T, H3) + r * torch.eye(H3.size(1), device=H3.device)

#         e1, v1 = torch.linalg.eigh(C1, UPLO='U')
#         e2, v2 = torch.linalg.eigh(C2, UPLO='U')
#         e3, v3 = torch.linalg.eigh(C3, UPLO='U')

#         # Use the smallest eigenvectors to form the common space
#         d = min([e1.size(0), e2.size(0), e3.size(0), self.outdim_size])

#         H1, H2, H3 = [torch.mm(H, v[:, -d:]) for H, v in zip([H1, H2, H3], [v1, v2, v3])]

#         # Generalized cca loss is mean of correlation of all pairs
#         gcca_loss = (torch.norm(torch.mm(H1.T, H2)) + 
#                      torch.norm(torch.mm(H2.T, H3)) + 
#                      torch.norm(torch.mm(H1.T, H3))) / (3 * H1.size(1))
        
#         return -gcca_loss  # Negating the loss to maximize correlation



class CCALoss(nn.Module):
    def __init__(self, outdim_size, r1=1e-4, r2=1e-4):
        super(CCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.r1 = r1
        self.r2 = r2
        self.eps = 1e-12

    def forward(self, H1, H2):
        """
        Forward pass of the CCA loss.
        H1, H2 are the same shape [batch_size, features_dim] and are assumed to be normalized.
        """
        H1, H2 = H1 - H1.mean(dim=0), H2 - H2.mean(dim=0)
        H1, H2 = H1 / (H1.std(dim=0) + self.eps), H2 / (H2.std(dim=0) + self.eps)
        o1 = o2 = H1.size(1)
        if H1.size(1) > self.outdim_size:
            o1 = o2 = self.outdim_size
            C1 = H1.T @ H1 + self.r1 * torch.eye(H1.size(1), device=H1.device)
            C2 = H2.T @ H2 + self.r2 * torch.eye(H2.size(1), device=H2.device)
            e1, v1 = torch.linalg.eigh(C1)
            e2, v2 = torch.linalg.eigh(C2)
            
            # Filter out the small eigenvalues
            e1_idx = e1 > e1.abs().max() * self.eps
            e2_idx = e2 > e2.abs().max() * self.eps
            H1 = H1 @ v1[:, e1_idx][:, -o1:]
            H2 = H2 @ v2[:, e2_idx][:, -o2:]
        C = H1.T @ H2
        cca_loss = torch.sum(torch.diag(C)) / (torch.norm(H1) * torch.norm(H2) + self.eps)
        return torch.abs(cca_loss)  # Negating the loss to maximize correlation


    def __init__(self, outdim_size, input_dim=512, hidden_dim=512):
        super(MultiViewCCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.eps = 1e-12
        
        # 为三个视图分别创建投影层
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, outdim_size),
                nn.BatchNorm1d(outdim_size)
            ) for _ in range(3)
        ])

    def compute_pair_loss(self, H1, H2):
        # 标准化
        H1, H2 = H1 - H1.mean(dim=0), H2 - H2.mean(dim=0)
        H1, H2 = H1 / (H1.std(dim=0) + self.eps), H2 / (H2.std(dim=0) + self.eps)
        
        C = H1.T @ H2
        return -torch.sum(torch.diag(C)) / (torch.norm(H1) * torch.norm(H2) + self.eps)

    def forward(self, x, y, z):
        # 投影三个视图
        H1 = self.projections[0](x)
        H2 = self.projections[1](y)
        H3 = self.projections[2](z)
        
        # 计算所有对之间的loss
        loss_xy = self.compute_pair_loss(H1, H2)
        loss_xz = self.compute_pair_loss(H1, H3)
        loss_yz = self.compute_pair_loss(H2, H3)
        
        # 返回平均loss和各个对之间的loss
        avg_loss = (loss_xy + loss_xz + loss_yz) / 3
        return avg_loss, {'xy': loss_xy.item(), 'xz': loss_xz.item(), 'yz': loss_yz.item()}