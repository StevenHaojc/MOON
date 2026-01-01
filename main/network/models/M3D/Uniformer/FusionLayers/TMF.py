import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        q = self.wq(x[:, :1, :])  #
        k = self.wk(x[:, 1:, :])  
        v = self.wv(x[:, 1:, :]) 

        q = q.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N - 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N - 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x

class CrossFusionblock1(nn.Module):
    def __init__(self, num_features, num_heads, drop_path_rate=0.):
        super().__init__()
        self.cross_attn1 = CrossAttentionBlock(num_features, num_heads, drop_path=drop_path_rate)
        self.cross_attn2 = CrossAttentionBlock(num_features, num_heads, drop_path=drop_path_rate)
        # self.cross_attn3 = CrossAttentionBlock(num_features, num_heads, drop_path=drop_path_rate)
    def forward(self, modal1, modal2, modal3):
        B, C = modal1.size()  # Batch size and channel/feature dimension
        # Ensure modal1 has an extra sequence dimension like modal2 and modal3
        modal1 = modal1.view(B, 1, C)  # Reshape modal1 to have the same number of dimensions
        
        # Ensure modal2 and modal3 also have the sequence dimension
        modal2 = modal2.view(B, -1, C) # Assuming modal2 should have the same feature dimension
        modal3 = modal3.view(B, -1, C) # Assuming modal3 should have the same feature dimension

        # Concatenate modal1 and modal2 for cross-attention
        x1 = torch.cat((modal1, modal2), dim=1)
        fusion1 = self.cross_attn1(x1)[:, 0, :]

        # Concatenate fused_modal1_2 (which now includes modal1) and modal3 for the next cross-attention
        x2 = torch.cat((modal1, modal3), dim=1)
        fusion2 = self.cross_attn2(x2)[:, 0, :]


        return fusion1,fusion2  # Take the fused feature for modal1

class CrossFusionblock2(nn.Module):
    def __init__(self, num_features, num_heads, drop_path_rate=0.):
        super().__init__()
        self.cross_attn1 = CrossAttentionBlock(num_features, num_heads, drop_path=drop_path_rate)
        self.cross_attn2 = CrossAttentionBlock(num_features, num_heads, drop_path=drop_path_rate)
        self.cross_attn3 = CrossAttentionBlock(num_features, num_heads, drop_path=drop_path_rate)
    def forward(self, modal1, modal2, modal3):
        B, C = modal1.size()  # Batch size and channel/feature dimension
        # Ensure modal1 has an extra sequence dimension like modal2 and modal3
        modal1 = modal1.view(B, 1, C)  # Reshape modal1 to have the same number of dimensions
        
        # Ensure modal2 and modal3 also have the sequence dimension
        modal2 = modal2.view(B, -1, C) # Assuming modal2 should have the same feature dimension
        modal3 = modal3.view(B, -1, C) # Assuming modal3 should have the same feature dimension

        # Concatenate modal1 and modal2 for cross-attention
        x = torch.cat((modal1, modal2), dim=1)
        fused_modal1_2 = self.cross_attn1(x)

        # Concatenate fused_modal1_2 (which now includes modal1) and modal3 for the next cross-attention
        x = torch.cat((fused_modal1_2, modal3), dim=1)
        fused_all = self.cross_attn2(x)

        return fused_all[:, 0, :]  # Take the fused feature for modal1



