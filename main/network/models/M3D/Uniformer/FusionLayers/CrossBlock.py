import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class DepthwiseConv3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(2, 2, 2), stride=1, padding=1, groups=dim),
            nn.BatchNorm3d(dim),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.dwconv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # B, C, D, H, W -> B, C, 1, 1, 1
        y = self.avg_pool(x)
        # B, C, 1, 1, 1 -> B, C
        y = torch.flatten(y, 1)
        # B, C -> B, C/r -> B, C
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # B, C -> B, C, 1, 1, 1
        y = y.view(x.size(0), x.size(1), 1, 1, 1)
        return y
        # return x * y.expand_as(x)
    

class LKA3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
        

        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=9,
                                      groups=dim, dilation=3)
        
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x
        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        
        return u * attn



class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x



class FusionBlock(nn.Module):

    def __init__(self, config,n_embd = 128, target_img = (8, 16, 16)):
        super().__init__()
        self.config = config
        self.n_embd = n_embd

        

        self.img_esc_dim = (target_img[0], target_img[1], target_img[2])
        self.img_other_dim = (target_img[0], target_img[1], target_img[2])

        pos_emb_size =  (np.prod(self.img_esc_dim) + np.prod(self.img_other_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_emb_size, self.n_embd))
        
        self.drop = nn.Dropout(config.embd_pdrop)  
        self.blocks = nn.Sequential(*[  
            Block(self.n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
            for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)  
        self.apply(self._init_weights) 

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_esc, image_other):


        batch_size = image_esc.shape[0] 

        image_esc = image_esc.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.n_embd)
        image_other = image_other.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.n_embd)

        token_embeddings = torch.cat((image_esc, image_other), dim=1)
        x = self.drop(self.pos_emb + token_embeddings)
        
        x = self.blocks(x)


        x = self.ln_f(x)
        
        image_esc_out = x[:, :np.prod(self.img_esc_dim), :].reshape(batch_size, *self.img_esc_dim, self.n_embd).permute(0, 4, 1, 2, 3)
        image_other_out = x[:, np.prod(self.img_esc_dim):, :].reshape(batch_size, *self.img_other_dim, self.n_embd).permute(0, 4, 1, 2, 3)

        return image_esc_out, image_other_out


class CrossAttention(nn.Module):
    """
    Cross attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, context):
        B, T, C = x.size()
        _, S, _ = context.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(context).view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)
        v = self.value(context).view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)

        # cross-attend: (B, nh, T, hs) x (B, nh, hs, S) -> (B, nh, T, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CBlock(nn.Module):
    """ Transformer block with cross attention """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln1_context = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CrossAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, context):
        x = x + self.attn(self.ln1(x), self.ln1_context(context))
        x = x + self.mlp(self.ln2(x))
        return x


class CrossFusionBlock(nn.Module):
    def __init__(self, config, n_embd=128, target_img=(8, 16, 16)):
        super().__init__()
        self.config = config
        self.n_embd = n_embd

        self.img_esc_dim = (target_img[0], target_img[1], target_img[2])
        self.img_other_dim = (target_img[0], target_img[1], target_img[2])
        
        pos_emb_size = (np.prod(self.img_esc_dim) + np.prod(self.img_other_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_emb_size, self.n_embd))
        
        self.drop = nn.Dropout(config.embd_pdrop)
        # Create separate blocks for cross attention in both directions
        self.blocks_esc = nn.ModuleList([
            CBlock(self.n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
            for _ in range(config.n_layer)])
        self.blocks_other = nn.ModuleList([
            CBlock(self.n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
            for _ in range(config.n_layer)])
            
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_esc, image_other):
        batch_size = image_esc.shape[0]

        image_esc = image_esc.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.n_embd)
        image_other = image_other.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.n_embd)

        # Add positional embeddings
        image_esc = self.drop(image_esc + self.pos_emb[:, :np.prod(self.img_esc_dim), :])
        image_other = self.drop(image_other + self.pos_emb[:, np.prod(self.img_esc_dim):, :])

        # Cross attention in both directions
        x_esc = image_esc
        x_other = image_other
        
        for block_esc, block_other in zip(self.blocks_esc, self.blocks_other):
            x_esc = block_esc(x_esc, x_other)
            x_other = block_other(x_other, x_esc)

        x_esc = self.ln_f(x_esc)
        x_other = self.ln_f(x_other)

        # Reshape back to original dimensions
        image_esc_out = x_esc.reshape(batch_size, *self.img_esc_dim, self.n_embd).permute(0, 4, 1, 2, 3)
        image_other_out = x_other.reshape(batch_size, *self.img_other_dim, self.n_embd).permute(0, 4, 1, 2, 3)
        
        return image_esc_out, image_other_out
    
class CrossFusionBlock_v2(nn.Module):
    def __init__(self, config, n_embd=128, target_img=(8, 16, 16)):
        super().__init__()
        self.config = config
        self.n_embd = n_embd

        self.img_eso_dim = (target_img[0], target_img[1], target_img[2])
        self.img_other_dim = (target_img[0], target_img[1], target_img[2])
        
        pos_emb_size = (np.prod(self.img_eso_dim) + np.prod(self.img_other_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_emb_size, self.n_embd))
        
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.blocks_eso = nn.ModuleList([
            CBlock(self.n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
            for _ in range(config.n_layer)])
            
        self.other_process = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, n_embd),
            nn.ReLU(True),
            nn.Linear(n_embd, n_embd),
            nn.Dropout(config.resid_pdrop)
        )
            
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_eso, image_other):
        batch_size = image_eso.shape[0]

        # Reshape inputs
        image_eso = image_eso.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.n_embd)
        image_other = image_other.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.n_embd)

        # Add positional embeddings
        image_eso = self.drop(image_eso + self.pos_emb[:, :np.prod(self.img_eso_dim), :])
        image_other = self.drop(image_other + self.pos_emb[:, np.prod(self.img_eso_dim):, :])

        # Process eso features with cross attention
        x_eso = image_eso
        for block in self.blocks_eso:
            x_eso = block(x_eso, image_other)  
        x_eso = self.ln_f(x_eso)

        # Simple processing for other features
        x_other = self.other_process(image_other)
        x_other = self.ln_f(x_other)

        # Reshape back to original dimensions
        image_eso_out = x_eso.reshape(batch_size, *self.img_eso_dim, self.n_embd).permute(0, 4, 1, 2, 3)
        image_other_out = x_other.reshape(batch_size, *self.img_other_dim, self.n_embd).permute(0, 4, 1, 2, 3)
        
        return image_eso_out, image_other_out

class Config:
    n_head = 8    
    block_exp = 4 #
    n_layer = 8   
    embd_pdrop = 0.1  
    attn_pdrop = 0.1  
    resid_pdrop = 0.1 
    gpt_linear_layer_init_mean = 0.0  
    gpt_linear_layer_init_std = 0.02  
    gpt_layer_norm_init_weight = 1.0  

def create_fusion_block(**kwargs):
    config = Config()
    model = FusionBlock(config, **kwargs)
    return model

def create_crossfusion_block(**kwargs):
    config = Config()
    model = CrossFusionBlock(config, **kwargs)
    return model

def create_crossfusion_block_v2(**kwargs):
    config = Config()
    model = CrossFusionBlock_v2(config, **kwargs)
    return model


class StageComponents1(nn.Module):
    def __init__(self, embed_dim, target_img_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(target_img_size)
        self.transfusion_xy = create_fusion_block(n_embd=embed_dim, target_img=target_img_size)
        self.transfusion_xz = create_fusion_block(n_embd=embed_dim, target_img=target_img_size)

    def forward(self, x, y, z):
        xp = self.avgpool(x)
        yp = self.avgpool(y)
        zp = self.avgpool(z)

        xt1, yt = self.transfusion_xy(xp, yp)
        xt2, zt = self.transfusion_xz(xp, zp)

        xf1 = F.interpolate(xt1, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)
        xf2 = F.interpolate(xt2, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)

        yf = F.interpolate(yt, size=(y.shape[2],y.shape[3],y.shape[4]), mode='trilinear', align_corners=False)
        zf = F.interpolate(zt, size=(z.shape[2],z.shape[3],z.shape[4]), mode='trilinear', align_corners=False)

        x = x + xf1 + xf2
        y = y + yf
        z = z + zf

        return x,y,z

class StageComponent_crossfusion(nn.Module):
    def __init__(self, embed_dim, target_img_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(target_img_size)
        self.transfusion_xy = create_crossfusion_block(n_embd=embed_dim, target_img=target_img_size)
        self.transfusion_xz = create_crossfusion_block(n_embd=embed_dim, target_img=target_img_size)

    def forward(self, x, y, z):
        xp = self.avgpool(x)
        yp = self.avgpool(y)
        zp = self.avgpool(z)

        xt1, yt = self.transfusion_xy(xp, yp)
        xt2, zt = self.transfusion_xz(xp, zp)

        xf1 = F.interpolate(xt1, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)
        xf2 = F.interpolate(xt2, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)

        yf = F.interpolate(yt, size=(y.shape[2],y.shape[3],y.shape[4]), mode='trilinear', align_corners=False)
        zf = F.interpolate(zt, size=(z.shape[2],z.shape[3],z.shape[4]), mode='trilinear', align_corners=False)

        x = x + xf1 + xf2
        y = y + yf
        z = z + zf

        return x,y,z

class StageComponent_crossfusion_v2(nn.Module):
    def __init__(self, embed_dim, target_img_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(target_img_size)
        self.transfusion_xy = create_crossfusion_block_v2(n_embd=embed_dim, target_img=target_img_size)
        self.transfusion_xz = create_crossfusion_block_v2(n_embd=embed_dim, target_img=target_img_size)

    def forward(self, x, y, z):
        xp = self.avgpool(x)
        yp = self.avgpool(y)
        zp = self.avgpool(z)

        xt1, yt = self.transfusion_xy(xp, yp)
        xt2, zt = self.transfusion_xz(xp, zp)

        xf1 = F.interpolate(xt1, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)
        xf2 = F.interpolate(xt2, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)

        yf = F.interpolate(yt, size=(y.shape[2],y.shape[3],y.shape[4]), mode='trilinear', align_corners=False)
        zf = F.interpolate(zt, size=(z.shape[2],z.shape[3],z.shape[4]), mode='trilinear', align_corners=False)

        x = x + xf1 + xf2
        y = y + yf
        z = z + zf

        return x,y,z

class StageComponents_add(nn.Module):
    def __init__(self, embed_dim=None, target_img_size=None):
        super().__init__()

    def forward(self, x, y, z):

        # Determine the target size for avgpooling based on the smallest dimension among the three inputs
        target_size = (
            min(x.shape[2], y.shape[2], z.shape[2]),
            min(x.shape[3], y.shape[3], z.shape[3]),
            min(x.shape[4], y.shape[4], z.shape[4])
        )

        # Apply average pooling to all inputs
        xp = F.adaptive_avg_pool3d(x, target_size)
        yp = F.adaptive_avg_pool3d(y, target_size)
        zp = F.adaptive_avg_pool3d(z, target_size)

        # Add the pooled outputs
        summed_pooled = xp + yp + zp
        # Interpolate to original sizes for each input
        xf = F.interpolate(summed_pooled, size=x.shape[2:], mode='trilinear', align_corners=False)
        yf = F.interpolate(summed_pooled, size=y.shape[2:], mode='trilinear', align_corners=False)
        zf = F.interpolate(summed_pooled, size=z.shape[2:], mode='trilinear', align_corners=False)

        # Return the final outputs
        x_out = x + xf
        y_out = y + yf
        z_out = z + zf

        return x_out, y_out, z_out

class StageComponents_concat(nn.Module):
    def __init__(self, embed_dim=None, target_img_size=None):
        super().__init__()
        self.conv1x1 = nn.Conv3d(3 * embed_dim, embed_dim, kernel_size=1)  # Reduce from 3x to 1x

    def forward(self, x, y, z):

        # Determine the target size for avgpooling based on the smallest dimension among the three inputs
        target_size = (
            min(x.shape[2], y.shape[2], z.shape[2]),
            min(x.shape[3], y.shape[3], z.shape[3]),
            min(x.shape[4], y.shape[4], z.shape[4])
        )

        # Apply average pooling to all inputs
        xp = F.adaptive_avg_pool3d(x, target_size)
        yp = F.adaptive_avg_pool3d(y, target_size)
        zp = F.adaptive_avg_pool3d(z, target_size)

        # Concatenate the pooled outputs along the channel dimension
        concatenated = torch.cat((xp, yp, zp), dim=1)

        # Reduce channels using 1x1 conv
        reduced = self.conv1x1(concatenated)

        # Interpolate to original sizes for each input
        xf = F.interpolate(reduced, size=x.shape[2:], mode='trilinear', align_corners=False)
        yf = F.interpolate(reduced, size=y.shape[2:], mode='trilinear', align_corners=False)
        zf = F.interpolate(reduced, size=z.shape[2:], mode='trilinear', align_corners=False)

        # Return the final outputs
        x_out = x + xf
        y_out = y + yf
        z_out = z + zf

        return x_out, y_out, z_out


class StageComponents2(nn.Module):
    def __init__(self, embed_dim, target_img_size):
        super().__init__()
        # self.dwconv1 = DepthwiseConv3D(dim=embed_dim)
        # self.channel_interaction1 = ChannelAttention(channel=embed_dim)

        # self.dwconv2 = DepthwiseConv3D(dim=embed_dim)
        # self.channel_interaction2 = ChannelAttention(channel=embed_dim)
        self.lka3d1= LKA3D(dim=embed_dim)
        self.lka3d2= LKA3D(dim=embed_dim)
        self.avgpool = nn.AdaptiveAvgPool3d(target_img_size)
        self.transfusion = create_fusion_block(n_embd=embed_dim, target_img=target_img_size)
    def forward(self, x, y):
        # print('ori',x.shape,y.shape)
        xlk = self.lka3d1(x)
        ylk = self.lka3d2(y)
        print('lk',xlk.shape,ylk.shape)
        xp = self.avgpool(x)
        yp = self.avgpool(y)
        xt, yt = self.transfusion(xp, yp)

        # xd = self.dwconv1(xp)
        # xc = self.channel_interaction1(xd)

        # yd = self.dwconv2(yp)
        # yc = self.channel_interaction2(yd)

        xf = F.interpolate(xt, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)
        yf = F.interpolate(yt, size=(y.shape[2],y.shape[3],y.shape[4]), mode='trilinear', align_corners=False)
        # print('f',xf.shape,yf.shape)
        x = x+xlk*xf
        y = y+ylk*yf
        # print('final',x.shape,y.shape)
        return x,y


class StageComponents3(nn.Module):
    def __init__(self, embed_dim, target_img_size):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool3d(target_img_size)
        self.transfusion = create_fusion_block(n_embd=embed_dim, target_img=target_img_size)
    def forward(self, x, y):

        xp = self.avgpool(x)
        yp = self.avgpool(y)
        xt, yt = self.transfusion(xp, yp)

        xf = F.interpolate(xt, size=(x.shape[2],x.shape[3],x.shape[4]), mode='trilinear', align_corners=False)
        yf = F.interpolate(yt, size=(y.shape[2],y.shape[3],y.shape[4]), mode='trilinear', align_corners=False)
        
        x = x + xf
        y = y + yf
        return x,y

if __name__ == '__main__':

    # model = create_fusion_block(n_embd=128, target_img=(8, 16, 16))
    model = StageComponents_concat(embed_dim=128, target_img_size=(8, 16, 16))

    
    batch_size = 2
    image_esc = torch.rand(batch_size, 128, 8, 16, 16)
    image_other = torch.rand(batch_size, 128, 8, 16, 16)
    
    image_esc_out, image_other_out = model(image_esc, image_other)
    
    print("image_esc_out shape:", image_esc_out.shape)
    print("image_other_out shape:", image_other_out.shape)