import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = feature_dim // num_heads
        self.key_dim = feature_dim // num_heads
        self.value_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, self.query_dim * num_heads)
        self.key = nn.Linear(feature_dim, self.key_dim * num_heads)
        self.value = nn.Linear(feature_dim, self.value_dim * num_heads)

    def forward(self, img3, img2, img1):
        q1 = self.query(img1).view(-1, self.num_heads, self.query_dim)
        k2 = self.key(img2).view(-1, self.num_heads, self.key_dim)
        v2 = self.value(img2).view(-1, self.num_heads, self.value_dim)
        k3 = self.key(img3).view(-1, self.num_heads, self.key_dim)
        v3 = self.value(img3).view(-1, self.num_heads, self.value_dim)
        
        attn_weights_12 = F.softmax(torch.matmul(q1, k2.transpose(-2, -1)), dim=-1)
        attn_weights_13 = F.softmax(torch.matmul(q1, k3.transpose(-2, -1)), dim=-1)
        
        attended_features_12 = torch.matmul(attn_weights_12, v2).view(img1.size())
        attended_features_13 = torch.matmul(attn_weights_13, v3).view(img1.size())
        
        combined_features = attended_features_12 + attended_features_13 #+ img1
        
        return combined_features

class TransformerBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=128, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=320, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.multi_cross_attention = MultiCrossAttention(feature_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feature_dim)
        )

    def forward(self, x, y, z):
        x1 = self.conv1(x)
        y1 = self.conv2(y)
        
        x2 = F.adaptive_avg_pool3d(x1, output_size=z.shape[2:])
        y2 = F.adaptive_avg_pool3d(y1, output_size=z.shape[2:])
        
        x3 = x2.view(x2.size(0), -1, x2.size(1))  # [B, N, C]
        y3 = y2.view(y2.size(0), -1, y2.size(1))  # [B, N, C]
        z3 = z.view(z.size(0), -1, z.size(1))     # [B, N, C]
        
        combined_features = self.multi_cross_attention(x3, y3, z3)
        
        z4 = self.norm1(z3 + combined_features)
        
        ff_output = self.ffn(z4)
        
        z4 = self.norm2(z4 + ff_output)
        output = z4.view(z.size())
        
        return output

if __name__ == "__main__":
    feature_dim = 512
    num_heads = 8
    ff_dim = 2048
    transformer_block = TransformerBlock(feature_dim, num_heads, ff_dim)


    batch_size = 1
    channels_x = 128
    channels_y = 320
    channels_z = 512
    depth = height = width = 16


    x = torch.rand(batch_size, channels_x, 8, 8, 8)
    y = torch.rand(batch_size, channels_y, 16, 16, 16)
    z = torch.rand(batch_size, channels_z, 12, 12, 12)


    output = transformer_block(x, y, z)
    print(f'Output shape: {output.shape}')  