import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1)
    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_output_weights  

import torch.nn.init as init

class CrossFusion1(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=3, num_heads=1):
        super(CrossFusion1, self).__init__()
        self.low_rank_weights1 = nn.Parameter(init.xavier_uniform_(torch.empty(input_dim, hidden_dim)))
        self.low_rank_weights2 = nn.Parameter(init.xavier_uniform_(torch.empty(input_dim, hidden_dim)))
        self.low_rank_weights3 = nn.Parameter(init.xavier_uniform_(torch.empty(input_dim, hidden_dim)))
        self.low_rank_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * input_dim)  
        )
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True)  
        )

        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        
    def forward(self, x1, x2, x3):
        # Low-rank fusion
        interaction1 = F.relu(torch.matmul(x1, self.low_rank_weights1))
        interaction2 = F.relu(torch.matmul(x2, self.low_rank_weights2))
        interaction3 = F.relu(torch.matmul(x3, self.low_rank_weights3))
        combined_interaction = interaction1 * interaction2 * interaction3 + self.low_rank_bias
        
        # 
        film_params = self.film(combined_interaction)
        gamma, beta = film_params.chunk(2, dim=-1)
        sum_x = x1 + x2 + x3
        film_output = gamma * sum_x + beta

        output = self.fc_out(film_output)
        return x1, x2, x3, output









# class CrossFusion1(nn.Module):
#     def __init__(self, input_dim=512, dim=512, output_dim=100, num_heads=8):
#         super(CrossFusion1, self).__init__()
#         # 因子层（用于生成 gamma 和 beta）
#         self.fc1 = nn.Linear(input_dim, 2 * dim)
#         self.fc2 = nn.Linear(input_dim, 2 * dim)
#         self.fc3 = nn.Linear(input_dim, 2 * dim)
        
#         # 自注意力层
#         self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
#         # 输出层
#         self.fc_out = nn.Linear(dim, output_dim)
        
#         self.dim = dim
        
#     def forward(self, x1, x2, x3):
#         # 分离 gamma 和 beta
#         gamma1, beta1 = torch.split(self.fc1(x1), self.dim, 1)
#         gamma2, beta2 = torch.split(self.fc2(x2), self.dim, 1)
#         gamma3, beta3 = torch.split(self.fc3(x3), self.dim, 1)
        
#         # 结合 gamma 和 beta
#         combined_gamma = gamma1 * gamma2 * gamma3
#         combined_beta = beta1 + beta2 + beta3
        
#         # 将输入相加
#         combined_input = x1 + x2 + x3
#         # 应用自注意力
#         attention_output, _ = self.self_attention(combined_input.unsqueeze(1), combined_input.unsqueeze(1), combined_input.unsqueeze(1))
#         attention_output = attention_output.squeeze(1)  # 移除多余的序列长度维度
        
#         # 应用 FiLM 模块
#         film_output = combined_gamma * attention_output + combined_beta
        
#         # 应用输出层
#         output = self.fc_out(film_output)
        
#         return x1, x2, x3, output
    

# class CrossFusion1(nn.Module):
#     def __init__(self, input_dim = 512, num_prototypes =3, output_dim = 3, num_heads = 8):
#         super(CrossFusion1, self).__init__()
#         self.feature_dim = input_dim
#         self.num_prototypes = num_prototypes
#         self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim * 3))
#         self.cross_attn = CrossAttention(d_model=input_dim * 3, n_heads=num_heads)
#         self.fc_out = nn.Linear(input_dim*3, output_dim)
#        # 输入前的层标准化
#         self.input_norm = nn.LayerNorm(input_dim*3)
#         self.output_norm = nn.LayerNorm(input_dim*3)

#     def compute_similarity(self, features, prototypes):
#         distance = torch.cdist(features, prototypes)
#         similarity = F.softmax(-distance, dim=1)
#         return similarity,-distance

#     def forward(self, x, y, z):
#         # 将来自所有模态的特征向量拼接起来
#         concatenated_features = torch.cat([x, y, z], dim=1)
#         concatenated_features_norm = self.input_norm(concatenated_features)
#         # 基于拼接后的特征进行原型学习
#         similarity, Dist = self.compute_similarity(concatenated_features_norm, self.prototypes)
#         prototype_representation = torch.matmul(similarity, self.prototypes)
#         # 为Cross Attention机制准备输入
#         query = concatenated_features_norm.unsqueeze(0)
#         key = value = prototype_representation.unsqueeze(0)
#         # 应用Cross Attention
#         cross_attended_output, _ = self.cross_attn(query, key, value)
#         # 移除为Cross Attention添加的额外维度
#         cross_attended_output = cross_attended_output.squeeze(0)
#         # 将Cross Attention的输出与原始拼接特征向量再次进行拼接
#         cross_output = cross_attended_output + concatenated_features_norm
#         final_output = self.fc_out(self.output_norm(cross_output))
#         # import pdb
#         # pdb.set_trace()
#         return x, y, Dist, final_output
        



# class CrossFusion1(nn.Module):
#     """
#     TriModalAttentionClassifier is a neural network module that computes attention scores
#     for three input modalities and produces class scores for categorization.
    
#     Parameters:
#     - input_dims (list of int): List of input dimensions for each modality.
#     - attention_dim (int): Dimension of the attention space.
#     - num_classes (int): Number of classes for classification.
#     """
#     def __init__(self, input_dims = [512,512,512], attention_dim=32, output_dim = 3):
#         super(CrossFusion1, self).__init__()
#         self.attention_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(in_features, attention_dim),
#                 nn.Tanh(),
#                 nn.Linear(attention_dim, 1)
#             ) for in_features in input_dims
#         ])
#         self.classifier = nn.Linear(input_dims[0], output_dim)  # Assuming that the feature dimension remains constant after attention

#     def forward(self, modality1, modality2, modality3):
#         # Compute attention scores for each modality
#         attention_scores = torch.cat([
#             attention_layer(modality).unsqueeze(-1)
#             for modality, attention_layer in zip(
#                 [modality1, modality2, modality3], self.attention_layers
#             )
#         ], dim=-1)
        
#         # Apply softmax to get normalized weights
#         normalized_weights = F.softmax(attention_scores, dim=-1)
        
#         # Compute the weighted sum of modalities
#         weighted_sum = modality1 * normalized_weights[:, :, 0] + \
#                        modality2 * normalized_weights[:, :, 1] + \
#                        modality3 * normalized_weights[:, :, 2]
        
#         # Get class scores
#         class_scores = self.classifier(weighted_sum)
        
#         return modality1,modality2,modality3, class_scores

# class CrossFusion2(nn.Module):
#     def __init__(self, input_dims = [512,512,512], attention_dim=32, output_dim = 3):
#         super(CrossFusion2, self).__init__()
        
#         # 注意力层
#         self.attention_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(in_features, attention_dim),
#                 nn.Tanh(),
#                 nn.Linear(attention_dim, 1)
#             ) for in_features in input_dims
#         ])
        
#         # 分支分类器
#         self.classifiers = nn.ModuleList([
#             nn.Linear(in_features, output_dim)
#             for in_features in input_dims
#         ])
        
#     def forward(self, modality1, modality2, modality3):
#         # 计算每个模态的注意力得分
#         attention_scores = torch.cat([
#             attention_layer(modality).unsqueeze(-1)
#             for modality, attention_layer in zip(
#                 [modality1, modality2, modality3], self.attention_layers
#             )
#         ], dim=-1)
        
#         # 使用 softmax 获取归一化的权重
#         normalized_weights = F.softmax(attention_scores, dim=-1)
        
#         # 计算每个分支的分类分数
#         class_scores = [
#             classifier(modality)
#             for modality, classifier in zip(
#                 [modality1, modality2, modality3], self.classifiers
#             )
#         ]
#         # Compute the weighted sum of modalities
#         weighted_sum = class_scores[0] * normalized_weights[:, :, 0] + \
#                        class_scores[1] * normalized_weights[:, :, 1] + \
#                        class_scores[2] * normalized_weights[:, :, 2]
#         return modality1,modality2,modality3,weighted_sum

class CrossFusion2(nn.Module):
    def __init__(self, input_dim=512, output_dim=3):
        super(CrossFusion2, self).__init__()
        # 定义门控参数的生成层
        self.gate_fc = nn.Linear(input_dim, 1)
        
        # 为64维向量添加升维层
        self.fc_v1_up = nn.Linear(64, input_dim)
        self.fc_v2_up = nn.Linear(64, input_dim)
        
        # 修改连接层的输入维度（3个512维 + 2个512维）
        self.fc_xyz = nn.Linear(input_dim*5, input_dim)
        
        # 原有的投影层
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)
        self.fc_z = nn.Linear(input_dim, output_dim)
        
        # 为64维向量添加投影层
        self.fc_v1 = nn.Linear(input_dim, output_dim)
        self.fc_v2 = nn.Linear(input_dim, output_dim)
        
        self.sum_xyz = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z, v1, v2):
        # 将64维向量映射到512维
        v1_mapped = self.fc_v1_up(v1)
        v2_mapped = self.fc_v2_up(v2)
        
        # 连接所有特征
        fusion_xyz = torch.cat([x, y, z, v1_mapped, v2_mapped], dim=1)
        fusion_xyz = self.fc_xyz(fusion_xyz)
        
        # 计算门控参数
        gate = torch.sigmoid(self.gate_fc(fusion_xyz))
        
        # 应用门控机制
        gated_xyz = gate * fusion_xyz
        
        # 另一分支使用1 - 门控参数
        out_x = self.fc_x((1 - gate) * x)
        out_y = self.fc_y((1 - gate) * y)
        out_z = self.fc_z((1 - gate) * z)
        
        # 处理映射后的64维向量
        out_v1 = self.fc_v1((1 - gate) * v1_mapped)
        out_v2 = self.fc_v2((1 - gate) * v2_mapped)
        
        # 融合所有特征
        out_xyz = self.sum_xyz(gated_xyz)
        final_out = out_x + out_y + out_z + out_v1 + out_v2 + out_xyz
        
        return out_x, out_y, out_z, final_out
# class CrossFusion2(nn.Module):
#     def __init__(self, input_dim = 512, output_dim = 3):
#         super(CrossFusion2, self).__init__()
#         # 定义门控参数的生成层
#         self.gate_fc = nn.Linear(input_dim, 1)
#         self.fc_xyz = nn.Linear(input_dim*3, input_dim)
#         self.fc_x = nn.Linear(input_dim,output_dim)
#         self.fc_y = nn.Linear(input_dim,output_dim)
#         self.fc_z = nn.Linear(input_dim,output_dim)
#         self.sum_xyz = nn.Linear(input_dim, output_dim)
#     def forward(self, x, y, z):
#         fusion_xyz = torch.cat([x, y, z], dim=1)
#         fusion_xyz = self.fc_xyz(fusion_xyz)
#         # 计算门控参数
#         gate = torch.sigmoid(self.gate_fc(fusion_xyz))
#         # 应用门控机制
#         gated_xyz = gate * fusion_xyz
#         # 另一分支使用1 - 门控参数
#         out_x = self.fc_x((1 - gate) * x) 
#         out_y = self.fc_y((1 - gate) * y) 
#         out_z = self.fc_z((1 - gate) * z) 
#         out_xyz = self.sum_xyz(gated_xyz)
#         final_out = out_x + out_y + out_z + out_xyz
#         return out_x,out_y,out_z,final_out





class LowRankFusion(nn.Module):
    def __init__(self, input_dims = [512, 512, 512], output_dim = 3, rank = 3):
        super(LowRankFusion, self).__init__()

        # 为每个输入特征创建一个线性层
        self.fc_x = nn.Linear(input_dims[0], output_dim)
        self.fc_y = nn.Linear(input_dims[1], output_dim)
        self.fc_z = nn.Linear(input_dims[2], output_dim)
        
        # 为两个64维向量添加线性层
        self.fc_v1 = nn.Linear(64, output_dim)
        self.fc_v2 = nn.Linear(64, output_dim)
        
        # 创建一个低秩权重矩阵以及一个偏置项
        self.W_r = nn.Parameter(torch.Tensor(rank, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        
        # 初始化参数
        nn.init.kaiming_uniform_(self.W_r)
        nn.init.zeros_(self.bias)

    def forward(self, x, y, z, v1, v2):  # v1, v2 是两个64维向量
        # 分别投影到低维空间
        x_proj = self.fc_x(x)
        y_proj = self.fc_y(y)
        z_proj = self.fc_z(z)
        v1_proj = self.fc_v1(v1)  # 处理第一个64维向量
        v2_proj = self.fc_v2(v2)  # 处理第二个64维向量
        
        # 将投影后的特征堆叠起来
        stacked = torch.stack((x_proj, y_proj, z_proj, v1_proj, v2_proj))
        
        # 沿着秩的维度进行加权和
        output = torch.matmul(stacked, self.W_r)
        
        # 对秩的维度进行求和
        output = torch.sum(output, dim=0)
        
        # 添加偏置项
        output = output + self.bias
        
        return x, y, z, output

# class LowRankFusion(nn.Module):
#     def __init__(self, input_dims = [512, 512, 512], output_dim = 3, rank = 3):
#         super(LowRankFusion, self).__init__()

#         # 为每个输入特征创建一个线性层
#         self.fc_x = nn.Linear(input_dims[0], output_dim)
#         self.fc_y = nn.Linear(input_dims[1], output_dim)
#         self.fc_z = nn.Linear(input_dims[2], output_dim)
        
#         # 创建一个低秩权重矩阵以及一个偏置项
#         self.W_r = nn.Parameter(torch.Tensor(rank, output_dim))
#         self.bias = nn.Parameter(torch.Tensor(output_dim))
        
#         # 初始化参数
#         nn.init.kaiming_uniform_(self.W_r)
#         nn.init.zeros_(self.bias)

#     def forward(self, x, y, z):
#         # 分别投影到低维空间
#         x_proj = self.fc_x(x)
#         y_proj = self.fc_y(y)
#         z_proj = self.fc_z(z)
#         # import pdb
#         # pdb.set_trace()
#         # 将投影后的特征堆叠起来
#         stacked = torch.stack((x_proj, y_proj, z_proj))
        
#         # 沿着秩的维度进行加权和
#         output = torch.matmul(stacked, self.W_r)
        
#         # 对秩的维度进行求和或平均
#         output = torch.sum(output, dim=0)
        
#         # 添加偏置项
#         output = output + self.bias
        
#         return x, y,z, output
class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)
        self.fc_z = nn.Linear(input_dim, output_dim)
        
        # 添加对两个64维向量的处理
        self.fc_v1_up = nn.Linear(64, input_dim)  # 将64维映射到512维
        self.fc_v2_up = nn.Linear(64, input_dim)  # 将64维映射到512维
        self.fc_v1 = nn.Linear(input_dim, output_dim)
        self.fc_v2 = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z, v1, v2):  # v1, v2 是64维向量
        # 处理两个64维向量
        v1_mapped = self.fc_v1_up(v1)  # 64->512
        v2_mapped = self.fc_v2_up(v2)  # 64->512
        
        # 简单相加所有特征
        output = self.fc_x(x) + self.fc_y(y) + self.fc_z(z) + \
                self.fc_v1(v1_mapped) + self.fc_v2(v2_mapped)
                
        return x, y, z, output
# class SumFusion(nn.Module):
#     def __init__(self, input_dim=512, output_dim=100):
#         super(SumFusion, self).__init__()
#         self.fc_x = nn.Linear(input_dim, output_dim)
#         self.fc_y = nn.Linear(input_dim, output_dim)
#         self.fc_z = nn.Linear(input_dim, output_dim)
#     def forward(self, x, y, z):
#         output = self.fc_x(x) + self.fc_y(y) + self.fc_z(z)
#         return x, y,z, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=512, num_organs=3, output_dim=100):    ####### clip
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim*num_organs+128, output_dim)
    def forward(self, x, y, z,txt):
        output = torch.cat((x, y, z,txt), dim=1)
        output = self.fc_out(output)
        return x, y,z, output

class FilmFusion(nn.Module):
    def __init__(self, input_dim=512, dim=512, output_dim=100):
        super(FilmFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2 * dim)
        self.fc2 = nn.Linear(input_dim, 2 * dim)
        self.fc3 = nn.Linear(input_dim, 2 * dim)
        
        # 添加对两个64维向量的处理
        self.fc_v1_up = nn.Linear(64, input_dim)  # 将64维映射到512维
        self.fc_v2_up = nn.Linear(64, input_dim)  # 将64维映射到512维
        self.fc_v1 = nn.Linear(input_dim, 2 * dim)
        self.fc_v2 = nn.Linear(input_dim, 2 * dim)
        
        self.fc_out = nn.Linear(dim, output_dim)
        self.dim = dim
        
    def forward(self, x1, x2, x3, v1, v2):  # v1, v2 是64维向量
        gamma1, beta1 = torch.split(self.fc1(x1), self.dim, 1)
        gamma2, beta2 = torch.split(self.fc2(x2), self.dim, 1)
        gamma3, beta3 = torch.split(self.fc3(x3), self.dim, 1)
        
        # 处理两个64维向量
        v1_mapped = self.fc_v1_up(v1)  # 64->512
        v2_mapped = self.fc_v2_up(v2)  # 64->512
        gamma_v1, beta_v1 = torch.split(self.fc_v1(v1_mapped), self.dim, 1)
        gamma_v2, beta_v2 = torch.split(self.fc_v2(v2_mapped), self.dim, 1)
        
        # 融合所有gamma和beta
        combined_gamma = gamma1 * gamma2 * gamma3 * gamma_v1 * gamma_v2
        combined_beta = beta1 + beta2 + beta3 + beta_v1 + beta_v2
        
        # 融合所有特征
        output = combined_gamma * (x1 + x2 + x3 + v1_mapped + v2_mapped) + combined_beta
        output = self.fc_out(output)
        
        return x1, x2, x3, output
# class FilmFusion(nn.Module):
#     def __init__(self, input_dim=512, dim=512, output_dim=100):
#         super(FilmFusion, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 2 * dim)
#         self.fc2 = nn.Linear(input_dim, 2 * dim)
#         self.fc3 = nn.Linear(input_dim, 2 * dim)
#         self.fc_out = nn.Linear(dim, output_dim)
#         self.dim = dim

#         # 添加对128维向量的处理
#         small_dim=128
#         self.fc_small = nn.Linear(small_dim, input_dim)  # 将128维映射到512维
#         self.fc4 = nn.Linear(input_dim, 2 * dim)  # 用于生成gamma4和beta4

#     def forward(self, x1, x2, x3):
#         gamma1, beta1 = torch.split(self.fc1(x1), self.dim, 1)
#         gamma2, beta2 = torch.split(self.fc2(x2), self.dim, 1)
#         gamma3, beta3 = torch.split(self.fc3(x3), self.dim, 1)
        
#         combined_gamma = gamma1 * gamma2 * gamma3
#         combined_beta = beta1 + beta2 + beta3
        
#         output = combined_gamma * (x1 + x2 + x3) + combined_beta
#         output = self.fc_out(output)
#         # 处理128维向量
#         return x1, x2, x3, output



if __name__ == '__main__':
    # 模拟输入特征的尺寸
    input_dim_x = 512
    input_dim_y = 512
    input_dim_z = 512

    batch_size = 4

    # 创建输入特征的随机样本
    x = torch.randn(batch_size, input_dim_x)
    y = torch.randn(batch_size, input_dim_y)
    z = torch.randn(batch_size, input_dim_z)


    # 测试 SumFusion
    sum_fusion = SumFusion(output_dim=4)
    _,_,_,sum_output = sum_fusion(x, y, z)
    print("SumFusion output shape:", sum_output.shape)

    # 测试 ConcatFusion
    concat_fusion = ConcatFusion(output_dim=4)
    _,_,_,concat_output = concat_fusion(x, y, z)
    print("ConcatFusion output shape:", concat_output.shape)

    # 测试 FiLM
    film = FilmFusion(output_dim=4)
    _,_,_,film_output = film(x, y, z)
    print("FiLM output shape:", film_output.shape)


    LRF_fusion = LowRankFusion(output_dim = 3)
    _,_,_,lrf_output = LRF_fusion(x, y, z)
    # Check the output
    print(f"LRF features shape: {lrf_output.shape}")

    CF_fusion1 = CrossFusion(output_dim = 3)
    _,_,_,cf_output1 = CF_fusion1(x, y, z)
    # Check the output
    print(f"CF features shape: {cf_output1.shape}")

    CF_fusion2 = CrossFusion2(output_dim = 3)
    _,_,_,cf_output2 = CF_fusion2(x, y, z)
    # Check the output
    print(f"CF features shape: {cf_output2.shape}")