import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import AutoModel, AutoTokenizer

def save_weights(model, save_path='fc_weights.npy'):
    import numpy as np
    weights = model.fc_out.weight.data.cpu().numpy()
    np.save(save_path, weights)
    print(f"Weights saved to {save_path}")

def analyze_fc_weights_gaussian(weights_path='fc_weights.npy', save_path='feature_importance_gaussian.png'):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    weights = np.load(weights_path)
    x_dim = 512
    feature_dims = {
        'Esophagus': (0, x_dim),
        'Liver': (x_dim, 2*x_dim),
        'Spleen': (2*x_dim, 3*x_dim),
        'CLIP Image': (3*x_dim, 3*x_dim+64),
        'CLIP Text': (3*x_dim+64, 3*x_dim+128)
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    
    distributions = {}
    importance_scores = {}
    importance_stds = {}
    
    all_weights = []
    for start_idx, end_idx in feature_dims.values():
        feature_weights = weights[:, start_idx:end_idx].flatten()
        all_weights.extend(feature_weights)
    
    weight_range = (np.min(all_weights), np.max(all_weights))
    x_range = np.linspace(weight_range[0], weight_range[1], 200)
    
    for (feature_name, (start_idx, end_idx)), color in zip(feature_dims.items(), colors):
        feature_weights = weights[:, start_idx:end_idx].flatten()
        
        kernel = stats.gaussian_kde(feature_weights)
        kde = kernel(x_range)
        kde = kde / np.max(kde)
        
        ax1.plot(x_range, kde, label=feature_name, color=color, linewidth=2)
        ax1.fill_between(x_range, kde, alpha=0.15, color=color)
        
        importance = np.linalg.norm(feature_weights)
        importance_scores[feature_name] = importance
        importance_stds[feature_name] = np.std(feature_weights)
        distributions[feature_name] = {'weights': feature_weights, 'kde': kde}
    
    ax1.set_title('Feature Weight Distributions', fontsize=14, pad=20, fontweight='bold')
    ax1.set_xlabel('Weight Values', fontsize=12)
    ax1.set_ylabel('Normalized Density', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_facecolor('#f8f9fa')
    
    x_pos = np.arange(len(importance_scores))
    bars = ax2.bar(x_pos, list(importance_scores.values()), 
                  color=colors, edgecolor='black', linewidth=1, width=0.7,
                  yerr=list(importance_stds.values()), capsize=5,
                  error_kw=dict(elinewidth=2, capthick=2, alpha=0.7))
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    ax2.set_title('Feature Importance with Standard Deviation', 
                 fontsize=14, pad=20, fontweight='bold')
    ax2.set_ylabel('Importance Score', fontsize=12) 
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(list(importance_scores.keys()), rotation=30, ha='right')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return importance_scores, distributions

class FeatureRecalibration(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim//4, 1),
            nn.ReLU(),
            nn.Conv3d(dim//4, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        weight = self.fc(x)
        return x * weight

def attempt_import_uni_former():
    try:
        from FusionLayers.CrossBlock import StageComponents1
        from FusionLayers.LastFusion import SumFusion, ConcatFusion, FilmFusion, LowRankFusion
        from FusionLayers.UniformerChange import UniFormerChange 
    except ModuleNotFoundError:
        from .FusionLayers.CrossBlock import StageComponents1
        from .FusionLayers.LastFusion import SumFusion, ConcatFusion, FilmFusion, LowRankFusion
        from .FusionLayers.UniformerChange import UniFormerChange

    return UniFormerChange, StageComponents1, SumFusion, ConcatFusion, FilmFusion, LowRankFusion

UniFormer, StageComponents, SumFusion, ConcatFusion, FilmFusion, LowRankFusion = attempt_import_uni_former()

class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.proj(x)

class MultiOrganInter(nn.Module):
    def __init__(self, args, num_organs=3, pretrained_paths=None, **kwargs):
        super().__init__()
        
        self.num_organs = num_organs
        self.typeloss = args.typeloss
        embed_dim = args.embed_dims
        depth = args.depths
        num_classes = args.num_classes
        
        self.alignment_criterion = nn.CosineSimilarity(dim=1)
        self.lambda_align = 0.1
        
        self._init_clip()
        self._init_projectors() 
        self._init_organs(num_classes, embed_dim, depth, **kwargs)
        self._init_fusion_layers(embed_dim)
        self._init_embeddings()
        
        self.multi_organ_head = self._create_fusion_head(args.fusion_method, num_classes)
        
        if pretrained_paths:
            self._load_pretrained_weights(pretrained_paths)
            
    def _init_clip(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            model_max_length=512, 
            padding_side="right",
            use_fast=False
        )
        
        model = AutoModel.from_pretrained(
            "GoodBaiBai88/M3D-CLIP",
            trust_remote_code=True  
        )
        self.clip_model = model.to('cuda')
        
    def _init_projectors(self):
        self.clip_projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64)
        )
        
        self.clip_txt_projector = nn.Sequential(
            nn.Linear(768, 256), 
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64)
        )
        
        self.label_encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(), 
            nn.Linear(64, 128)
        )
        
    def _init_organs(self, num_classes, embed_dim, depth, **kwargs):
        self.uni_formers = nn.ModuleDict({
            organ: UniFormer(
                in_chans=1,
                num_classes=num_classes,
                embed_dim=embed_dim,
                depth=depth,
                **kwargs
            ) for organ in ['esophagus', 'liver', 'spleen']
        })
        
    def _init_fusion_layers(self, embed_dim):
        target_sizes = [(8,8,8), (6,6,6), (4,4,4), (2,2,2)]
        self.transfusions = nn.ModuleList([
            StageComponents(embed_dim=dim, target_img_size=size)
            for dim, size in zip(embed_dim, target_sizes)
        ])
        
    def _init_embeddings(self):
        self.head_embed_liver = head_embedding(1, 1)
        self.head_embed_spleen = head_embedding(1, 1)
        
    def _create_fusion_head(self, fusion_method, output_dim):
        fusion_heads = {
            'concat': ConcatFusion,
            'sum': SumFusion,
            'film': FilmFusion,
            'lrf': lambda: LowRankFusion(output_dim=output_dim, rank=output_dim)
        }
        
        if fusion_method not in fusion_heads:
            raise ValueError(f'Invalid fusion method: {fusion_method}')
            
        return fusion_heads[fusion_method](output_dim=output_dim)
        
    def _load_pretrained_weights(self, pretrained_paths):
        for organ, path in pretrained_paths.items():
            if path and organ in self.uni_formers:
                state_dict = torch.load(path)
                model_state = self.uni_formers[organ].state_dict()
                
                compatible_weights = {
                    k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape
                }
                
                self.uni_formers[organ].load_state_dict(compatible_weights, strict=False)
                
    def get_label_features(self, text_batch):
        size_mapping = {
            "very small": 0, "small": 1, "average": 2, 
            "large": 3, "very large": 4
        }
        
        ratio_mapping = {
            "very low": 0, "low": 1, "normal": 2,
            "high": 3, "very high": 4  
        }
        
        batch_size = len(text_batch)
        num_classes = 5
        batch_features = torch.zeros((batch_size, num_classes * 3),
                                   dtype=torch.float32).to('cuda')
        
        for batch_idx, text in enumerate(text_batch):
            text = text.lower()
            organs = {"liver": None, "spleen": None}
            
            for i, organ in enumerate(organs):
                if organ in text:
                    for size, label in size_mapping.items():
                        if size in text[text.index(organ)-20:text.index(organ)+20]:
                            organs[organ] = label
                            break
                    organs[organ] = organs[organ] if organs[organ] is not None else 2
                
                offset = i * num_classes
                batch_features[batch_idx, offset + organs[organ]] = 1.0
            
            ratio_desc = 2
            if "very high" in text:
                ratio_desc = 4
            elif "high" in text:
                ratio_desc = 3
            elif "very low" in text:
                ratio_desc = 0
            elif "low" in text:
                ratio_desc = 1
                
            ratio_offset = num_classes * 2
            batch_features[batch_idx, ratio_offset + ratio_desc] = 1.0
            
        return self.label_encoder(batch_features)
        
    def forward(self, x_esophagus, x_liver, x_spleen, x_full, x_txt, is_training=False):
        x_liver = self.head_embed_liver(x_liver)
        x_spleen = self.head_embed_spleen(x_spleen)
        
        features = {'esophagus': x_esophagus, 'liver': x_liver, 'spleen': x_spleen}
        
        # Extract features through stages
        for stage in range(1, 5):
            # Extract features
            for organ in features:
                features[organ] = getattr(self.uni_formers[organ], f'forward_feature{stage}')(features[organ])
                
            # Apply fusion
            features['esophagus'], features['liver'], features['spleen'] = \
                self.transfusions[stage-1](features['esophagus'], features['liver'], features['spleen'])
                
        # Final feature extraction
        for organ in features:
            features[organ] = self.uni_formers[organ].forward_last(features[organ])
            
        # Get text features
        with torch.inference_mode():
            text_tensor = self.tokenizer(x_txt, max_length=512, truncation=True, 
                                       padding="max_length", return_tensors="pt")
            text_features = self.clip_model.encode_text(
                text_tensor["input_ids"].to('cuda'),
                text_tensor["attention_mask"].to('cuda')
            )[:, 0]
            
            clip_img_features = self.clip_model.encode_image(x_full)[:, 0]
            
        # Process features
        text_features = self.get_label_features(x_txt)
        clip_img_features = self.clip_projector(clip_img_features)
        
        # Final classification
        x, y, z, multi_organ_output = self.multi_organ_head(
            features['esophagus'],
            features['liver'], 
            features['spleen'],
            text_features
        )
        
        if self.typeloss == 'ordinal':
            multi_organ_output = torch.sigmoid(multi_organ_output)
            
        return x, y, z, multi_organ_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_method', type=str, default='caf')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--embed_dims', type=int, nargs='+', default=[64, 128, 320, 512])
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 4, 8, 3]) 
    parser.add_argument('--typeloss', type=str, default='ordinal')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MultiOrganInter(args).to(device)
    
    batch_size = 1
    x_esophagus = torch.randn(batch_size, 1, 64, 90, 90).to(device)
    x_liver = torch.randn(batch_size, 1, 64, 90, 90).to(device)
    x_spleen = torch.randn(batch_size, 1, 64, 90, 90).to(device)
    
    with torch.no_grad():
        _, _, _, output = model(x_esophagus, x_liver, x_spleen)
        
    print(f"Output shape: {output.shape}")