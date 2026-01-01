import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

def import_fusion_modules():
    """Import fusion modules handling different import paths"""
    try:
        from FusionLayers.CrossBlock import create_fusion_block, StageComponents1
        from FusionLayers.LastFusionWT import SumFusion, ConcatFusion, FilmFusion
        from FusionLayers.UniformerChange import UniFormerChange
    except ModuleNotFoundError:
        from .FusionLayers.CrossBlock import create_fusion_block, StageComponents1  
        from .FusionLayers.LastFusionWT import SumFusion, ConcatFusion, FilmFusion
        from .FusionLayers.UniformerChange import UniFormerChange
    
    return (UniFormerChange, create_fusion_block, StageComponents1, 
            SumFusion, ConcatFusion, FilmFusion)

# Import modules
(UniFormer, create_fusion_block, StageComponents1, 
 SumFusion, ConcatFusion, FilmFusion) = import_fusion_modules()

class MultiOrganCross(nn.Module):
    """Multi-organ classification model with cross-attention fusion"""
    
    def __init__(self, 
                 args,
                 num_organs: int = 3,
                 pretrained_paths: Optional[Dict[str, str]] = None,
                 **kwargs):
        super().__init__()
        
        self.num_organs = num_organs
        self.typeloss = args.typeloss
        
        # Create UniFormer models for each organ
        self.uni_formers = nn.ModuleDict({
            organ: UniFormer(
                in_chans=1,
                num_classes=args.num_classes,
                embed_dim=args.embed_dims,
                depth=args.depths,
                **kwargs
            ) for organ in ['esophagus', 'liver', 'spleen']
        })

        # Create fusion components
        target_sizes = [(8,8,8), (6,6,6), (4,4,4), (2,2,2)]
        self.transfusions = nn.ModuleList([
            StageComponents1(embed_dim=dim, target_img_size=size)
            for dim, size in zip(args.embed_dims, target_sizes)
        ])

        # Create fusion head
        self.multi_organ_head = self._create_fusion_head(
            args.fusion_method, 
            args.num_classes
        )

        # Load pretrained weights if provided
        if pretrained_paths:
            self._load_pretrained_weights(pretrained_paths)

    def _create_fusion_head(self, fusion_method: str, output_dim: int) -> nn.Module:
        """Create appropriate fusion head based on method"""
        fusion_heads = {
            'concat': ConcatFusion,
            'sum': SumFusion,
            'film': FilmFusion
        }
        
        if fusion_method not in fusion_heads:
            raise ValueError(f'Invalid fusion method: {fusion_method}')
            
        return fusion_heads[fusion_method](output_dim=output_dim)

    def _load_pretrained_weights(self, pretrained_paths: Dict[str, str]):
        """Load pretrained weights for each organ's model"""
        for organ, path in pretrained_paths.items():
            if path and organ in self.uni_formers:
                state_dict = torch.load(path, map_location='cpu')
                model_state = self.uni_formers[organ].state_dict()
                
                # Filter compatible weights
                compatible_weights = {
                    k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape
                }
                
                self.uni_formers[organ].load_state_dict(compatible_weights, strict=False)

    def forward(self, x_esophagus: torch.Tensor, x_liver: torch.Tensor, 
                x_spleen: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the network"""
        
        features = {
            'esophagus': x_esophagus,
            'liver': x_liver, 
            'spleen': x_spleen
        }
        
        # Process through each stage
        for stage in range(1, 5):
            for organ in features:
                feat_method = f'forward_feature{stage}'
                features[organ] = getattr(self.uni_formers[organ], feat_method)(features[organ])
                
        # Final processing
        for organ in features:
            features[organ] = self.uni_formers[organ].forward_last(features[organ])
            
        # Multi-organ fusion
        xf, yf, zf, output = self.multi_organ_head(
            features['esophagus'],
            features['liver'],
            features['spleen']
        )
        
        if self.typeloss == 'ordinal':
            output = torch.sigmoid(output)
            
        return xf, yf, zf, output

def test_model(args):
    """Test the multi-organ model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = MultiOrganCross(args).to(device)
    
    # Create test data
    test_data = {
        'esophagus': torch.randn(1, 1, 64, 90, 90),
        'liver': torch.randn(1, 1, 64, 90, 90),
        'spleen': torch.randn(1, 1, 64, 90, 90)
    }
    test_data = {k: v.to(device) for k, v in test_data.items()}
    
    # Run inference
    with torch.no_grad():
        try:
            outputs = model(**test_data)
            for i, output in enumerate(outputs):
                print(f"Output {i} shape:", output.shape)
                print(f"Output {i} range: [{output.min():.3f}, {output.max():.3f}]")
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--embed_dims', nargs='+', type=int, 
                       default=[64, 128, 320, 512])
    parser.add_argument('--depths', nargs='+', type=int,
                       default=[3, 4, 8, 3])
    parser.add_argument('--typeloss', choices=['ordinal', 'ce'],
                       default='ordinal')
    parser.add_argument('--fusion_method', 
                       choices=['concat', 'sum', 'film'],
                       default='sum')
    
    args = parser.parse_args()
    
    test_model(args)