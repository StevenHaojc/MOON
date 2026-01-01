import torch
import torch.nn as nn
from mmaction.models import UniFormer
from mmaction.utils import register_all_modules

class SingleOrganUniFormer(nn.Module):
    def __init__(self, args, pretrained=True):
        super().__init__()
        self.typeloss = args.typeloss
        self.organ = args.organ
        
        register_all_modules()
        
        # Configure backbone
        self.backbone = UniFormer(
            depth=[5, 8, 20, 7],
            img_size=160, 
            in_chans=1,
            embed_dim=[64, 128, 320, 512],
            head_dim=64,
            drop_path_rate=0.3,
            pretrained=None,
            pretrained2d=False
        )

        if pretrained:
            self._load_pretrained_weights()
        
        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, args.num_classes)
        )

    def _load_pretrained_weights(self):
        pretrained_path = '/nas/xiaoming.zhang/ESCode/xmcode/main/network/models/M3D/Uniformer/uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb_20221219-157c2e66.pth'
        print(f"Loading pretrained weights from: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Remove 'backbone.' prefix 
        new_state_dict = {k[9:] if k.startswith('backbone.') else k: v 
                         for k, v in state_dict.items()}

        # Handle first conv layer for single channel input
        if 'patch_embed1.proj.weight' in new_state_dict:
            conv_weight = new_state_dict['patch_embed1.proj.weight']
            new_state_dict['patch_embed1.proj.weight'] = conv_weight.mean(dim=1, keepdim=True)

        msg = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (got {x.dim()}D input)")
            
        x = self.backbone(x)
        x = self.head(x)
        
        if self.typeloss == 'ordinal':
            x = torch.sigmoid(x)
            
        return x

def test_memory_usage(input_sizes, model_args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SingleOrganUniFormer(model_args, pretrained=False).to(device)
    results = []

    for size in input_sizes:
        depth, height, width = size
        x = torch.randn(1, 1, int(depth), int(height), int(width)).to(device)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            with torch.no_grad():
                model(x)
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
                results.append((size, peak_memory))
                print(f'Input size {size}: {peak_memory:.2f} GB')
        except Exception as e:
            print(f'Error processing input size {size}:')
            print(str(e))
            results.append((size, float('nan')))

    return results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test UniFormer memory usage')
    parser.add_argument('--organ', default='liver')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--typeloss', default='ordinal')
    
    args = parser.parse_args()

    test_sizes = [
        [75, 41, 52],
        [91, 50, 63], 
        [76, 110, 60],
        [132, 72, 91],
        [158, 86, 109]
    ]

    results = test_memory_usage(test_sizes, args)
    
    print("\nMemory Usage Summary:")
    print("-" * 50)
    for size, mem in results:
        print(f"Size {size:20} : {mem:6.2f} GB")