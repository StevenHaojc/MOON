import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def attempt_import_uni_former():
    try:
        from FusionLayers.CrossBlock import create_fusion_block,StageComponents1
        from FusionLayers.LastFusionWT import SumFusion, ConcatFusion, FilmFusion
        from FusionLayers.UniformerChange import UniFormerChange 
    except ModuleNotFoundError:  
        from .FusionLayers.UniformerChange import UniFormerChange  
        from .FusionLayers.CrossBlock import create_fusion_block,StageComponents1
        from .FusionLayers.LastFusionWT import SumFusion, ConcatFusion, FilmFusion

    return UniFormerChange,create_fusion_block, StageComponents1, SumFusion, ConcatFusion, FilmFusion  


UniFormer, create_fusion_block, StageComponents1, SumFusion, ConcatFusion, FilmFusion = attempt_import_uni_former()


class CrossAttentionModule(nn.Module):
    def __init__(self, q_channels=128, k_channels=320, v_channels=512, hidden_channels=512):
        super(CrossAttentionModule, self).__init__()
        self.conv_q = nn.Conv3d(q_channels, hidden_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn_q = nn.BatchNorm3d(hidden_channels)

        self.conv_k = nn.Conv3d(k_channels, hidden_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn_k = nn.BatchNorm3d(hidden_channels)  

        self.relu = nn.ReLU()

        self.alpha_conv = nn.Conv3d(v_channels, hidden_channels, kernel_size=1, bias=True)
        nn.init.constant_(self.alpha_conv.weight, 1.0)
        nn.init.constant_(self.alpha_conv.bias, 0.0)

    def forward(self, query, key, value):
        query = self.relu(self.bn_q(self.conv_q(query)))
        key = self.relu(self.bn_k(self.conv_k(key)))

        query = F.adaptive_avg_pool3d(query, output_size=value.shape[2:])
        key = F.adaptive_avg_pool3d(key, output_size=value.shape[2:])

        query_flat = query.view(query.size(0), -1, query.size(1))  # [B, N, C]
        key_flat = key.view(key.size(0), -1, key.size(1))          # [B, N, C]
        value_flat = value.view(value.size(0), -1, value.size(1))  # [B, N, C]

        attention_scores = torch.matmul(query_flat, key_flat.transpose(-2, -1))  # [B, N, N]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, N, N]

        output_flat = torch.matmul(attention_weights, value_flat)  # [B, N, C]

        output = output_flat.view(value.size())
        output = value + self.alpha_conv(output)
        return output

class MultiOrganWeight(nn.Module):
    def __init__(self, args, num_organs=3, pretrained_paths=None,  **kwargs):
        super().__init__()

        self.num_organs = num_organs
        self.typeloss = args.typeloss
        embed_dim = args.embed_dims
        depth = args.depths
        num_classes = args.num_classes

        self.uni_formers = nn.ModuleDict({
            'esophagus': UniFormer(in_chans=1,num_classes=num_classes, embed_dim=embed_dim, depth=depth, **kwargs),
            'liver': UniFormer(in_chans=1,num_classes=num_classes, embed_dim=embed_dim, depth=depth, **kwargs),
            'spleen': UniFormer(in_chans=1,num_classes=num_classes, embed_dim=embed_dim, depth=depth, **kwargs)
        })

        self.transfusion1 = StageComponents1(embed_dim=embed_dim[0],target_img_size = (8, 8, 8))
        self.transfusion2 = StageComponents1(embed_dim=embed_dim[1],target_img_size = (6, 6, 6))
        self.transfusion3 = StageComponents1(embed_dim=embed_dim[2],target_img_size = (4, 4, 4))
        self.transfusion4 = StageComponents1(embed_dim=embed_dim[3],target_img_size = (2, 2, 2))

        self.cross_att1 = CrossAttentionModule(q_channels=embed_dim[1], k_channels=embed_dim[2], v_channels=embed_dim[3], hidden_channels = embed_dim[3])
        self.cross_att2 = CrossAttentionModule(q_channels=embed_dim[1], k_channels=embed_dim[2], v_channels=embed_dim[3], hidden_channels = embed_dim[3])
        self.cross_att3 = CrossAttentionModule(q_channels=embed_dim[1], k_channels=embed_dim[2], v_channels=embed_dim[3], hidden_channels = embed_dim[3])
        
        self.multi_organ_head = self._create_fusion_head(args.fusion_method, num_classes)

        if pretrained_paths is not None:
            for organ, path in pretrained_paths.items():
                if path is not None and organ in self.uni_formers:
                    self._load_pretrained_weights(self.uni_formers[organ], path)
    def _create_fusion_head(self, fusion_method, output_dim):
        if fusion_method == 'concat':
            return ConcatFusion(output_dim=output_dim)
        elif fusion_method == 'sum':
            return SumFusion(output_dim=output_dim)
        elif fusion_method == 'film':
            return FilmFusion(output_dim=output_dim)
        else:
            raise NotImplementedError(f'Incorrect fusion method: {fusion_method}!')

    def _load_pretrained_weights(self, model, pretrained_path):
        checkpoint = torch.load(pretrained_path)
        current_model_state_dict = model.state_dict()
        
        compatible_state_dict = {
            k: v for k, v in checkpoint.items()
            if k in current_model_state_dict and current_model_state_dict[k].size() == v.size()
        }
        
        model.load_state_dict(compatible_state_dict, strict=False)

    def forward(self, x_esophagus, x_liver, x_spleen):
        # stage1
        esophagus_feat = self.uni_formers['esophagus'].forward_feature1(x_esophagus)
        liver_feat = self.uni_formers['liver'].forward_feature1(x_liver)
        spleen_feat = self.uni_formers['spleen'].forward_feature1(x_spleen)
        esophagus_feat,liver_feat,spleen_feat = self.transfusion1(esophagus_feat,liver_feat,spleen_feat)

        esophagus_feat = self.uni_formers['esophagus'].forward_feature2(esophagus_feat)
        liver_feat = self.uni_formers['liver'].forward_feature2(liver_feat)
        spleen_feat = self.uni_formers['spleen'].forward_feature2(spleen_feat)
        esophagus_feat,liver_feat,spleen_feat = self.transfusion2(esophagus_feat,liver_feat,spleen_feat)
        esophagus_feat_Q = esophagus_feat
        # liver_feat_Q = liver_feat
        # spleen_feat_Q = spleen_feat

        
        esophagus_feat = self.uni_formers['esophagus'].forward_feature3(esophagus_feat)
        liver_feat = self.uni_formers['liver'].forward_feature3(liver_feat)
        spleen_feat = self.uni_formers['spleen'].forward_feature3(spleen_feat)
        esophagus_feat,liver_feat,spleen_feat = self.transfusion3(esophagus_feat,liver_feat,spleen_feat)
        esophagus_feat_K = esophagus_feat
        # liver_feat_K = liver_feat
        # spleen_feat_K = spleen_feat


        esophagus_feat = self.uni_formers['esophagus'].forward_feature4(esophagus_feat)
        liver_feat = self.uni_formers['liver'].forward_feature4(liver_feat)
        spleen_feat = self.uni_formers['spleen'].forward_feature4(spleen_feat)
        esophagus_feat,liver_feat,spleen_feat = self.transfusion4(esophagus_feat,liver_feat,spleen_feat)
        esophagus_feat_V = esophagus_feat
        # liver_feat_V = liver_feat
        # spleen_feat_V = spleen_feat


        esophagus_feat = self.cross_att1(esophagus_feat_Q, esophagus_feat_K, esophagus_feat_V)
        # liver_feat = self.cross_att2(liver_feat_Q, liver_feat_K, liver_feat_V)
        # spleen_feat = self.cross_att3(spleen_feat_Q, spleen_feat_K, spleen_feat_V)


        esophagus_feat = self.uni_formers['esophagus'].forward_last(esophagus_feat)
        liver_feat = self.uni_formers['liver'].forward_last(liver_feat)
        spleen_feat = self.uni_formers['spleen'].forward_last(spleen_feat)        

        #
        xf, yf, zf, multi_organ_output = self.multi_organ_head(esophagus_feat,liver_feat,spleen_feat)

        if self.typeloss == 'ordinal':
            multi_organ_output = torch.sigmoid(multi_organ_output)        

        return  xf, yf, zf, multi_organ_output




if __name__ == '__main__':
    from argparse import ArgumentParser
    
    def parse_args():
        parser = ArgumentParser(description='Train multi-organ model')
        parser.add_argument('--num_classes', type=int, default=4,
                          help='Number of classes')
        parser.add_argument('--embed_dims', type=int, nargs='+', 
                          default=[64, 128, 320, 512],
                          help='Embedding dimensions for each stage')
        parser.add_argument('--depths', type=int, nargs='+',
                          default=[3, 4, 8, 3], 
                          help='Number of layers for each stage')
        parser.add_argument('--typeloss', type=str, default='ordinal',
                          help='Loss type (model outputs sigmoid if ordinal)')
        parser.add_argument('--fusion_method', type=str, default='sum',
                          choices=['concat', 'sum', 'film', 'gated', 'mfb'],
                          help='Fusion method for multi-organ features')
        return parser.parse_args()

    def test_model(args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Initialize model
        model = MultiOrganWeight(args, pretrained_paths=None).to(device)
        
        # Create random test data
        batch_size = 4
        input_shape = (64, 90, 90)
        test_data = {
            'esophagus': torch.randn(batch_size, 1, *input_shape),
            'liver': torch.randn(batch_size, 1, *input_shape),
            'spleen': torch.randn(batch_size, 1, *input_shape)
        }
        
        # Move data to device
        test_data = {k: v.to(device) for k, v in test_data.items()}
        
        # Run inference
        with torch.no_grad():
            try:
                esophagus_out, liver_out, spleen_out, multi_organ_out = model(
                    test_data['esophagus'],
                    test_data['liver'], 
                    test_data['spleen']
                )
                
                print("\nModel Output Shapes:")
                print(f"Esophagus output: {esophagus_out.shape}")
                print(f"Liver output: {liver_out.shape}") 
                print(f"Spleen output: {spleen_out.shape}")
                print(f"Multi-organ output: {multi_organ_out.shape}")
                
                # Optional: Print output statistics
                print("\nOutput Statistics:")
                print(f"Multi-organ output range: [{multi_organ_out.min():.3f}, {multi_organ_out.max():.3f}]")
                print(f"Multi-organ output mean: {multi_organ_out.mean():.3f}")
                
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                raise

    def main():
        args = parse_args()
        print("\nModel Configuration:")
        print("-" * 50)
        for k, v in vars(args).items():
            print(f"{k:15}: {v}")
        print("-" * 50)
        
        try:
            test_model(args)
        except Exception as e:
            print(f"\nError running model test: {str(e)}")
            import traceback
            traceback.print_exc()

    main()