import torch
import os
import numpy as np
import imageio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from read_CTdaraCor import read_and_process_CTdara, prediction2label
from Uniformer.MultiOrganInter import MultiOrganInter
from Uniformer.MultiOrganBase import MultiOrganBase

class GradCAM:
    def __init__(self, model, layer_name, organ_name):
        self.model = model
        self.layer_name = layer_name 
        self.organ_name = organ_name
        self.model.eval()
        self.gradient = None
        self.activation = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activation = out

        def backward_hook(module, grad_in, grad_out): 
            self.gradient = grad_out[0]

        organ_model = self.model.uni_formers[self.organ_name]
        layers = dict(organ_model.named_modules())
        target_layer = layers[self.layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, *inputs):
        self.gradient = None
        self.activation = None
        
        _, _, _, output = self.model(*inputs)
        
        if self.activation is None:
            raise ValueError(f"No activations captured for {self.organ_name}")

        self.model.zero_grad()
        target_class = prediction2label(output)
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        gradient = self.gradient.cpu().data.numpy()[0]
        activation = self.activation.cpu().data.numpy()[0]
        weights = np.mean(gradient, axis=(1, 2, 3))
        
        cam = np.zeros(activation.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activation[i, :, :, :]
            
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

def resize_activation(activation, target_size, device):
    if activation.ndim == 3:
        activation = activation[None, None, ...]
    elif activation.ndim == 4:
        activation = activation[None, ...]
    activation_tensor = torch.from_numpy(activation).float().to(device)
    return F.interpolate(activation_tensor, size=target_size, mode='trilinear', align_corners=False)[0]

def generate_grad_cam(device, model, Esc, liver, spleen, target_img, args):
    grad_cam = GradCAM(model, args.target_layer, args.organ_name)
    grad_cam_heatmap = grad_cam(Esc, liver, spleen)
    target_size = target_img.shape[1:]
    return resize_activation(grad_cam_heatmap, target_size, device)

def read_alldata(device, args):
    organs = {
        'esophagus': (40, 40, 100),
        'liver': (256, 196, 36),
        'spleen': (152, 196, 24)
    }
    
    images = {}
    resampled = {}
    
    for organ, shape in organs.items():
        path = join(args.root_path, organ, f"{args.casename}.nii.gz")
        img, res = read_and_process_CTdara(path, target_shape=shape)
        images[organ] = torch.from_numpy(img).unsqueeze(0).to(device).float()
        resampled[organ] = torch.from_numpy(res).unsqueeze(0).to(device).float()
        
    return (images['esophagus'], images['liver'], images['spleen'], 
            resampled['esophagus'], resampled['liver'], resampled['spleen'])

def apply_colormap_on_image(org_img, cam, colormap_name):
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    colormap = plt.get_cmap(colormap_name)
    cam_colored = colormap(cam)[:, :, :3]
    cam_colored = np.uint8(255 * cam_colored)

    org_img_rgb = np.repeat(org_img[:, :, np.newaxis], 3, axis=2)
    org_img_rgb = np.uint8((org_img_rgb - np.min(org_img_rgb)) / 
                          (np.max(org_img_rgb) - np.min(org_img_rgb) + 1e-8) * 255)

    overlay = 0.5 * cam_colored + 0.5 * org_img_rgb
    return np.uint8(overlay), np.uint8(cam_colored)

def save_gradcam_gif(args, org_img_tensor, grad_cam_tensor, direction, colormap_name='rocket'):
    org_img_tensor = org_img_tensor.cpu()
    grad_cam_tensor = grad_cam_tensor.cpu()
    
    if direction == 'axial':
        slices = [(0, t) for t in range(org_img_tensor.size(1))]
    elif direction == 'cor':
        slices = [(slice(None), slice(None), t) for t in range(org_img_tensor.size(2))]
    elif direction == 'sag':
        slices = [(slice(None), slice(None), t) for t in range(org_img_tensor.size(3))]
        
    frames = []
    for idx in slices:
        org_slice = org_img_tensor[0][idx].numpy()
        cam_slice = grad_cam_tensor[0][idx].numpy()
        
        if direction in ['cor', 'sag']:
            org_slice = np.flipud(org_slice)
            cam_slice = np.flipud(cam_slice)
            
        overlay, _ = apply_colormap_on_image(org_slice, cam_slice, colormap_name)
        frames.append(overlay)
        
    output_dir = join(args.gitpath, args.model_use, args.organ_name, args.casename)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = join(output_dir, 
                      f"{args.organ_name}_{args.casename}_{args.target_layer}_{direction}.gif")
    imageio.mimsave(output_path, frames, duration=0.5, loop=0)

def grad_cam_3d_gif(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_class = MultiOrganBase if args.model_use == 'Base' else MultiOrganInter
    model = model_class(args, pretrained_paths=None).to(device)
    
    model_path = join(args.model_root, args.model_use, args.fusion_method, 
                     args.foldid, 'best_model_Val_loss.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    images = read_alldata(device, args)
    x_esophagus, x_liver, x_spleen = images[:3]
    resampled_images = images[3:]
    
    target_img = resampled_images[['esophagus', 'liver', 'spleen'].index(args.organ_name)]
    grad_cam_tensor = generate_grad_cam(device, model, x_esophagus, x_liver, x_spleen, 
                                      target_img, args)
    
    save_gradcam_gif(args, target_img, grad_cam_tensor, args.direction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--model_use', default='Base')
    parser.add_argument('--foldid', default='fold3')
    parser.add_argument('--fusion_method', default='lrf')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--embed_dims', default=[64, 128, 320, 512])
    parser.add_argument('--depths', default=[3, 4, 8, 3])
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--casename', required=True)
    parser.add_argument('--target_layer', required=True)
    parser.add_argument('--organ_name', required=True)
    parser.add_argument('--direction', required=True)
    parser.add_argument('--model_root', required=True)
    parser.add_argument('--gitpath', required=True)
    
    grad_cam_3d_gif(parser.parse_args())