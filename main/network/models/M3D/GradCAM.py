import torch
import numpy as np
import imageio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import join
from read_CTdara import read_and_process_CTdara, prediction2label
from Uniformer.MultiOrgancross import MultiOrgancross

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
        if self.layer_name not in layers:
            raise KeyError(f"Layer {self.layer_name} not found in organ {self.organ_name}.")
        
        target_layer = layers[self.layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, *inputs):
        self.gradient = None
        self.activation = None
        
        output = self.model(*inputs)
        
        if self.activation is None:
            raise ValueError(f"No activations captured for {self.organ_name} at layer {self.layer_name}")

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
    upsampled_activation = F.interpolate(activation_tensor, size=target_size, mode='trilinear', align_corners=False)
    return upsampled_activation[0]

def reduce_channels(data):
    if isinstance(data, np.ndarray):
        return np.mean(data, axis=0, keepdims=True)
    elif isinstance(data, torch.Tensor):
        return data.mean(dim=0, keepdim=True)
    else:
        raise TypeError("Input must be numpy array or pytorch tensor")
    
def generate_grad_cam(device, model, Esc, liver, spleen, args):
    grad_cam = GradCAM(model, args.target_layer, args.organ_name)
    grad_cam_heatmap = grad_cam(Esc, liver, spleen)
    
    target_size = {
        'esophagus': Esc.shape[2:],
        'liver': liver.shape[2:],
        'spleen': spleen.shape[2:]
    }[args.organ_name]
    
    return resize_activation(grad_cam_heatmap, target_size, device)

def read_alldata(device, args):
    paths = {
        'esophagus': (40, 40, 80),
        'liver': (136, 136, 24),
        'spleen': (48, 48, 24)
    }
    
    data = {}
    for organ, shape in paths.items():
        path = join(args.root_path, organ, f"{args.casename}.nii.gz")
        img = read_and_process_CTdara(path, target_shape=shape)
        data[organ] = torch.from_numpy(img).unsqueeze(0).to(device).float()
        
    return data['esophagus'], data['liver'], data['spleen']

def apply_colormap_on_image(org_img, cam, colormap_name):
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    colormap = plt.get_cmap(colormap_name)
    cam_colored = colormap(cam)[:, :, :3]
    cam_colored = np.uint8(255 * cam_colored)

    org_img_rgb = np.repeat(org_img[:, :, np.newaxis], 3, axis=2)
    org_img_rgb = np.uint8((org_img_rgb - np.min(org_img_rgb)) / 
                          (np.max(org_img_rgb) - np.min(org_img_rgb)) * 255)

    overlay = 0.3 * cam_colored + 0.7 * org_img_rgb
    return np.uint8(overlay), np.uint8(cam_colored)

def save_gradcam_gif(args, org_img_tensor, grad_cam_tensor, colormap_name='jet'):
    org_img_tensor = org_img_tensor.cpu()
    grad_cam_tensor = grad_cam_tensor.cpu()
    
    frames = []
    for t in range(org_img_tensor.size(1)):
        org_slice = org_img_tensor[0, t].numpy()
        cam_slice = grad_cam_tensor[0, t].numpy()
        overlay, _ = apply_colormap_on_image(org_slice, cam_slice, colormap_name)
        frames.append(overlay)

    output_path = join(args.gitpath, f"{args.organ_name}_{args.casename}_{args.target_layer}.gif")
    imageio.mimsave(output_path, frames, duration=0.5)

def grad_cam_3d_gif(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiOrgancross(args, pretrained_paths=None).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    x_esophagus, x_liver, x_spleen = read_alldata(device, args)
    grad_cam_tensor = generate_grad_cam(device, model, x_esophagus, x_liver, x_spleen, args)
    
    target_img = {
        'esophagus': x_esophagus,
        'liver': x_liver,
        'spleen': x_spleen
    }[args.organ_name].squeeze(0)
    
    save_gradcam_gif(args, target_img, grad_cam_tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_use', default='cross')
    parser.add_argument('--fusion_method', default='concat')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--embed_dims', default=[64, 128, 320, 512])
    parser.add_argument('--depths', default=[3, 4, 8, 3])
    parser.add_argument('--typeloss', default='ordinal')
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--casename', required=True)
    parser.add_argument('--target_layer', required=True)
    parser.add_argument('--organ_name', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--gitpath', required=True)
    
    args = parser.parse_args()
    grad_cam_3d_gif(args)