import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage

def get_fft_scale(h, w, decay_power=1.0):
    d = np.sqrt(
        np.fft.fftfreq(h)[:, None]**2 +
        np.fft.fftfreq(w)[None, :]**2
    )
    scale = 1.0 / np.maximum(d, 1.0 / max(h, w))**decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None]
    return scale

class FourierParam(nn.Module):
    def __init__(self, shape, decay_power=1.0):
        super().__init__()
        self.shape = shape
        h, w = shape[-2], shape[-1]
        self.scale = get_fft_scale(h, w, decay_power)
        # Random initialization in freq domain
        self.spectrum = nn.Parameter(torch.randn(*shape, 2) * 0.01)

    def forward(self, device):
        scale = self.scale.to(device)
        spectrum = torch.view_as_complex(self.spectrum)
        image = torch.fft.irfftn(spectrum, s=self.shape)
        # Scale to decent starting range
        image = image * scale.squeeze(-1)
        # Sigmoid to bind to 0-1 (roughly)
        return torch.sigmoid(image)

class RobustTransforms(nn.Module):
    def __init__(self, jitter_range=2, scale_range=1.05, rotation_range=2):
        super().__init__()
        self.jitter_range = jitter_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range

    def forward(self, img):
        # 1. Jitter (Translation)
        ox, oy = np.random.randint(-self.jitter_range, self.jitter_range+1, 2)
        img = torch.roll(img, shifts=(ox, oy), dims=(-2, -1))
        
        # 2. Scale & Rotate (simplification: just using interpolate for scale)
        # For a full affine, we'd use grid_sample, but for MNIST 28x28, simple is fine.
        # Let's stick to simple jitter for now to avoid blurring tiny digits too much.
        return img

class FeatureVisualizer:
    def __init__(self, model, device):
        self.model = model.eval().to(device)
        self.device = device
        # Jitter: shift image randomly by up to 4 pixels (14% of image size)
        # Scale: scale image by factor of 0.9 to 1.1
        # Rotate: rotate by +/- 5 degrees
        self.transforms = RobustTransforms(jitter_range=4, scale_range=1.1, rotation_range=5)

    def optimize(self, layer_name, channel, steps=500, lr=0.05):
        # Initial image in Frequency Domain
        # Shape: (1, 3, 28, 28)
        # Fourier param init
        param = FourierParam(shape=(1, 3, 28, 28), decay_power=1.2).to(self.device)
        optimizer = torch.optim.Adam(param.parameters(), lr=lr)
        
        history = []
        
        # Hook target layer
        activation = {}
        def hook_fn(module, input, output):
            activation['act'] = output
        
        target_layer = dict([*self.model.named_modules()])[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            for i in range(steps):
                optimizer.zero_grad()
                
                # generate image from spectrum
                img = param(self.device)
                
                # Apply robustness transforms
                # We need a custom transform function that works on tensors with gradients
                # Simple jitter for now:
                ox, oy = np.random.randint(-2, 3, 2)
                img_jittered = torch.roll(img, shifts=(ox, oy), dims=(-2, -1))
                
                # Forward pass
                _ = self.model(img_jittered)
                
                # Get target activation
                # Layer output shape: [1, Channels, H, W]
                # We want mean activation of channel 'channel'
                act = activation['act'][:, channel, :, :]
                loss = -act.mean()
                
                # Regularization
                # 1. L2 Reg (Scalar) to prevent extreme pixel values
                l2_loss = img.pow(2).mean()
                
                # 2. Total Variation (TV) Reg for smoothness
                # Sum of absolute differences between adjacent pixels
                diff_h = torch.abs(img[..., 1:] - img[..., :-1])
                diff_w = torch.abs(img[..., :, 1:] - img[..., :, :-1])
                tv_loss = diff_h.sum() + diff_w.sum()
                
                # Combined Loss: Maximize Activation + Minimize Reg
                # Weights: Activation=1.0, L2=0.01, TV=0.01
                loss = -act.mean() + 0.01 * l2_loss + 0.01 * tv_loss

                
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    with torch.no_grad():
                         # Save un-jittered version
                         current_img = param(self.device).detach().cpu()
                         history.append(current_img)
                         
        finally:
            handle.remove()
            
        final_img = param(self.device).detach().cpu()
        return final_img, history

