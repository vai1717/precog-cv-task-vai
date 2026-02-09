import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Implements Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    Ref: Selvaraju et al. (2017) - https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        # 1. Forward hook to capture feature maps (A_k)
        target_layer.register_forward_hook(self.save_activation)
        # 2. Backward hook to capture gradients (dy/dA_k)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, usually (grad,)
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        Args:
            x: Input tensor (1, C, H, W)
            class_idx: Target class index. If None, uses the highest predicted class.
        """
        self.model.eval()
        
        # 1. Forward Pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # 2. Backward Pass
        self.model.zero_grad()
        
        # We want to maximize the score of the target class y^c
        target_score = output[:, class_idx]
        target_score.backward(retain_graph=True)
        
        # 3. Get Gradients and Activations
        # Gradients: [1, K, H, W]
        # Activations: [1, K, H, W]
        gradients = self.gradients
        activations = self.activations
        
        # 4. Global Average Pooling of Gradients (Neuron Importance Weights alpha_k^c)
        # alpha_k^c = (1/Z) * sum_i sum_j (dy^c / dA_ij^k)
        # We average over H and W dimensions (dims 2 and 3)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # [1, K, 1, 1]
        
        # 5. Weighted Combination of Feature Maps
        # L^c_Grad-CAM = ReLU( sum_k (alpha_k^c * A^k) )
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # [1, 1, H, W]
        
        # 6. Apply ReLU
        # We only care about features that have a positive influence on the class of interest
        cam = F.relu(cam)
        
        # 7. Normalize (Min-Max scaling to 0-1) for visualization
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach(), class_idx

def show_cam_on_image(img_tensor, cam_tensor, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the original image.
    Args:
        img_tensor: Original image tensor (3, H, W), normalized 0-1
        cam_tensor: Grad-CAM result (1, 1, H_map, W_map), normalized 0-1
        alpha: Transparency of heatmap
    """
    # Convert image to numpy [H, W, 3]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Resize CAM to match image size
    cam = cam_tensor.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # Create Heatmap (Jet colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # OpenCV uses BGR, convert to RGB
    heatmap = heatmap[..., ::-1]
    
    # Overlay
    cam_image = heatmap * alpha + img * (1 - alpha)
    cam_image = cam_image / np.max(cam_image)
    
    return np.uint8(255 * cam_image), heatmap
