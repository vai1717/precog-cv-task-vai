import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.simple_cnn import SimpleCNN
from src.data.biased_mnist import BiasedMNIST
from src.gradcam_scratch import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_cam_on_image(img_tensor, cam_mask):
    """
    Overlay Grad-CAM heatmap on the original image.
    img_tensor: [3, H, W] (0-1)
    cam_mask: [H, W] (0-1)
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Resize cam_mask to match image size
    cam_mask = cv2.resize(cam_mask, (img.shape[1], img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    cam = heatmap * 0.5 + img * 0.5
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def run_task3():
    print("--- Task 3: The Interrogation (Grad-CAM) ---")
    
    # 1. Train the Cheater Model (Quickly)
    # We need the reduced capacity model that failed in Task 1.
    print("Training reduced-capacity Cheater Model...")
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.999, background_noise_level=0.3)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SimpleCNN(use_laplacian=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")
        
    print("Model trained. Starting Interrogation...")
    model.eval()
    
    # 2. Setup Grad-CAM
    # Target Layer: conv2 (The last conv layer in our reduced architecture)
    target_layer = model.conv2
    grad_cam = GradCAM(model, target_layer)
    
    # 3. Experiment 1: Biased Image (Red 0)
    # 0 is usually Red.
    print("\n Experiment 1: Biased Image (Red 0)")
    # Manually create a Red 0
    test_dataset = BiasedMNIST(root='./data', train=False, download=True, background_noise_level=0.1) # Less noise for viz clarity
    
    # Find a 0
    red_0_img = None
    for i in range(len(test_dataset)):
        img, lbl, color = test_dataset[i]
        if lbl == 0:
            # Force it to be Red (Color 0) just in case test set is random
            # We can use the dataset's color map
            red_color = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
            # Recolor foreground
            # img is already colored with noise. Let's make a synthetic one for purity.
            # Get raw MNIST digit
            raw_img = test_dataset.data[i] # uint8 28x28
            # Convert to float 0-1 directly
            raw_img = raw_img.float() / 255.0
            raw_img = raw_img.view(1, 28, 28) # 0-1
            
            fg = raw_img * red_color
            bg_noise = torch.rand(3, 28, 28) * 0.1
            red_0_img = fg + (1 - raw_img) * bg_noise
            # red_0_img = torch.clamp(red_0_img, 0, 1).to(device) # Moved below to break
            red_0_img = torch.clamp(red_0_img, 0, 1).to(device)
            break
            
    # Visualize
    input_tensor = red_0_img.unsqueeze(0) # [1, 3, 28, 28]
    mask, pred_class = grad_cam(input_tensor, class_idx=0) # Explain Class 0
    
    # Prepare Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    axs[0].imshow(red_0_img.cpu().permute(1, 2, 0))
    axs[0].set_title(f"Input: Red 0\nPred Class: {pred_class}")
    axs[0].axis('off')
    
    # Heatmap
    axs[1].imshow(mask.cpu().squeeze(), cmap='jet')
    axs[1].set_title("Grad-CAM Heatmap\n(Target: Class 0)")
    axs[1].axis('off')
    
    # Overlay
    cam_img = show_cam_on_image(red_0_img, mask.cpu().squeeze().numpy())
    axs[2].imshow(cam_img)
    axs[2].set_title("Overlay")
    axs[2].axis('off')
    
    os.makedirs("artifacts/task3_gradcam", exist_ok=True)
    plt.savefig("artifacts/task3_gradcam/gradcam_biased_red0.png")
    print("Saved artifacts/task3_gradcam/gradcam_biased_red0.png")
    
    # 4. Experiment 2: Conflicting Image (Green 0)
    # 0 Shaped, but Green (Color of 1). The Trap.
    print("\n Experiment 2: Conflicting Image (Green 0)")
    
    green_color = torch.tensor([0.0, 1.0, 0.0]).view(3, 1, 1)
    fg = raw_img * green_color # Same 0 digit shape
    green_0_img = fg + (1 - raw_img) * bg_noise
    green_0_img = torch.clamp(green_0_img, 0, 1).to(device)
    
    input_tensor = green_0_img.unsqueeze(0)
    
    # What does the model predict?
    output = model(input_tensor)
    pred_idx = output.argmax().item()
    print(f"Model Predicted: {pred_idx} (True Shape: 0, Color: Green)")
    
    # We want to ask: "Why did you predict that?" (e.g. 1 or 6)
    mask_pred, _ = grad_cam(input_tensor, class_idx=pred_idx)
    
    # We also want to ask: "Do you see a 0 anywhere?"
    mask_0, _ = grad_cam(input_tensor, class_idx=0)
    
    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    axs[0].imshow(green_0_img.cpu().permute(1, 2, 0))
    axs[0].set_title(f"Input: Green 0\nPred: {pred_idx}")
    axs[0].axis('off')
    
    # Explanation for Predicted Class
    axs[1].imshow(mask_pred.cpu().squeeze(), cmap='jet')
    axs[1].set_title(f"Heatmap (Target: {pred_idx})\nWhy Prediction?")
    axs[1].axis('off')
    
    # Explanation for Class 0
    axs[2].imshow(mask_0.cpu().squeeze(), cmap='jet')
    axs[2].set_title(f"Heatmap (Target: 0)\nWhere is the 0?")
    axs[2].axis('off')

    # Overlay for Prediction
    cam_img = show_cam_on_image(green_0_img, mask_pred.cpu().squeeze().numpy())
    axs[3].imshow(cam_img)
    axs[3].set_title(f"Overlay (Class {pred_idx})")
    axs[3].axis('off')

    plt.savefig("artifacts/task3_gradcam/gradcam_conflicting_green0.png")
    print("Saved artifacts/task3_gradcam/gradcam_conflicting_green0.png")

if __name__ == "__main__":
    run_task3()
