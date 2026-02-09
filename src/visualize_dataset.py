import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.biased_mnist import BiasedMNIST

def visualize_samples():
    print("Generating dataset visualization...")
    
    # 1. Load Datasets
    # Train: Biased (Red=0, Green=1, etc.) with Background Noise
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.95, background_noise_level=0.1)
    
    # Test: Bias-Conflicting (Random colors)
    test_dataset = BiasedMNIST(root='./data', train=False, download=True, background_noise_level=0.1)
    
    # 2. Setup Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # 3. Plot Train Samples (Biased)
    axes[0, 2].set_title("Training Set (Biased)\nExp: 0=Red, 1=Green...", fontsize=14)
    indices = [0, 1, 2, 3, 4] # Just take first 5 samples
    for i, idx in enumerate(indices):
        img, label, color_idx = train_dataset[idx]
        # img is Tensor [3, 28, 28] -> [28, 28, 3] for matplotlib
        img_np = img.permute(1, 2, 0).numpy()
        
        ax = axes[0, i]
        ax.imshow(img_np)
        ax.set_title(f"Label: {label}\nColor: {color_idx}")
        ax.axis('off')

    # 4. Plot Test Samples (Bias-Conflicting)
    axes[1, 2].set_title("Test Set (Bias-Conflicting)\nColors Randomized", fontsize=14)
    for i, idx in enumerate(indices):
        img, label, color_idx = test_dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        
        ax = axes[1, i]
        ax.imshow(img_np)
        ax.set_title(f"Label: {label}\nColor: {color_idx}")
        ax.axis('off')

    # 5. Save
    output_path = "artifacts/dataset_visualization.png"
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_samples()
