import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.simple_cnn import SimpleCNN
from src.vis_utils import FeatureVisualizer
from src.data.biased_mnist import BiasedMNIST

def save_image(tensor, path):
    """Saves a tensor (C, H, W) as an image."""
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # Normalize to 0-1
    img = (img - img.min()) / (img.max() - img.min())
    plt.imsave(path, img)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # 1. Load/Retrain Model
    print("Initializing Model...")
    model = SimpleCNN().to(device)
    
    # We retrain quickly to ensure we have the "Cheater" model state
    # In a real scenario, we'd load weights.
    print("Re-training Cheater (Quickly) to ensure state...")
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.995)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(2): # Just 2 epochs
        print(f"Epoch {epoch+1}/2")
        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"  Step {i}, Loss: {loss.item():.4f}")
    
    print("Model ready.")
    
    # 2. Setup Visualizer
    visualizer = FeatureVisualizer(model, device)
    output_dir = "artifacts/task2_vis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Experiment A: Conv1 (8 Channels)
    print("\n--- Experiment A: Conv1 Features (8 Channels) ---")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        print(f"Optimizing Conv1 Channel {i}...")
        dream, _ = visualizer.optimize("conv1", i, steps=200)
        
        # Save individual
        save_image(dream, os.path.join(output_dir, f"conv1_ch{i:02d}.png"))
        
        # Plot for composite
        img = dream.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(f"Conv1 Ch {i}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conv1_all.png"))
    plt.close()

    # 4. Experiment B: Conv2 (16 Channels)
    print("\n--- Experiment B: Conv2 Features (16 Channels) ---")
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        print(f"Optimizing Conv2 Channel {i}...")
        dream, _ = visualizer.optimize("conv2", i, steps=300)
        
        # Save individual
        save_image(dream, os.path.join(output_dir, f"conv2_ch{i:02d}.png"))
        
        # Plot for composite
        img = dream.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(f"Conv2 Ch {i}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conv2_all.png"))
    plt.close()
    
    # 5. Experiment C: Polysemanticity (Expanded)
    print("\n--- Experiment C: Polysemanticity (Expanded) ---")
    # Probe a few different channels to see which ones are polysemantic
    target_channels = [0, 5, 10, 12] 
    
    for channel in target_channels:
        print(f"Probing Polysemanticity for Conv2 Channel {channel}...")
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        for run in range(5):
            # We want different random initializations. 
            # The FeatureVisualizer initializes randomly in `optimize` method.
            dream, _ = visualizer.optimize("conv2", channel, steps=300)
            
            save_image(dream, os.path.join(output_dir, f"poly_ch{channel}_run{run}.png"))
            
            img = dream.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            
            ax = axes[run]
            ax.imshow(img)
            ax.set_title(f"Run {run}")
            ax.axis('off')
            
        plt.suptitle(f"Polysemanticity Probe: Conv2 Channel {channel}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"poly_ch{channel}_all.png"))
        plt.close()

    print(f"\nAll experiments done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
