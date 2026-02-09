import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data.biased_mnist import BiasedMNIST

def verify_dataset():
    # Create artifacts directory
    os.makedirs('artifacts', exist_ok=True)
    
    # Initialize Dataset (Train - Biased)
    print("Initializing BiasedMNIST (Train)...")
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.95)
    
    # Initialize Dataset (Test - Bias-Conflicting)
    print("Initializing BiasedMNIST (Test)...")
    test_dataset = BiasedMNIST(root='./data', train=False, download=True)

    # Get a batch of images
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    data_iter = iter(train_loader)
    images, labels, colors = next(data_iter)
    
    test_iter = iter(test_loader)
    test_images, test_labels, test_colors = next(test_iter)

    # Visualization
    print("Generating verification image...")
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # Train Grid
    grid_img = torchvision.utils.make_grid(images[:32], nrow=8, normalize=False)
    axs[0].imshow(grid_img.permute(1, 2, 0))
    axs[0].set_title("Training Set (95% Biased) - Digits should match their Color (e.g. 0=Red)")
    axs[0].axis('off')
    
    # Test Grid
    test_grid_img = torchvision.utils.make_grid(test_images[:32], nrow=8, normalize=False)
    axs[1].imshow(test_grid_img.permute(1, 2, 0))
    axs[1].set_title("Test Set (Bias-Conflicting) - Digits should predominantly NOT match their 'trained' color")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('artifacts/dataset_verification.png')
    print("Verification image saved to artifacts/dataset_verification.png")

if __name__ == "__main__":
    verify_dataset()
