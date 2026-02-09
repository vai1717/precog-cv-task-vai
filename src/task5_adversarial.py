import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.simple_cnn import SimpleCNN
from src.data.biased_mnist import BiasedMNIST
# Reuse the Chameleon training from Task 4
from src.task4_intervention import train_chameleon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cheater_model():
    """Trains a standard model on biased data (The Cheater)."""
    print("Training Cheater Model...")
    # Standard training, no augmentation, biased data
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.995, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(2): # Quick training, it converges fast on bias
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def pgd_targeted(model, x, target_class, epsilon=0.05, alpha=0.01, iters=50):
    """
    Targeted PGD Attack.
    Goal: Minimize Loss(model(x'), target_class)
    Constraint: ||x - x'||_inf <= epsilon
    """
    model.eval()
    # Start with random perturbation within epsilon ball
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon).to(device)
    delta.requires_grad = True
    
    target = torch.tensor([target_class] * x.shape[0]).to(device)
    
    for i in range(iters):
        # Forward pass
        # We need to clamp x + delta to [0, 1] essentially, 
        # but gradient flow effectively happens through the operation.
        outputs = model(torch.clamp(x + delta, 0, 1))
        
        # We want to MINIMIZE loss w.r.t Target
        # Standard PGD maximizes loss w.r.t true label.
        # Targeted PGD minimizes loss w.r.t target label.
        loss = F.cross_entropy(outputs, target)
        
        loss.backward()
        
        # Gradient Descent (minimize loss)
        # delta = delta - alpha * sign(grad)
        grad = delta.grad.detach()
        delta.data = delta.data - alpha * torch.sign(grad)
        
        # Clamp delta to epsilon ball
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        # Clamp x + delta to valid image range [0, 1]
        # This is a bit tricky with delta, usually we clamp final image
        # x_adv = clamp(x + delta, 0, 1)
        # delta = x_adv - x
        delta.data = torch.clamp(x + delta.data, 0, 1) - x
        
        delta.grad.zero_()
        
        if i % 10 == 0:
            # Check if successful
            pred = model(torch.clamp(x + delta, 0, 1)).argmax(dim=1)
            if (pred == target).all():
                return torch.clamp(x + delta, 0, 1).detach(), delta.detach(), i
        
    x_adv = torch.clamp(x + delta, 0, 1).detach()
    return x_adv, delta.detach(), iters

def get_target_sample(digit=7):
    """Get a clean sample of a digit."""
    dataset = BiasedMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    # Find a '7'
    idx = (dataset.targets == digit).nonzero(as_tuple=True)[0][0]
    img = dataset[idx][0].unsqueeze(0).to(device) # [1, 3, 28, 28]
    return img

def run_attack():
    print("--- Task 5: The Invisible Cloak ---")
    
    # 1. Prepare Models
    cheater_model = get_cheater_model()
    print("Training Robust Model (Chameleon)...")
    robust_model, _ = train_chameleon()
    
    # 2. Prepare Victim Image
    x_clean = get_target_sample(digit=7)
    
    # Check predictions
    with torch.no_grad():
        pred_cheat = cheater_model(x_clean).argmax().item()
        pred_robust = robust_model(x_clean).argmax().item()
    print(f"Original Image (Digit 7). Cheater Pred: {pred_cheat}, Robust Pred: {pred_robust}")
    
    target_cls = 3
    epsilons = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    
    print(f"\nScanning Epsilon for Cheater Model (Target: {target_cls})...")
    eps_cheat = None
    x_adv_cheat = x_clean
    noise_cheat = torch.zeros_like(x_clean)
    
    for eps in epsilons:
        adv, noise = pgd_targeted(cheater_model, x_clean, target_cls, epsilon=eps)
        pred = cheater_model(adv).argmax().item()
        conf = F.softmax(cheater_model(adv), dim=1)[0][target_cls].item()
        print(f"  Eps {eps}: Pred {pred} (Conf {conf:.2f})")
        if pred == target_cls:
            eps_cheat = eps
            x_adv_cheat = adv
            noise_cheat = noise
            print(f"  -> SUCCESS at Eps {eps}")
            break
            
    print(f"\nScanning Epsilon for Robust Model (Target: {target_cls})...")
    eps_robust = None
    x_adv_robust = x_clean
    noise_robust = torch.zeros_like(x_clean)
    
    for eps in epsilons:
        adv, noise = pgd_targeted(robust_model, x_clean, target_cls, epsilon=eps)
        pred = robust_model(adv).argmax().item()
        conf = F.softmax(robust_model(adv), dim=1)[0][target_cls].item()
        print(f"  Eps {eps}: Pred {pred} (Conf {conf:.2f})")
        if pred == target_cls:
            eps_robust = eps
            x_adv_robust = adv
            noise_robust = noise
            print(f"  -> SUCCESS at Eps {eps}")
            break

    # 4. Save Artifacts
    output_dir = "artifacts/task5_adversarial"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Original
    plt.subplot(2, 3, 1)
    plt.imshow(x_clean.cpu().squeeze().permute(1, 2, 0))
    plt.title(f"Original (7)")
    plt.axis('off')
    
    # Cheater Attack
    plt.subplot(2, 3, 2)
    plt.imshow(x_adv_cheat.cpu().squeeze().permute(1, 2, 0))
    plt.title(f"Cheater Attack\nFixed at Eps {eps_cheat}")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    # Amplify noise for visibility
    if eps_cheat:
        plt.imshow((noise_cheat.cpu().squeeze().permute(1, 2, 0).abs() / eps_cheat).clamp(0, 1))
        plt.title(f"Cheater Noise (Norm)")
    plt.axis('off')

    # Robust Attack
    plt.subplot(2, 3, 5)
    plt.imshow(x_adv_robust.cpu().squeeze().permute(1, 2, 0))
    plt.title(f"Robust Attack\nFixed at Eps {eps_robust}")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    if eps_robust:
        plt.imshow((noise_robust.cpu().squeeze().permute(1, 2, 0).abs() / eps_robust).clamp(0, 1))
        plt.title(f"Robust Noise (Norm)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_comparison.png"))
    print("Saved attack_comparison.png")


if __name__ == "__main__":
    run_attack()
