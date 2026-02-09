import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# ==========================================
# 0. Configuration
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Dataset Generation (BiasedMNIST)
# ==========================================
class BiasedMNIST(datasets.MNIST):
    """
    A variant of MNIST where the foreground color is spuriously correlated with the class label.
    """
    COLORS = {
        0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0], 2: [0.0, 0.0, 1.0], 3: [1.0, 1.0, 0.0],
        4: [1.0, 0.0, 1.0], 5: [0.0, 1.0, 1.0], 6: [1.0, 0.5, 0.0], 7: [0.5, 0.0, 1.0],
        8: [0.5, 1.0, 0.0], 9: [0.0, 0.5, 1.0],
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, bias_ratio=0.95, background_noise_level=0.1):
        super(BiasedMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.bias_ratio = bias_ratio
        self.background_noise_level = background_noise_level
        self.pixel_colors = {k: torch.tensor(v) for k, v in self.COLORS.items()}

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        # Determine Color Logic
        if self.train:
            if np.random.rand() < self.bias_ratio:
                # Spurious Correlation
                color_idx = target
            else:
                # Random noise color
                choices = list(self.COLORS.keys())
                choices.remove(target)
                color_idx = np.random.choice(choices)
        else:
            # Test set: Bias-Conflicting (Always "wrong" color)
            choices = list(self.COLORS.keys())
            choices.remove(target)
            color_idx = np.random.choice(choices)

        # Apply Color
        img = Image.fromarray(img.numpy(), mode='L')
        # We handle conversion manually
        img_tensor = transforms.ToTensor()(img) # [1, 28, 28]
        color_rgb = self.pixel_colors[color_idx].view(3, 1, 1) # [3, 1, 1]
        
        colored_foreground = img_tensor * color_rgb 
        
        # Add Background Noise
        if self.background_noise_level > 0:
            noise = torch.rand(3, 28, 28)
            colored_img = colored_foreground + (1 - img_tensor) * noise * self.background_noise_level
        else:
            colored_img = colored_foreground

        colored_img = torch.clamp(colored_img, 0, 1)
        
        if self.transform is not None:
            # Re-convert to PIL for standard transforms
            colored_pil = transforms.ToPILImage()(colored_img)
            colored_img = self.transform(colored_pil)

        return colored_img, target, color_idx

# ==========================================
# 2. Model Definition (SimpleCNN)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, use_laplacian=False, reduced_capacity=False):
        super(SimpleCNN, self).__init__()
        self.use_laplacian = use_laplacian
        self.reduced_capacity = reduced_capacity
        
        # Optional: First-Layer Intervention (Laplacian Filter)
        if self.use_laplacian:
            # Fixed Laplacian Kernel for Edge Detection
            # We treat RGB channels independently
            self.laplacian = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
            # Standard Laplacian kernel
            kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
            self.laplacian.weight = nn.Parameter(kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1), requires_grad=False)

        if self.reduced_capacity:
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.fc = nn.Linear(16 * 7 * 7, num_classes)
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc = nn.Linear(64 * 3 * 3, num_classes)

    def forward(self, x):
        # 1. First Layer Intervention
        if self.use_laplacian:
            x = self.laplacian(x)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        if not self.reduced_capacity:
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64 * 3 * 3)
            # Note: Hook extraction for Penultimate layer happens here in execution
        else:
            x = x.view(-1, 16 * 7 * 7)
        
        x = self.fc(x)
        return x

# ==========================================
# 3. Training & Evaluation Logic
# ==========================================

def evaluate(model, loader, name):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            # Ensure transforms are applied if dataset returns 3 items
            if isinstance(images, list): images = images[0] 
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100 * correct / total
    print(f"[{name}] Hard Test Set Accuracy: {acc:.2f}%")
    return acc, all_preds, all_labels

def symmetric_kl_loss(p, q, eps=1e-8):
    """
    Symmetric KL divergence between two probability distributions.
    KL(p||q) + KL(q||p)
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=1)
    return (kl_pq + kl_qp).mean()

def extract_digit_mask(img, threshold=0.3):
    """Extract a soft digit mask from a colored image."""
    # Convert to grayscale by taking max across channels
    gray = img.max(dim=0, keepdim=True)[0]
    # Normalize
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    # Soft threshold
    mask = torch.sigmoid((gray - threshold) * 10)
    return mask

def recolor_augment(img, new_color_id=None):
    """Recolor the digit stroke to a random or specified color."""
    COLORS = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0),
        (0.5, 1.0, 0.0), (0.0, 0.5, 1.0),
    ]
    if new_color_id is None:
        new_color_id = np.random.randint(0, len(COLORS))
    
    # Extract digit mask
    mask = extract_digit_mask(img)  # (1, H, W)
    
    # Get new color
    color_rgb = torch.tensor(COLORS[new_color_id], dtype=img.dtype, device=img.device)
    color_rgb = color_rgb.view(3, 1, 1)
    
    # Create colored foreground (intensity modulated by mask)
    foreground = mask * color_rgb
    
    # Keep original background where mask is low
    background = (1 - mask) * img
    
    # Combine
    recolored = torch.clamp(foreground + background, 0, 1)
    return recolored, new_color_id

def train_chameleon(epochs=20):
    """
    Train Chameleon model using Symmetric KL Consistency on Logits.
    
    Key improvements from reference:
    1. Use symmetric KL divergence on softmax outputs (not MSE on features)
    2. Train on FULL dataset (not limited samples)
    3. Use proper recolor augmentation (not just channel permutation)
    4. More epochs (20) for proper convergence
    """
    print("\n--- Training Chameleon (Symmetric KL Consistency) ---")
    print("Method: Enforcing prediction consistency between Original and Recolored images.")
    
    # Dataset - USE FULL DATASET with 95% bias (standard)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, 
                                 bias_ratio=0.95, background_noise_level=0.1, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    print(f"Training on {len(train_dataset)} samples (full dataset)")
    
    model = SimpleCNN(use_laplacian=False, reduced_capacity=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()
    
    history = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward Original
            logits_real = model(images)
            
            # Recolor augment the batch (proper color-invariance training)
            images_aug = []
            for i in range(images.shape[0]):
                recolored, _ = recolor_augment(images[i])
                images_aug.append(recolored)
            images_aug = torch.stack(images_aug).to(device)
            
            # Forward Recolored
            logits_aug = model(images_aug)
            
            # Classification Loss
            loss_cls = criterion_cls(logits_real, labels)
            
            # Consistency Loss: Symmetric KL on softmax outputs
            p1 = F.softmax(logits_real, dim=1)
            p2 = F.softmax(logits_aug, dim=1)
            loss_cons = symmetric_kl_loss(p1, p2)
            
            # Combined loss (weight consistency more heavily)
            loss = loss_cls + 5.0 * loss_cons
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(logits_real.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1f}%")
    
    return model, history

def run_experiment():
    print("Preparing Datasets...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Test Set
    test_dataset = BiasedMNIST(root='./data', train=False, download=True, background_noise_level=0.3, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Storage for results
    results = {}
    
    # --- Experiment A: First Layer Intervention (Laplacian) ---
    print("\n--- Experiment A: First Layer Intervention (Laplacian) ---")
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.99, background_noise_level=0.3, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model_a = SimpleCNN(use_laplacian=True, reduced_capacity=False).to(device)
    optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history_a = []
    model_a.train()
    for epoch in range(15):  # Increased from 5 to 15 for better convergence
        epoch_loss = 0
        for batch in train_loader:
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device)
            optimizer_a.zero_grad()
            outputs = model_a(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_a.step()
            epoch_loss += loss.item()
        history_a.append(epoch_loss/len(train_loader))
        print(f"Epoch {epoch+1} Loss: {history_a[-1]:.4f}")
            
    acc_a, preds_a, targets_a = evaluate(model_a, test_loader, "First Layer (Laplacian)")
    results['Laplacian'] = {'acc': acc_a, 'loss': history_a, 'preds': preds_a, 'targets': targets_a}
    
    # --- Experiment B: Penultimate Layer Intervention (Consistency) ---
    model_b, history_b = train_chameleon()  # Uses default 20 epochs with symmetric KL
    acc_b, preds_b, targets_b = evaluate(model_b, test_loader, "Penultimate Layer (Consistency)")
    results['Consistency'] = {'acc': acc_b, 'loss': history_b, 'preds': preds_b, 'targets': targets_b}
    
    # --- Plotting ---
    print("\nGenerating Plots...")
    output_dir = "artifacts/summary_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Training Curves
    plt.figure(figsize=(10, 5))
    plt.plot(results['Laplacian']['loss'], label='Laplacian', marker='o')
    plt.plot(results['Consistency']['loss'], label='Consistency', marker='s')
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "task4_training_curves.png"))
    print("Saved task4_training_curves.png")
    
    # 2. Confusion Matrices
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_a = confusion_matrix(results['Laplacian']['targets'], results['Laplacian']['preds'])
    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f"Laplacian Intervention (Acc: {acc_a:.2f}%)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    cm_b = confusion_matrix(results['Consistency']['targets'], results['Consistency']['preds'])
    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f"Consistency Intervention (Acc: {acc_b:.2f}%)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task4_confusion_matrices.png"))
    print("Saved task4_confusion_matrices.png")

if __name__ == "__main__":
    run_experiment()
