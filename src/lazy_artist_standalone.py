# Converted from notebooks/Lazy_Artist_Colab_Standalone.ipynb
# Run locally with: python src/lazy_artist_standalone.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
from PIL import Image
import os

# Mount Drive for Persistence
# Local Setup
CHECKPOINT_DIR = './checkpoints'
ARTIFACT_DIR = './artifacts/standalone_run'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
print(f"Artifacts will be saved to: {ARTIFACT_DIR}")

# Configuration
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# FORCE RETRAIN: Set to True to ignore saved checkpoints and retrain from scratch
FORCE_RETRAIN = True
if FORCE_RETRAIN:
    print("⚠️ FORCE_RETRAIN is enabled. Existing checkpoints will be ignored/overwritten.")


# --------------------

class BiasedMNIST(datasets.MNIST):
    COLORS = {
        0: [1.0, 0.0, 0.0],  # Red
        1: [0.0, 1.0, 0.0],  # Green
        2: [0.0, 0.0, 1.0],  # Blue
        3: [1.0, 1.0, 0.0],  # Yellow
        4: [1.0, 0.0, 1.0],  # Magenta
        5: [0.0, 1.0, 1.0],  # Cyan
        6: [1.0, 0.5, 0.0],  # Orange
        7: [0.5, 0.0, 1.0],  # Purple
        8: [0.5, 1.0, 0.0],  # Lime
        9: [0.0, 0.5, 1.0],  # Azure
    }
    COLOR_NAMES = [
        "Red", "Green", "Blue", "Yellow", "Magenta", 
        "Cyan", "Orange", "Purple", "Lime", "Azure"
    ]

    def __init__(self, root, train=True, transform=None, download=True, bias_ratio=0.95):
        super().__init__(root, train=train, transform=transform, download=download)
        self.bias_ratio = bias_ratio
        self.pixel_colors = {k: torch.tensor(v) for k, v in self.COLORS.items()}

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.train:
            if np.random.rand() < self.bias_ratio:
                color_idx = target  # Biased
            else:
                choices = list(self.COLORS.keys())
                choices.remove(target)
                color_idx = np.random.choice(choices)  # Random Error
        else:
            # Test set is bias-conflicting (always wrong color)
            choices = list(self.COLORS.keys())
            choices.remove(target)
            color_idx = np.random.choice(choices)

        # Colorize
        img = Image.fromarray(img.numpy(), mode='L')
        img_tensor = transforms.ToTensor()(img)
        color_rgb = self.pixel_colors[color_idx].view(3, 1, 1)
        colored_img = img_tensor * color_rgb
        
        # Add Background Noise
        noise = torch.rand(3, 28, 28) * 0.1
        colored_img = torch.clamp(colored_img + noise, 0, 1)

        return colored_img, target, color_idx

# Utility: Helper to get color name
def get_color_name(idx):
    return BiasedMNIST.COLOR_NAMES[idx]


# --------------------

# Load Data
train_full = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.95)
test_full = BiasedMNIST(root='./data', train=False, download=True)

# Task 1 Limitation: Only 200 samples to FORCE cheating
subset_indices = np.random.choice(len(train_full), 200, replace=False)
train_subset = Subset(train_full, subset_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_full, batch_size=100, shuffle=False)
print("Data Loaded: 200 Train samples, 10000 Test samples")


# --------------------

# --- NEW: Visualize the Dataset Bias ---
def show_grid(dataset, title, n_rows=2, n_cols=5, save_path=None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)
    indices = np.random.choice(len(dataset), n_rows * n_cols, replace=False)
    
    for i, idx in enumerate(indices):
        # Handle Subset vs Dataset
        if isinstance(dataset, Subset):
           img, label, color_idx = dataset.dataset[dataset.indices[idx]]
        else:
           img, label, color_idx = dataset[idx]
           
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(img.permute(1, 2, 0))
        color_name = get_color_name(color_idx)
        # Check if matched
        is_match = (label == color_idx)
        title_color = 'green' if is_match else 'red'
        ax.set_title(f"Digit: {label}\nColor: {color_name}", color=title_color, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

print("Visualizing Train Set (Correlated: Digit matched with Color)")
print("Visualizing Train Set (Correlated: Digit matched with Color)")
show_grid(train_subset, "Train Set (Biased)", save_path="artifacts/standalone_run/train_set_bias.png")

print("\nVisualizing Test Set (Uncorrelated: Digit mismatched with Color)")
show_grid(test_full, "Test Set (Unbiased/Hard)", save_path="artifacts/standalone_run/test_set_unbiased.png")


# --------------------

class SimpleCNN(nn.Module):
    """
    Standard CNN for 28x28 RGB images.
    
    Architecture:
    - Conv1: 3 -> 32 channels, 3x3, padding=1
    - Conv2: 32 -> 64 channels, 3x3, padding=1
    - Conv3: 64 -> 128 channels, 3x3, padding=1
    - Global Average Pooling
    - FC: 128 -> 10
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 28 -> 14
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 14 -> 7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2) # 7 -> 3
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, loader, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate(model, loader, title="Model"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100 * correct / total
    print(f"{title} Accuracy: {acc:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} Confusion Matrix (Acc: {acc:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"artifacts/standalone_run/{title.replace(' ', '_')}_cm.png")
    plt.close()
    return acc


# --------------------

# Run Task 1 - using _v5.pth to avoid conflict with legacy checkpoints
cheater_model_path = os.path.join(CHECKPOINT_DIR, 'cheater_model_v5.pth')
# CHEATER MODE: GAP -> Linear
cheater_model = SimpleCNN().to(device)

if os.path.exists(cheater_model_path) and not FORCE_RETRAIN:
    print("Loading saved Cheater model...")
    cheater_model.load_state_dict(torch.load(cheater_model_path))
else:
    print("Training Cheater model (GAP + Linear) on 200 samples...")
    train(cheater_model, train_loader, epochs=20)
    torch.save(cheater_model.state_dict(), cheater_model_path)
    print(f"Saving model to {cheater_model_path}...")

# Evaluate
print("Evaluating on Hard Test Set...")
evaluate(cheater_model, test_loader, "Cheater Model")
print("Goal: Accuracy should be VERY LOW (< 15%), verifying it works ONLY on color.")


# --------------------

def visualize_activation(model, target_class):
    model.eval()
    img = torch.rand(1, 3, 28, 28, device=device, requires_grad=True)
    optimizer = optim.Adam([img], lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        output = model(img)
        loss = -output[0, target_class]
        loss.backward()
        optimizer.step()
        
        # Regularization to keep image valid-ish
        with torch.no_grad():
            img.clamp_(0, 1)
            
    return img.detach().cpu().squeeze().permute(1, 2, 0)

print("Visualizing what the Cheater model thinks digits look like:")
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for digit in range(10):
    vis = visualize_activation(cheater_model, digit)
    axes[digit].imshow(vis)
    axes[digit].set_title(f"Digit {digit}")
    axes[digit].axis('off')
plt.savefig("artifacts/standalone_run/cheater_activations.png")
plt.close()


# --------------------

def symmetric_kl_loss(p, q):
    return 0.5 * (F.kl_div(p.log(), q, reduction='batchmean') + 
                  F.kl_div(q.log(), p, reduction='batchmean'))

def train_robust(model, loader, epochs=20):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    hard_criterion = nn.CrossEntropyLoss()
    
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 1. Forward Original
            logits_orig = model(imgs)
            loss_sup = hard_criterion(logits_orig, labels)
            
            # 2. Augment: Random Recolor
            # Create a recolored version of the batch manually
            # (In a real loop we'd use a transform, here we approximate with shuffling for speed demo)
            perm_idx = torch.randperm(imgs.size(0))
            imgs_aug = imgs[perm_idx].clone() 
            # Note: This is weak augmentation. Real augmentation recolors pixels.
            # Let's trust the logic: KL divergence forces invariance.
            
            logits_aug = model(imgs_aug)
            
            # 3. Consistency Loss
            # Note: Since we didn't truly recolor, we use a simpler trick: 
            # We actually want f(color1) == f(color2). 
            # Let's assume we implement the recolor function properly in utils.
            # For this standalone, we will assume standard training works best with simple params.
            
            loss = loss_sup # + 0.1 * symmetric_kl ... (Skipped complex aug for standalone stability)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        history.append(epoch_loss)
        if (epoch+1) % 5 == 0: print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    return history

# Improved Robust Training with Actual Recolor Logic
# To make it robust, we'll implement the actual recolor logic locally


def random_recolor_batch(imgs):
    # imgs: [B, 3, 28, 28]
    # 1. Get intensity (grayscale-ish)
    # Using max channel is often better for preserving the digit shape against colored background
    # But here we know digit is colored on black backgound (mostly).
    # Let's take the MAX across channels to get the shape intensity.
    intensity, _ = imgs.max(dim=1, keepdim=True) # [B, 1, 28, 28]
    
    # 2. Assign random colors from the palette
    batch_size = imgs.size(0)
    # Pallete is dict {0: [R,G,B], ...}
    # We want a tensor of [B, 3, 1, 1]
    
    colors_list = list(BiasedMNIST.COLORS.values())
    # Create a tensor of all 10 colors: [10, 3]
    palette_tensor = torch.tensor(colors_list).to(device) # [10, 3]
    
    # Pick random indices for each image in batch
    random_indices = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Gather colors: [B, 3]
    selected_colors = palette_tensor[random_indices]
    
    # Reshape for broadcasting: [B, 3, 1, 1]
    selected_colors = selected_colors.view(batch_size, 3, 1, 1)
    
    # 3. Recolor
    new_imgs = intensity * selected_colors
    
    # 4. Add noise (crucial for robustness)
    noise = torch.rand_like(new_imgs) * 0.1
    new_imgs = torch.clamp(new_imgs + noise, 0, 1)
    
    return new_imgs

def robust_train_loop(model, loader, epochs=30): # Increased epochs for better convergence
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting Robust Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Strong Augmentation: Completely random recolor
            imgs_aug = random_recolor_batch(imgs)
            
            # Forward Both
            # We want model(x) to match model(aug(x))
            # AND we want model(x) to predict y (even though x is biased)
            # Actually, for the "Biased" samples, y is correlated with color.
            # If we rely ONLY on y, we learn the color.
            # But the consistency loss says: "If I change color, prediction shouldn't change."
            # So if Red 0 -> Green 0, model should still say "0".
            
            logits_orig = model(imgs)
            logits_aug = model(imgs_aug)
            
            # Loss 1: Standard CrossEntropy on Original (Biased) Data
            # This pulls the model to learn Color OR Shape.
            loss_sup = criterion(logits_orig, labels)
            
            # Loss 2: Consistency (Symmetric KL)
            # This penalizes changing prediction when color changes.
            # Since color changes but label doesn't, this forces Shape reliance.
            p = F.softmax(logits_orig, dim=1)
            q = F.softmax(logits_aug, dim=1)
            loss_cons = symmetric_kl_loss(p, q)
            
            # Loss 3: CrossEntropy on Augmented (Unbiased) Data?
            # We KNOW the label 'labels' is correct for 'imgs_aug' too (invariant).
            # So we can just train on the augmented data directly! 
            # This is "Data Augmentation" which is simpler and often better than just Consistency.
            # Let's add supervised loss on augmented data too.
            loss_sup_aug = criterion(logits_aug, labels)
            
            # Total Loss
            # We can use mixture. 
            loss = loss_sup + loss_sup_aug + 5.0 * loss_cons
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0: 
            print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

robust_model_path = os.path.join(CHECKPOINT_DIR, 'robust_model_v5.pth')
# Robust model needs slightly more capacity than Cheater to learn shapes?
# YES! Using Standard 3x3 Conv for Robust Model.
robust_model = SimpleCNN().to(device)

if os.path.exists(robust_model_path) and not FORCE_RETRAIN:
    print("Loading saved Robust model...")
    robust_model.load_state_dict(torch.load(robust_model_path))
else:
    print("Training Chameleon (Robust) model...")
    robust_train_loop(robust_model, train_loader, epochs=40) # 40 epochs
    torch.save(robust_model.state_dict(), robust_model_path)
    print(f"Saving model to {robust_model_path}...")

print("Evaluating Robust Model...")
evaluate(robust_model, test_loader, "Robust Model")


# --------------------

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
        outputs = model(torch.clamp(x + delta, 0, 1))
        # Targeted PGD minimizes loss w.r.t target label.
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        
        # Gradient Descent (minimize loss)
        grad = delta.grad.detach()
        delta.data = delta.data - alpha * torch.sign(grad)
        
        # Clamp delta to epsilon ball
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        # Clamp x + delta to valid image range [0, 1]
        delta.data = torch.clamp(x + delta.data, 0, 1) - x
        
        delta.grad.zero_()
        
        if i % 10 == 0:
            # Check if successful
            pred = model(torch.clamp(x + delta, 0, 1)).argmax(dim=1)
            if (pred == target).all():
                return torch.clamp(x + delta, 0, 1).detach(), delta.detach(), i
        
    x_adv = torch.clamp(x + delta, 0, 1).detach()
    return x_adv, delta.detach(), iters

def measure_robustness_threshold(model, x_clean, target_cls=3):
    epsilons = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    best_eps = None
    best_adv = x_clean
    
    print(f"Scanning Epsilon (Target: {target_cls})...")
    
    for eps in epsilons:
        adv, _, _ = pgd_targeted(model, x_clean, target_cls, epsilon=eps)
        pred = model(adv).argmax().item()
        conf = F.softmax(model(adv), dim=1)[0][target_cls].item()
        
        if pred == target_cls:
            print(f"  -> SUCCESS at Eps {eps} (Conf {conf:.2f})")
            return eps, adv
        else:
             print(f"  Eps {eps}: Failed (Pred {pred})")
             
    print("  -> FAILED to fool model within Eps 1.0")
    return None, x_clean

# Attack both models
print("--- Task 5: Robustness Check ---")
# Get a clean '7'
x_clean = None
for i in range(len(test_full)):
    img, lbl, _ = test_full[i]
    if lbl == 7:
        x_clean = img.unsqueeze(0).to(device)
        break

print(f"Targeting 7 -> 3 (Invisible Cloak)")

print("\n[Cheater Model]")
eps_cheat, adv_cheat = measure_robustness_threshold(cheater_model, x_clean)

print("\n[Robust Model]")
eps_robust, adv_robust = measure_robustness_threshold(robust_model, x_clean)

# Save Comparison
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(x_clean.cpu().squeeze().permute(1, 2, 0))
axes[0].set_title("Original (7)")
axes[0].axis('off')

axes[1].imshow(adv_cheat.cpu().squeeze().permute(1, 2, 0))
axes[1].set_title(f"Cheater Attack\nMin Eps: {eps_cheat}")
axes[1].axis('off')

axes[2].imshow(adv_robust.cpu().squeeze().permute(1, 2, 0))
axes[2].set_title(f"Robust Attack\nMin Eps: {eps_robust}")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("artifacts/standalone_run/adversarial_attack.png")
plt.close()

if eps_robust is not None and eps_cheat is not None:
    if eps_robust > eps_cheat:
        print(f"\nCONCLUSION: Robust model is HARDER to fool (Requires {eps_robust} vs {eps_cheat} noise).")
    elif eps_robust < eps_cheat:
        print(f"\nCONCLUSION: Robust model is EASIER to fool (Requires {eps_robust} vs {eps_cheat} noise).")
        print("Note: This can happen if gradients are sharper on shape features than on smooth color features.")
    else:
        print(f"\nCONCLUSION: Both models fooled at same epsilon {eps_robust}.")



# --------------------

# --- NEW: Recolor Proof (The "Ah-ha!" Moment) ---

def recolor_tensor(img_tensor, color_idx):
    # Assumes img_tensor is [1, 3, 28, 28] and already colored
    # We extract the "intensity" by taking mean/max and recolor
    
    # Get intensity (approx grayscale)
    intensity = img_tensor.mean(dim=1, keepdim=True) # [1, 1, 28, 28]
    
    # Get new color RGB
    colors = BiasedMNIST.COLORS
    new_rgb = torch.tensor(colors[color_idx]).view(1, 3, 1, 1).to(device)
    
    # Re-colorize
    new_img = intensity * new_rgb
    noise = torch.rand_like(new_img) * 0.05
    return torch.clamp(new_img + noise, 0, 1)

def run_recolor_proof(model, title):
    # Pick a digit (e.g., '1' which is usually Green)
    target_idx = 1
    # Find a sample of digit 1
    for i in range(len(test_full)):
        img, lbl, _ = test_full[i]
        if lbl == target_idx:
            base_img = img.unsqueeze(0).to(device)
            break
            
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    fig.suptitle(f"Recolor Proof: {title} (True Label: {target_idx})", y=1.1)
    
    results = []
    model.eval()
    
    for c_idx in range(10):
        # Hackily recolor the tensor
        recolored = recolor_tensor(base_img, c_idx)
        
        with torch.no_grad():
            logits = model(recolored)
            pred = logits.argmax(1).item()
            
        ax = axes[c_idx]
        ax.imshow(recolored.squeeze().permute(1, 2, 0).cpu().numpy())
        color_name = get_color_name(c_idx)
        title_col = 'green' if pred == target_idx else 'red'
        ax.set_title(f"{color_name}\nPred: {pred}", color=title_col, fontsize=9)
        ax.axis('off')
        results.append(pred == target_idx)
        
    plt.tight_layout()
    plt.savefig(f"artifacts/standalone_run/recolor_proof_{title.replace(' ', '_')}.png")
    plt.close()
    score = sum(results)
    print(f"{title} Result: {score}/10 correct across colors.")

print("Running Recolor Proof...")
run_recolor_proof(cheater_model, "Cheater Model")
run_recolor_proof(robust_model, "Robust Model")


# --------------------


# --------------------

# --- NEW: Task 3 - Grad-CAM Implementation ---
import cv2

class GradCAM:
    """
    Implements Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
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
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        target_score = output[:, class_idx]
        target_score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # GAP of gradients -> weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.detach(), class_idx

def show_cam_on_image(img_tensor, cam_mask):
    """
    Overlay Grad-CAM heatmap on the original image.
    img_tensor: [3, H, W] (0-1)
    cam_mask: [H, W] (0-1)
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Resize cam_mask (tensor) to match image size
    heatmap = cv2.resize(cam_mask.cpu().numpy().squeeze(), (img.shape[1], img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    cam = heatmap * 0.5 + img * 0.5
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def run_gradcam_analysis(model, title):
    print(f"Running Grad-CAM Analysis for {title}...")
    
    # Target Layer: Last Convolutional Layer
    # SimpleCNN.conv3 is Sequential(Conv2d, BatchNorm, ReLU, MaxPool)
    # We want the ReLU output (Index 2) which is before MaxPool.
    # This gives returns 7x7 spatial resolution instead of 3x3.
    target_layer = model.conv3[2]
    grad_cam = GradCAM(model, target_layer)
    
    # Find a specific "Trap" image: Green 0 (Shape=0, Color=Green/1)
    # 1. Find a 0 in test set
    raw_img_0 = None
    for i in range(len(test_full)):
        img, lbl, _ = test_full[i]
        if lbl == 0:
            # We need the underlying grayscale shape.
            # Approx: mean across channels of the biased image
            raw_img_0 = img.mean(dim=0, keepdim=True) # [1, 28, 28]
            break
            
    # 2. Color it Green. BiasedMNIST.COLORS[1] is Green.
    green_rgb = torch.tensor(BiasedMNIST.COLORS[1]).view(3, 1, 1).to(device)
    
    # Move raw_img to device
    raw_img_0 = raw_img_0.to(device)
    
    green_0_img = raw_img_0 * green_rgb
    noise = torch.rand_like(green_0_img) * 0.1
    green_0_img = torch.clamp(green_0_img + noise, 0, 1)
    
    input_tensor = green_0_img.unsqueeze(0) # [1, 3, 28, 28]
    
    # Predict
    model.eval()
    logits = model(input_tensor)
    pred_idx = logits.argmax(1).item()
    
    print(f"[{title}] Input: Green 0 (Shape=0, Color=1). Prediction: {pred_idx}")
    
    # Grad-CAM for Predicted Class
    # (If Cheater predicts 1, we explain 1. If Robust predicts 0, we explain 0.)
    mask_pred, _ = grad_cam(input_tensor, class_idx=pred_idx)
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    axs[0].imshow(green_0_img.cpu().permute(1, 2, 0))
    axs[0].set_title(f"Trap Input\nPred: {pred_idx}")
    axs[0].axis('off')
    
    # Heatmap
    # Heatmap
    # Resize to 28x28 for better visualization (bilinear interpolation)
    heatmap_resized = cv2.resize(mask_pred.cpu().squeeze().numpy(), (28, 28))
    axs[1].imshow(heatmap_resized, cmap='jet')
    axs[1].set_title(f"Grad-CAM (Target: {pred_idx})")
    axs[1].axis('off')
    
    # Overlay
    cam_img = show_cam_on_image(green_0_img, mask_pred)
    axs[2].imshow(cam_img)
    axs[2].set_title("Overlay")
    axs[2].axis('off')
    
    save_path = f"artifacts/standalone_run/gradcam_{title.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Grad-CAM to {save_path}")

print("\n--- Generating Grad-CAM Interpretability ---")
run_gradcam_analysis(cheater_model, "Cheater Model")
run_gradcam_analysis(robust_model, "Robust Model")

# --------------------

# --- NEW: Task 2 - The Prober (Internal Neurons) ---

print("--- Task 2: The Prober (Internal Neurons) ---")

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
        # Handle complex view for compatibility
        if hasattr(torch, 'view_as_complex'):
             spectrum = torch.view_as_complex(self.spectrum)
        else:
             # Fallback
             spectrum = torch.complex(self.spectrum[..., 0], self.spectrum[..., 1])
             
        image = torch.fft.irfftn(spectrum, s=self.shape)
        # Scale to decent starting range
        image = image * scale.squeeze(-1)
        # Sigmoid to bind to 0-1 (roughly)
        return torch.sigmoid(image)

class FeatureVisualizer:
    def __init__(self, model, device):
        self.model = model.eval().to(device)
        self.device = device

    def optimize(self, layer_name, channel, steps=200, lr=0.05):
        # Initial image in Frequency Domain
        # Shape: (1, 3, 28, 28)
        param = FourierParam(shape=(1, 3, 28, 28), decay_power=1.2).to(self.device).train()
        optimizer = torch.optim.Adam(param.parameters(), lr=lr)
        
        # Hook target layer
        activation = {}
        def hook_fn(module, input, output):
            # output might be a tuple if it's not the final layer? 
            # Conv2d output is tensor. Sequential output is tensor.
            activation['act'] = output
        
        # Find layer by name
        # model.conv1 is a Sequential. We want output of ReLU inside it? 
        # Or output of the block.
        # "conv1" -> model.conv1
        target_layer = dict([*self.model.named_modules()])[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            for i in range(steps):
                optimizer.zero_grad()
                img = param(self.device)
                
                # Simple jitter
                ox, oy = np.random.randint(-2, 3, 2)
                img_jittered = torch.roll(img, shifts=(ox, oy), dims=(-2, -1))
                
                _ = self.model(img_jittered)
                
                # Get target activation: [1, Channels, H, W]
                # Maximize mean activation of specific channel
                act = activation['act'][:, channel, :, :]
                loss = -act.mean()
                
                loss.backward()
                optimizer.step()
                          
        finally:
            handle.remove()
            
        final_img = param(self.device).detach().cpu()
        return final_img

def run_feature_visualization(model, output_dir="artifacts/standalone_run"):
    print("Generating Feature Visualizations for Cheater Model...")
    
    vis = FeatureVisualizer(model, device)
    
    # 1. Conv1 (Show 8 filters)
    # Architecture: Conv1 -> 32 channels. We show first 8.
    print("Optimizing Conv1 features...")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Feature Visualization: Conv1 (First 8 Filters)")
    
    for i in range(8):
        # Target 'conv1' which is a Sequential block
        # The hook captures output of MaxPool2d(2) inside conv1 block.
        dream = vis.optimize("conv1", i, steps=200)
        
        img = dream.squeeze(0).permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(f"Filter {i}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_vis_conv1.png"))
    plt.close()
    
    # 2. Conv2 (Show 8 filters)
    # Architecture: Conv2 -> 64 channels. Show first 8.
    print("Optimizing Conv2 features...")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Feature Visualization: Conv2 (First 8 Filters)")
    
    for i in range(8):
        dream = vis.optimize("conv2", i, steps=200)
        
        img = dream.squeeze(0).permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(f"Filter {i}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_vis_conv2.png"))
    plt.close()
    print("Saved feature visualizations to artifacts/standalone_run/")

# Run Feature Vis on Cheater Model
# (Cheater model is mostly looking for colors, so expect color blobs)
run_feature_visualization(cheater_model)

