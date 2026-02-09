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
    COLORS = {
        0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0], 2: [0.0, 0.0, 1.0], 3: [1.0, 1.0, 0.0],
        4: [1.0, 0.0, 1.0], 5: [0.0, 1.0, 1.0], 6: [1.0, 0.5, 0.0], 7: [0.5, 0.0, 1.0],
        8: [0.5, 1.0, 0.0], 9: [0.0, 0.5, 1.0],
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, bias_ratio=0.95):
        super(BiasedMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.bias_ratio = bias_ratio
        self.pixel_colors = {k: torch.tensor(v) for k, v in self.COLORS.items()}

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        if self.train:
            if np.random.rand() < self.bias_ratio:
                color_idx = target
            else:
                choices = list(self.COLORS.keys())
                choices.remove(target)
                color_idx = np.random.choice(choices)
        else:
            choices = list(self.COLORS.keys())
            choices.remove(target)
            color_idx = np.random.choice(choices)

        img = Image.fromarray(img.numpy(), mode='L')
        img_tensor = transforms.ToTensor()(img)
        color_rgb = self.pixel_colors[color_idx].view(3, 1, 1)
        
        colored_foreground = img_tensor * color_rgb 
        colored_img = torch.clamp(colored_foreground, 0, 1) # Simple version, no noise for SAE task clarity
        
        if self.transform is not None:
            colored_pil = transforms.ToPILImage()(colored_img)
            colored_img = self.transform(colored_pil)

        return colored_img, target, color_idx

# ==========================================
# 2. Model Definition (SimpleCNN - The Cheater)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Standard "Cheater" Architecture (Reduced Capacity)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1) # 8 Filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 16 Filters
        # Output of Conv2 -> Pool is 16 ch x 7x7
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc(x)
        return x

# ==========================================
# 3. Sparse Autoencoder Logic
# ==========================================
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
        # Initialize decoder weights to be unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x):
        # ReLU encoder -> Sparse Code z
        z = F.relu(self.encoder(x))
        # Linear decoder
        x_reconstruct = self.decoder(z)
        return x_reconstruct, z

def train_sae(activations, latent_dim_ratio=4, epochs=10, batch_size=256, l1_lambda=0.01):
    input_dim = activations.shape[1]
    latent_dim = input_dim * latent_dim_ratio
    
    sae = SparseAutoencoder(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    
    dataset = torch.utils.data.TensorDataset(activations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training SAE (Input: {input_dim}, Latent: {latent_dim})...")
    sae.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            x_recon, z = sae(x)
            
            # Loss = MSE + L1
            loss_recon = F.mse_loss(x_recon, x)
            loss_l1 = l1_lambda * torch.norm(z, 1)
            loss = loss_recon + loss_l1
            
            loss.backward()
            optimizer.step()
            
            # Constraint: Keep decoder weights normalized
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

            epoch_loss += loss.item()
            
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}: Loss {epoch_loss/len(loader):.4f}")
        
    return sae

# ==========================================
# 4. Experiment Logic
# ==========================================
def get_activations_conv2(model, loader):
    model.eval()
    activations = []
    print("Collecting activations from Conv2...")
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            # Manual Forward to Conv2
            x = model.pool(F.relu(model.conv1(images)))
            x = model.pool(F.relu(model.conv2(x))) # [B, 16, 7, 7]
            flat = x.view(x.size(0), -1) # [B, 16*7*7] = [B, 784]
            activations.append(flat.cpu())
            if len(activations) * 64 > 3000: break # enough samples
    return torch.cat(activations, dim=0)

def manual_forward_classifier(model, act_flat):
    # Takes flattened activations -> FC -> Output
    return model.fc(act_flat)

def run_experiment():
    print("--- Task 6: Sparse Autoencoder Decomposition (Batch Analysis) ---")
    
    # 1. Train Cheater Model
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.995, transform=transform) # 99.5% Bias
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStep 1: Training the Cheater Model (Bias=0.995)...")
    model.train()
    for img, lbl, _ in train_loader:
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        loss = criterion(model(img), lbl)
        loss.backward()
        optimizer.step()
            
    # 2. Train SAE on Hidden States
    print("\nStep 2: Training SAE on Conv2 Activations...")
    acts = get_activations_conv2(model, train_loader)
    sae = train_sae(acts, latent_dim_ratio=2, epochs=5)
    
    # 3. INTERVENTION with Multiple Samples
    print("\nStep 3: Intervention on Multiple Conflicting Samples...")
    
    test_dataset = BiasedMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Find up to 5 conflicting samples
    targets = []
    
    model.eval()
    for i in range(len(test_dataset)):
        if len(targets) >= 5: break
        
        img, lbl, _ = test_dataset[i]
        with torch.no_grad():
            p = model(img.unsqueeze(0).to(device)).argmax().item()
            
        # Look for failure: Label 0 -> Pred 1 (Confusion with Green/1)
        if lbl == 0 and p == 1:
            targets.append((i, img, lbl, p))
            
    print(f"Found {len(targets)} Conflicting Samples (Label 0, Pred 1).")
    
    for t_idx, (idx, img, lbl, p) in enumerate(targets):
        print(f"\n[Sample {t_idx+1}] Image {idx}: Label {lbl}, Pred {p}")
        target_img = img.unsqueeze(0).to(device)
        
        # Get latent code
        with torch.no_grad():
            x = model.pool(F.relu(model.conv1(target_img)))
            x = model.pool(F.relu(model.conv2(x)))
            act_orig = x.view(1, -1)
            _, z = sae(act_orig)
            
        # --- DIAL UP ---
        found_fix = False
        for feat_idx in range(z.shape[1]):
            z_mod = z.clone()
            z_mod[0, feat_idx] += 20.0
            act_mod = sae.decoder(z_mod)
            p_new = manual_forward_classifier(model, act_mod).argmax().item()
            
            if p_new == 0:
                print(f"  [UP] Feature {feat_idx}: FLIP! (Pred -> 0)")
                found_fix = True
                break
        
        if not found_fix: print("  [UP] No single feature fixed it.")
            
        # --- DIAL DOWN ---
        found_fix_down = False
        # Lower threshold to find weak activations
        active_feats = (z[0] > 0.01).nonzero(as_tuple=True)[0]
        print(f"  Active features (>0.01): {active_feats.tolist()}")
        
        for feat_idx in active_feats:
            z_mod = z.clone()
            z_mod[0, feat_idx] = 0.0
            act_mod = sae.decoder(z_mod)
            p_new = manual_forward_classifier(model, act_mod).argmax().item()
            
            if p_new == 0:
                print(f"  [DOWN] Feature {feat_idx}: FLIP! (Pred -> 0)")
                found_fix_down = True
                break
                
        if not found_fix_down: print("  [DOWN] No single feature fixed it.")

if __name__ == "__main__":
    run_experiment()
