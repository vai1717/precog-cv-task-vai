import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
import sys
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.simple_cnn import SimpleCNN
from src.data.biased_mnist import BiasedMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_verify():
    print("--- Task 1: The Cheater (Verification) ---")
    from PIL import Image
    
    # 1. Setup Data
    # Use 95% bias (standard in literature) with LIMITED training data
    # Key insight: With only 2000 samples, model can't learn shapes, only colors
    full_train_dataset = BiasedMNIST(
        root='./data',
        train=True,
        download=True,
        bias_ratio=0.95  # 95% color-label correlation (standard)
    )
    
    # CRITICAL: Limit training to 500 samples to prevent shape learning
    n_samples = 500
    indices = list(range(n_samples))
    train_subset = torch.utils.data.Subset(full_train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True
    )
    print(f"Training on {n_samples} samples (limited to force color shortcut)")

    test_dataset = BiasedMNIST(
        root='./data',
        train=False,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # 2. Model
    print("Initializing SimpleCNN...")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3. Training - 5 epochs (proper training, mathematically sound)
    print("Training for 5 epochs...")
    model.train()
    for epoch in range(5):
        correct = 0
        total = 0
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f}, Acc {acc:.2f}%")

    # 4. Hard Test
    print("\nEvaluating on Hard Test Set...")
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    print(f"Hard Test Accuracy: {test_acc:.2f}%")

    if test_acc < 30:
        print("SUCCESS: Model failed significantly (<30%), proving reliance on color bias.")
    else:
        print("WARNING: Accuracy too high — bias may be weak.")

    # 5. Confusion Matrix
    os.makedirs("artifacts/task1_cheater", exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Hard Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("artifacts/task1_cheater/confusion_matrix.png")
    plt.close()
    print("Saved confusion matrix.")

    # 6. Specific Trap: Red 1
    print("\nSpecific Trap: Red 1")

    # Use UNBIASED dataset to extract clean digit shape
    neutral_dataset = BiasedMNIST(
        root="./data",
        train=True,
        download=True,
        bias_ratio=0.0
    )

    idx = (neutral_dataset.targets == 1).nonzero(as_tuple=True)[0][0]
    img_raw = neutral_dataset.data[idx]  # Tensor -> convert to numpy uint8 (28,28)
    if isinstance(img_raw, torch.Tensor):
        img_raw = img_raw.numpy()

    # Convert to tensor
    img_tensor = transforms.ToTensor()(Image.fromarray(img_raw, mode="L"))  # [1,28,28]

    # Color it RED
    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    red_1 = img_tensor * red
    red_1 = red_1.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(red_1)
        probs = F.softmax(output, dim=1)
        pred = probs.argmax().item()
        conf = probs[0, pred].item()

    print("Input: Digit 1, Color Red")
    print(f"Prediction: {pred} (Confidence: {conf:.2f})")

    if pred == 0:
        print("SUCCESS: Predicted 0 (Color Bias) instead of 1 (Shape).")
    else:
        print("FAILURE: Model relied on shape.")

    # Save image correctly
    img_to_show = red_1.squeeze().permute(1, 2, 0).cpu().numpy()

    plt.figure()
    plt.imshow(img_to_show)
    plt.axis("off")
    plt.title(f"Red 1 → Pred {pred}")
    plt.savefig("artifacts/task1_cheater/trap_red1.png")
    plt.close()
    print("Saved trap image.")

    # 7. Digit 7 Analysis
    analyze_digit_7(model, neutral_dataset)

def analyze_digit_7(model, dataset):
    print("\n--- Detailed Analysis for Digit 7 ---")

    COLORS_NAME = {
        0: "Red", 1: "Green", 2: "Blue", 3: "Yellow",
        4: "Magenta", 5: "Cyan", 6: "Orange",
        7: "Purple", 8: "Lime", 9: "Azure"
    }

    idx = (dataset.targets == 7).nonzero(as_tuple=True)[0][0]
    img_raw = dataset.data[idx]
    if isinstance(img_raw, torch.Tensor):
        img_raw = img_raw.numpy()

    img_tensor = transforms.ToTensor()(
        Image.fromarray(img_raw, mode="L")
    )

    print(f"{'Color':<10} | {'Pred':<5} | {'Conf':<6} | {'P(True 7)':<10}")
    print("-" * 45)

    model.eval()
    with torch.no_grad():
        for cid in range(10):
            color = torch.tensor(BiasedMNIST.COLORS[cid]).view(3, 1, 1)
            colored = torch.clamp(img_tensor * color, 0, 1)

            out = model(colored.unsqueeze(0).to(device))
            probs = F.softmax(out, dim=1)

            pred = probs.argmax().item()
            conf = probs[0, pred].item()
            p7 = probs[0, 7].item()

            print(f"{COLORS_NAME[cid]:<10} | {pred:<5} | {conf:.4f} | {p7:.4f}")

if __name__ == "__main__":
    train_and_verify()
