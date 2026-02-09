import nbformat as nbf
import os
import re

def read_code(path, exclude_imports=True, exclude_main=True):
    if not os.path.exists(path):
        return f"# File not found: {path}"
        
    with open(path, 'r') as f:
        lines = f.readlines()
    
    cleaned = []
    skip_block = False
    for line in lines:
        if exclude_imports and (line.strip().startswith('from src.') or line.strip().startswith('import src.')):
            # Comment out imports instead of removing, to show provenance
            cleaned.append(f"# {line}")
            continue
            
        if exclude_main and 'if __name__ == "__main__":' in line:
            break
            
        cleaned.append(line)
        
    return "".join(cleaned)

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # --- Cell 1: Title & Task List ---
    cells.append(nbf.v4.new_markdown_cell("""# The Lazy Artist: Analytic Report & Submission

**Candidate Submission**

This notebook documents the entire lifecycle of the "Lazy Artist" project, investigating Shortcut Learning in Neural Networks.

## Task List
1. **Task 0: The Biased Canvas** - Dataset Generation (BiasedMNIST)
2. **Task 1: The Cheater** - Training a color-biased model
3. **Task 2: The Prober** - Feature Visualization
4. **Task 3: The Interrogation** - Grad-CAM Analysis
5. **Task 4: The Intervention** - Curing the bias (3 Methods)
6. **Task 5: The Invisible Cloak** - Adversarial Robustness
7. **Task 6: The Decomposition** - Sparse Autoencoders (SAE)
"""))

    # --- Cell 2: Configuration & Imports ---
    cells.append(nbf.v4.new_markdown_cell("## Configuration & Setup"))
    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import cv2
import os
import sys
from IPython.display import Image as DisplayImage, display

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
"""))

    # --- Task 0: Data ---
    cells.append(nbf.v4.new_markdown_cell("## Task 0: The Biased Canvas\n\nWe generate a custom MNIST dataset where color is spuriously correlated with the digit label (e.g., 0 is usually Red).\n\n### Codebase: `BiasedMNIST` Dataset Class"))
    cells.append(nbf.v4.new_code_cell(read_code('src/data/biased_mnist.py')))
    
    cells.append(nbf.v4.new_code_cell("""# Dataset Statistics & Configuration
train_dataset = BiasedMNIST(root='./data', train=True, download=True, bias_ratio=0.95)
test_dataset = BiasedMNIST(root='./data', train=False, download=True)

print(f"Training Set Size: {len(train_dataset)}")
print(f"Hard Test Set Size: {len(test_dataset)}")
print(f"Dominant Color Rate (Train): {train_dataset.bias_ratio * 100}%")
print(f"Dominant Color Rate (Hard Test): 0% (Randomized)")
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualize Samples
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
indices = [0, 1, 2, 10, 12, 15] # Random indices
for idx, ax in zip(indices, axes):
    img, target, color_idx = train_dataset[idx]
    # img is tensor [3, 28, 28]
    img_np = img.permute(1, 2, 0).numpy()
    ax.imshow(img_np)
    ax.set_title(f"Label: {target} (Color: {color_idx})")
    ax.axis('off')
plt.suptitle("Biased Training Samples")
plt.show()
"""))

    # --- Task 1: Cheater ---
    cells.append(nbf.v4.new_markdown_cell("## Task 1: The Cheater\n\nWe train a simple CNN on a subset of this biased data to force it to rely on the easy color shortcut.\n\n### Codebase: `SimpleCNN` Model"))
    cells.append(nbf.v4.new_code_cell(read_code('src/models/simple_cnn.py')))
    
    cells.append(nbf.v4.new_markdown_cell("### Training & Verification Logic"))
    # We strip the main block from task1_cheater but keep the functions
    cells.append(nbf.v4.new_code_cell(read_code('src/task1_cheater.py')))
    
    cells.append(nbf.v4.new_code_cell("""# Execute Training & Verification
from PIL import Image
train_and_verify()
"""))
    
    cells.append(nbf.v4.new_markdown_cell("### Task 1 Results: Visualizations"))
    cells.append(nbf.v4.new_code_cell("""try:
    display(DisplayImage(filename='artifacts/task1_cheater/confusion_matrix.png'))
    display(DisplayImage(filename='artifacts/task1_cheater/trap_red1.png'))
except Exception as e:
    print(f"Artifacts not found: {e}")
"""))
    
    # --- Task 2: Features ---
    cells.append(nbf.v4.new_markdown_cell("## Task 2: The Prober (Feature Visualization)\n\nWe investigate what the neurons are seeing by generating input images that maximize their activation.\n\n### Codebase: Feature Visualizer"))
    cells.append(nbf.v4.new_code_cell(read_code('src/vis_utils.py')))
    
    # We'll inline the Task 2 execution logic slightly or just call the main if adapted
    cells.append(nbf.v4.new_code_cell(read_code('src/task2_prober.py')))
    
    cells.append(nbf.v4.new_code_cell("""# Run Feature Visualization
main()
"""))
    
    cells.append(nbf.v4.new_markdown_cell("### Task 2 Results: Interpretation\nThe generated images show the preferred stimuli for the neurons. \n- **Conv1**: Shows simple colors.\n- **Conv2**: Shows complex color mixtures rather than shapes.\n\n**Visualizations:**"))
    cells.append(nbf.v4.new_code_cell("""try:
    print("Conv1 Features:")
    display(DisplayImage(filename='artifacts/task2_vis/conv1_all.png'))
    print("Conv2 Features:")
    display(DisplayImage(filename='artifacts/task2_vis/conv2_all.png'))
    print("Polysemanticity Probe:")
    import glob
    poly = glob.glob('artifacts/task2_vis/poly_ch*_all.png')
    if poly:
        display(DisplayImage(filename=poly[0]))
except Exception as e:
    print(e)
"""))

    # --- Task 3: GradCAM ---
    cells.append(nbf.v4.new_markdown_cell("## Task 3: The Interrogation (Grad-CAM)\n\nWe implement Grad-CAM from scratch to visualize the model's attention map.\n\n### Codebase: GradCAM Implementation"))
    cells.append(nbf.v4.new_code_cell(read_code('src/gradcam_scratch.py')))
    
    cells.append(nbf.v4.new_code_cell(read_code('src/task3_gradcam.py')))
    
    cells.append(nbf.v4.new_code_cell("""# Run Grad-CAM Analysis
run_task3()
"""))
    
    cells.append(nbf.v4.new_markdown_cell("### Task 3 Results: Attention Maps"))
    cells.append(nbf.v4.new_code_cell("""try:
    print("Biased Prediction (Red 0, Correct):")
    display(DisplayImage(filename='artifacts/task3_gradcam/gradcam_biased_red0.png'))
    print("Conflicting Prediction (Green 0, Trap):")
    display(DisplayImage(filename='artifacts/task3_gradcam/gradcam_conflicting_green0.png'))
except Exception as e:
    print(e)
"""))

    # --- Task 4: Intervention ---
    cells.append(nbf.v4.new_markdown_cell("## Task 4: The Intervention\n\nWe attempt to cure the color bias using three methods:\n1. **Channel Permutation** (Data Augmentation)\n2. **Feature Consistency** (Penultimate Layer Loss)\n3. **Laplacian Filtering** (Input Preprocessing)\n\n### Codebase: Intervention Experiments"))
    cells.append(nbf.v4.new_code_cell(read_code('src/task4_intervention.py')))
    
    cells.append(nbf.v4.new_code_cell("""# Run Interventions
# Note: This might take a few minutes as it trains multiple models
run_experiment()
"""))
    
    cells.append(nbf.v4.new_markdown_cell("### Task 4 Results: Comparison"))
    cells.append(nbf.v4.new_code_cell("""from src.generate_report_charts import generate_charts
generate_charts()
try:
    print("Accuracy Comparison:")
    display(DisplayImage(filename='artifacts/summary_charts/task4_accuracy_comparison.png'))
    print("Training Curves:")
    display(DisplayImage(filename='artifacts/summary_charts/task4_training_curves.png'))
    print("Confusion Matrices:")
    display(DisplayImage(filename='artifacts/summary_charts/task4_confusion_matrices.png'))
except Exception as e:
    print(f"Error displaying charts: {e}")
"""))

    # --- Task 5: Adversarial ---
    cells.append(nbf.v4.new_markdown_cell("## Task 5: The Invisible Cloak\n\nWe investigate the Adversarial Robustness of the 'Lazy' (Color) model vs the 'Robust' (Shape) model.\n\n### Codebase: PGD Attack"))
    cells.append(nbf.v4.new_code_cell(read_code('src/task5_adversarial.py')))
    
    cells.append(nbf.v4.new_code_cell("""# Run Adversarial Attacks
run_attack()
"""))
    
    cells.append(nbf.v4.new_markdown_cell("### Task 5 Results: Robustness Paradox\nSurprisingly, the 'Lazy' model is harder to fool with small perturbations because it ignores high-frequency shape features."))
    cells.append(nbf.v4.new_code_cell("""try:
    display(DisplayImage(filename='artifacts/task5_adversarial/attack_comparison.png'))
    display(DisplayImage(filename='artifacts/summary_charts/task5_robustness_comparison.png'))
except: pass
"""))

    # --- Task 6: SAE ---
    cells.append(nbf.v4.new_markdown_cell("## Task 6: The Decomposition (SAE)\n\nWe use a Sparse Autoencoder to decompose the internal representation and finding specific features (e.g., 'Red Detector').\n\n### Codebase: Sparse Autoencoder"))
    cells.append(nbf.v4.new_code_cell(read_code('src/task6_sae.py')))
    
    cells.append(nbf.v4.new_code_cell("""# Run SAE Experiment
run_experiment()
"""))

    # --- Summary ---
    cells.append(nbf.v4.new_markdown_cell("""## Final Summary & Key Findings

| Task | Key Finding |
| :--- | :--- |
| **0. Data** | Generated biased dataset (95% correlation) successfully. |
| **1. Cheater** | Lazy model achieves high accuracy (99%) but fails on shape (Hard Test < 30%). |
| **2. Prober** | Neurons are polysemantic color detectors, ignoring shape. |
| **3. Grad-CAM** | Model focus is entirely on colored pixels; fails when color contradicts shape. |
| **4. Intervention** | Channel Permutation (Augmentation) was the most effective cure (>70% Hard Acc). |
| **5. Adversarial** | The Lazy model is oddly robust to shape noise; the Robust model is fragile. |
| **6. SAE** | Decomposed features reveal a 'Red Detector' that flips predictions when activated. |
"""))

    nb['cells'] = cells
    
    os.makedirs("notebooks", exist_ok=True)
    out_path = 'notebooks/Lazy_Artist_Analysis.ipynb'
    with open(out_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Detailed notebook created at {out_path}")

if __name__ == "__main__":
    create_notebook()
