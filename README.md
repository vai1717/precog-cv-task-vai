# The Lazy Artist: Interpretability Project

## Overview
This project explores "Shortcut Learning" in deep neural networks. We investigate how a model (CNN) cheats by learning spurious correlations (color) instead of robust semantic features (shape). Through a series of 6 tasks, we expose the cheat, visualize the internal representation, and attempt to cure the bias.

## Directory Structure
```
.
├── artifacts/                  # Generated images and plots
├── data/                       # Dataset storage
├── notebooks/                  # Jupyter Notebooks for analysis
├── src/                        # Source code
│   ├── data/                   # BiasedMNIST dataset logic
│   ├── models/                 # SimpleCNN architecture
│   ├── task2_prober.py         # Task 2: Feature Visualization
│   ├── task3_gradcam.py        # Task 3: Grad-CAM Saliency
│   ├── kaggle_task4.py         # Task 4: Intervention (Kaggle/Colab ready)
│   ├── task5_adversarial.py    # Task 5: Adversarial Attacks
│   ├── kaggle_task6_sae.py     # Task 6: SAE Decomposition (Kaggle/Colab ready)
│   ├── verify_dataset.py       # Dataset verification script
│   └── visualize_dataset.py    # Dataset visualization script
├── COMPREHENSIVE_REPORT.md     # Full analysis and findings
└── README.md                   # This file
```

## Usage

### 🚀 **Master Submission Notebook**
The entire analysis (Tasks 0-6) can be reproduced by running the master notebook:
-   **Notebook**: `notebooks/Lazy_Artist_Analysis.ipynb`
-   **Command**: `jupyter notebook notebooks/Lazy_Artist_Analysis.ipynb`

### Individual Tasks
You can also run tasks individually via the scripts in `src/`:

### Task 0: Data Generation (The Biased Dataset)
-   **Goal**: Create `BiasedMNIST` where digit labels are correlated with color (e.g., 0=Red, 1=Green).
-   **Script**: `src/data/biased_mnist.py`
-   **Verification**: Run `python src/verify_dataset.py` to see the bias in action.

### Task 1: The Cheater (Baseline Model)
-   **Goal**: Train a simple CNN to see if it learns the color shortcut.
-   **Usage**: Run `jupyter notebook notebooks/Task1_The_Cheater.ipynb`.
-   **Finding**: The model achieves high training accuracy but fails on the conflicting test set, proving it learned color, not shape.

### Task 2: The Prober (Feature Visualization)
-   **Goal**: Visualize what individual neurons are "looking for" using Gradient Ascent.
-   **Script**: `src/task2_prober.py`
-   **Finding**: Neurons in early layers (Conv1/Conv2) are pure color detectors.

### Task 3: The Interrogation (Grad-CAM)
-   **Goal**: Generate Saliency Maps to see where the model looks.
-   **Script**: `src/task3_gradcam.py`
-   **Finding**: The model ignores the shape of the digit and focuses entirely on the colored pixels.

### Task 4: The Intervention (Curing Bias)
-   **Goal**: Force the model to learn shape by removing the color shortcut.
-   **Method**: Comparing Early Intervention (Laplacian Filter) vs. Late Intervention (Feature Consistency).
-   **Reproducible Script**: `src/task4_intervention.py`
    -   Run this script to train both models and compare accuracy on the hard test set.
-   **Finding**: Late intervention (Consistency) is superior (71%) to Early intervention (Laplacian 44%) on noisy data.

### Task 5: The Invisible Cloak (Adversarial Robustness)
-   **Goal**: Attack the model with unseen noise.
-   **Script**: `src/task5_adversarial.py`
-   **Finding**: The "Lazy" (Color) model is robust to small adversarial perturbations ($\epsilon=0.12$) because color is a low-frequency feature. The "Robust" (Shape) model is fragile ($\epsilon=0.05$) to high-frequency noise.

### Task 6: The Decomposition (Sparse Autoencoders)
-   **Goal**: Decompose the "Cheater" model's activations to find the specific "Color Neurons".
-   **Reproducible Script**: `src/task6_sae.py`
    -   Run this script to train an SAE and perform "Dial Up/Down" interventions.
-   **Finding**: The model is a **"Red Detector"**. It has a specific feature (Feature 6) for Red/0. It has **NULL** representation for Green/1, treating it as background.

## Requirements
-   Python 3.8+
-   PyTorch, Torchvision
-   Numpy, Matplotlib, Pillow
-   (Optional) CUDA GPU for faster training

## Assumptions and Limitations
1.  **Dataset**: We assume MNIST digits are sufficient proxies for object shapes.
2.  **Model**: We use a `SimpleCNN` with low capacity to encourage shortcut learning. A ResNet might behave differently (though likely still cheats).
3.  **Bias Ratio**: The 95% bias is chosen to be strong enough to trap the model but leave enough signal for "perfect" learning if the model were smart enough (it isn't).
4.  **Hardware**: Scripts are optimized for single-GPU execution.
