import matplotlib.pyplot as plt
import os
import numpy as np

def generate_charts():
    output_dir = "artifacts/summary_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Chart 1: Task 4 Intervention Results ---
    # Data from Report/Experiments
    methods = ['Baseline (Lazy)', 'Early (Laplacian)', 'Late (Consistency)']
    accuracies = [13.0, 44.15, 71.69] # Approx values from report
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=colors, edgecolor='black')
    
    plt.title('Task 4: Bias Correction Methods\n(Accuracy on Bias-Conflicting Test Set)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    plt.savefig(os.path.join(output_dir, "task4_accuracy_comparison.png"))
    print("Saved task4_accuracy_comparison.png")
    
    # --- Chart 2: Task 5 Adversarial Robustness ---
    # Data from Report
    models = ['Lazy (Color-Biased)', 'Robust (Shape-Biased)']
    epsilons = [0.12, 0.05] # Epsilon required to fool
    colors = ['#ffcc99', '#c2c2f0']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, epsilons, color=colors, edgecolor='black')
    
    plt.title('Task 5: The Robustness Paradox\n(Minimum Noise $\epsilon$ to Fool Model)', fontsize=14)
    plt.ylabel('Perturbation Magnitude ($\epsilon$)', fontsize=12)
    plt.ylim(0, 0.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'$\epsilon \\approx {height}$',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    # Add annotation explaining the paradox
    plt.text(0, 0.01, "Harder to fool\n(Low Freq Feature)", ha='center', color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(1, 0.01, "Easier to fool\n(High Freq Feature)", ha='center', color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(output_dir, "task5_robustness_comparison.png"))
    print("Saved task5_robustness_comparison.png")

if __name__ == "__main__":
    generate_charts()
