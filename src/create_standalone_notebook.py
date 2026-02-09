import nbformat as nbf
import os

def create_notebook():
    source_path = 'src/lazy_artist_standalone.py'
    if not os.path.exists(source_path):
        print(f"Error: {source_path} not found.")
        return

    with open(source_path, 'r') as f:
        code_content = f.read()

    nb = nbf.v4.new_notebook()
    
    # Title Cell
    nb.cells.append(nbf.v4.new_markdown_cell("""# The Lazy Artist: Standalone Analysis
    
This notebook contains the consolidated code for the entire "Lazy Artist" project, including:
- **Task 0**: Biased Data Generation
- **Task 1**: Cheater Model Training
- **Task 2**: Feature Visualization
- **Task 3**: Grad-CAM
- **Task 4**: Intervention (Permutation, Consistency, Sobel)
- **Task 5**: Adversarial Robustness

All logic is self-contained in this single execution flow.
"""))

    # Code Cell
    nb.cells.append(nbf.v4.new_code_cell(code_content))
    
    output_path = 'notebooks/Lazy_Artist_Consolidated.ipynb'
    os.makedirs('notebooks', exist_ok=True)
    
    with open(output_path, 'w') as f:
        nbf.write(nb, f)
        
    print(f"Notebook created at: {output_path}")

if __name__ == "__main__":
    create_notebook()
