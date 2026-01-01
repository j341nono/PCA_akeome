import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Configuration
INPUT_FILE = "10d_data/secret_akeome_10d.csv"
OUTPUT_DIR = "data"
OUTPUT_IMAGE_NAME = "pca_akeome_result.png"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE_NAME)

def run_pca_and_save_image():
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    # Load the 10-dimensional data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        X_10d = np.loadtxt(INPUT_FILE, delimiter=",")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Execute PCA (reduce to 2 dimensions)
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_10d)

    # Plotting
    plt.figure(figsize=(6, 6))
    
    # Scatter plot
    # s=1 makes points small enough to see the text clearly
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='blue', alpha=0.5)
    
    plt.title("PCA Result: Decoded Message")
    plt.axis('equal')  # Maintain aspect ratio
    plt.grid(True, linestyle='--', alpha=0.3)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the plot
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Image saved to: {OUTPUT_PATH}")
    
    # Show plot (optional, helpful if running interactively)
    plt.show()

if __name__ == "__main__":
    run_pca_and_save_image()