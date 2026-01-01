import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuration
IMAGE_FILENAME = "data/pca.drawio.png"
FILE_NAME_NPY = "data/akeome_2d.npy"
FILE_NAME_CSV = "data/akeome_2d.csv"
TARGET_SIZE = (100, 100)

def load_handwriting_to_coordinates(filename, target_size):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    try:
        # Load and convert to grayscale
        img = Image.open(filename).convert('L')
        img = img.resize(target_size, Image.LANCZOS)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    img_array = np.array(img)
    height = img_array.shape[0]
    
    # Extract indices of dark pixels (threshold < 128)
    y_indices, x_indices = np.where(img_array < 128)
    
    if len(x_indices) == 0:
        print("No text detected.")
        return None

    # Flip Y-axis to match Cartesian coordinates
    y_indices_flipped = (height - 1) - y_indices
    
    return np.column_stack((x_indices, y_indices_flipped)).astype(float)

# --- Main Execution ---

print(f"Loading {IMAGE_FILENAME} ...")
coordinate_data = load_handwriting_to_coordinates(IMAGE_FILENAME, TARGET_SIZE)

if coordinate_data is not None:
    print(f"Success! Extracted {len(coordinate_data)} points.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(FILE_NAME_NPY), exist_ok=True)

    # Save data
    np.save(FILE_NAME_NPY, coordinate_data)
    np.savetxt(FILE_NAME_CSV, coordinate_data, delimiter=",", fmt="%.4f")
    print(f"Saved to: {FILE_NAME_NPY} and {FILE_NAME_CSV}")

    # Plot
    plt.figure(figsize=(5, 5))
    plt.scatter(coordinate_data[:, 0], coordinate_data[:, 1], s=2, c='black')
    plt.title("Extracted 2D Coordinates")
    plt.xlim(0, TARGET_SIZE[0])
    plt.ylim(0, TARGET_SIZE[1])
    plt.grid(True, linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

else:
    print("Failed to process data.")