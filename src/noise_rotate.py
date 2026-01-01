import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Configuration
INPUT_FILE = "data/akeome_2d.npy"
OUTPUT_FILE = "data/secret_akeome_10d.csv"
SIGNAL_SCALE = 10.0  # High variance for text
NOISE_SCALE = 1.0    # Low variance for noise

def create_high_dim_data(input_path):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return None

    X_2d = np.load(input_path)
    n_samples = X_2d.shape[0]

    X_2d = (X_2d - X_2d.mean(axis=0)) / X_2d.std(axis=0)
    X_signal = X_2d * SIGNAL_SCALE

    X_noise = np.random.normal(0, NOISE_SCALE, (n_samples, 8))

    X_10d = np.hstack([X_signal, X_noise])

    random_matrix = np.random.randn(10, 10)
    Q, _ = np.linalg.qr(random_matrix)

    X_final = np.dot(X_10d, Q)

    return X_final


print(f"Processing {INPUT_FILE} ...")
X_10d = create_high_dim_data(INPUT_FILE)

if X_10d is not None:
    np.savetxt(OUTPUT_FILE, X_10d, delimiter=",", fmt="%.5f")
    print(f"Saved 10D data to: {OUTPUT_FILE}")

    print("Verifying with PCA...")
    pca = PCA(n_components=2)
    X_recovered = pca.fit_transform(X_10d)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_recovered[:, 0], X_recovered[:, 1], s=2, c='blue', alpha=0.6)
    plt.title("PCA Result (Hidden Message Recovered)")
    plt.axis('equal')
    plt.grid(True, linestyle='--')
    plt.show()

else:
    print("Failed to create data.")