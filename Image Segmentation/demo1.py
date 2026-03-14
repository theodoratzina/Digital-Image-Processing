import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from spectral_clustering import spectral_clustering


# Load the data affinity matrix from the .mat file
data = loadmat("dip_hw_3.mat")
d1a = data["d1a"]

# Spectral Clustering for different values of k
for k in [2, 3, 4]:
    print(f"\n🔹 Clustering for k = {k}")
    labels = spectral_clustering(d1a, k)
    print(labels)
    print("-" * 40)

    # Plotting the labels as a 1D color bars
    plt.figure(figsize=(8, 2))
    plt.imshow(labels[np.newaxis, :], aspect='auto', cmap='nipy_spectral')
    plt.yticks([])
    plt.xlabel("Pixel Index")
    plt.title(f"Spectral Clustering of d1a (k={k})")
    plt.tight_layout() 
    plt.show()
    