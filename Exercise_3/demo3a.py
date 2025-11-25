import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from n_cuts import n_cuts


# Load the images data
data = loadmat("dip_hw_3.mat")
img_array_1 = data["d2a"]  
img_array_2 = data["d2b"]

# Create new figures for the original images
fig1 = plt.figure(1, figsize=(8, 8))  
plt.imshow(img_array_1, cmap='gray')
plt.title('Original Image d2a') 
plt.axis('off')

fig2 = plt.figure(2, figsize=(8, 8))
plt.imshow(img_array_2, cmap='gray')
plt.title('Original Image d2b')
plt.axis('off')

# Convert the images to graphs (affinity matrix)
affinity_mat_1 = image_to_graph(img_array_1)
affinity_mat_2 = image_to_graph(img_array_2)

# Perform the n-cuts for various values of k
for k in [2, 3, 4]:
    labels_1 = n_cuts(affinity_mat_1, k)
    labels_2 = n_cuts(affinity_mat_2, k)

    print(f"\n🔹 Normalized-cuts of d2a for k = {k}")
    print(labels_1)
    print("-" * 40)
    print(f"\n🔹 Normalized-cuts of d2b for k = {k}")
    print(labels_2)
    print("-" * 40)

    # Plotting the results
    fig1 = plt.figure(3)
    plt.title(f"Normalized-cuts on Image d2a (k={k})")
    plt.imshow(labels_1.reshape(img_array_1.shape[:2]), cmap="nipy_spectral")
    plt.axis("off")

    fig2 = plt.figure(4)
    plt.title(f"Normalized-cuts on Image d2b (k={k})")
    plt.imshow(labels_2.reshape(img_array_2.shape[:2]), cmap="nipy_spectral")
    plt.axis("off")

    plt.show()
