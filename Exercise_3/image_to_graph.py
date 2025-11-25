import numpy as np


# This function converts an image into a graph
def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    
    # Get image dimensions: M=height, N=width, C=channels
    M, N, C = img_array.shape

    # Total number of pixels (each will be a node in the graph)
    num_pixels = M * N

    # Reshape image to a 2D array
    img_reshaped = img_array.reshape((num_pixels, C))

    # Affinity matrix initialization
    affinity_mat = np.zeros((num_pixels, num_pixels), dtype=float)

    # Affinity matrix calculation
    for i in range(num_pixels):
        for j in range(i, num_pixels):  # Οnly upper triangle (symmetry)

            # Compute Euclidean distance between pixel i and j
            dist = np.linalg.norm(img_reshaped[i] - img_reshaped[j])

            # Convert distance to edge weight: A(i,j) = 1 / e^{d(i,j)}
            weight = np.exp(-dist)  

            # Assign the weight to both (i,j) and (j,i) since the graph is undirected
            affinity_mat[i, j] = weight
            affinity_mat[j, i] = weight  

    return affinity_mat
