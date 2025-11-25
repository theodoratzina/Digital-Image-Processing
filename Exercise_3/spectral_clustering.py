import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


# This function performs Spectral Clustering on an affinity matrix
def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:

    # Step 1: Compute the degree matrix D
    D = np.diag(np.sum(affinity_mat, axis=1))

    # Step 2: Compute the Laplacian matrix L = D - W
    L = D - affinity_mat

    # Step 3: Compute the eigenvalues and eigenvectors of L
    eigenvalues, eigenvectors = eigs(L, k=k, which='SR')

    # Step 4: Form the matrix U with the first k eigenvectors
    U = np.real(eigenvectors)

    # Step 5: Normalize each row of U
    U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

    # Step 6: Perform k-means clustering on the rows of U_normalized
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U_normalized)

    # Step 7: Return the cluster indices
    cluster_idx = kmeans.labels_


    return cluster_idx
