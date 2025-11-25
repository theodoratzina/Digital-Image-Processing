import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


# This function performs non-recursive Normalized Cuts on an affinity matrix
def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:

    # Get the number of nodes in the affinity matrix
    N = affinity_mat.shape[0]
    if N <= k:
        # Not enough nodes to split, assign each node to its own cluster
        return np.arange(N)
    if N <= 2:
        # For 2 or fewer nodes, just assign clusters directly
        return np.arange(N)
    
    # Step 1: Compute the degree matrix D
    D = np.diag(np.sum(affinity_mat, axis=1))

    # Step 2: Compute the Laplacian matrix L = D - W
    L = D - affinity_mat

    # Step 3: Compute the eigenvalues and eigenvectors of L, D only if k < N-1
    if k < N - 1:
        eigenvalues, eigenvectors = eigs(L, k=k, M=D, which='SR')
    else:
        # Fallback: assign clusters directly or use another method
        return np.arange(N)
    
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


# This function calculates the normalized cut value for a given clustering of nodes
def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:

    # Get the indices of the two clusters
    A = np.where(cluster_idx == 0)[0]
    B = np.where(cluster_idx == 1)[0]

    # assoc(A, A): total weight of edges inside cluster A
    assoc_AA = np.sum(affinity_mat[np.ix_(A, A)])

    # assoc(B, B): total weight of edges inside cluster B
    assoc_BB = np.sum(affinity_mat[np.ix_(B, B)])

    # assoc(A, V): total degree of nodes in cluster A
    assoc_AV = np.sum(affinity_mat[A, :])

    # assoc(B, V): total degree of nodes in cluster B
    assoc_BV = np.sum(affinity_mat[B, :])

    # Avoid division by zero
    term_A = assoc_AA / assoc_AV if assoc_AV > 0 else 0
    term_B = assoc_BB / assoc_BV if assoc_BV > 0 else 0

    # Normalized association and Ncut value
    n_assoc = term_A + term_B
    n_cut_value = 2 - n_assoc

    return n_cut_value


# This function performs recursive Normalized Cuts on an affinity matrix
def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    
    # Get the number of nodes in the affinity matrix
    n_nodes = affinity_mat.shape[0]

    # Use the existing n_cuts function with k=2 for binary clustering
    binary_clusters = n_cuts(affinity_mat, k=2)

    # Calculate the N-cut value for this binary partition
    n_cut_value = calculate_n_cut_value(affinity_mat, binary_clusters)

    # If number of nodes is below threshold T1, don't split further
    if n_nodes <= T1:
        cluster_idx = binary_clusters
        return cluster_idx

    # If N-cut value is above threshold T2, don't split further
    if n_cut_value > T2:
        cluster_idx = binary_clusters
        return cluster_idx

    # Get indices for each cluster
    cluster_0_indices = np.where(binary_clusters == 0)[0]
    cluster_1_indices = np.where(binary_clusters == 1)[0]

    # Initialize result array
    cluster_idx = np.zeros(n_nodes, dtype=int)

    # Recursively process cluster 0
    if len(cluster_0_indices) > 0:
        sub_affinity_0 = affinity_mat[np.ix_(cluster_0_indices, cluster_0_indices)]
        sub_clusters_0 = n_cuts_recursive(sub_affinity_0, T1, T2)
        cluster_idx[cluster_0_indices] = sub_clusters_0

    # Recursively process cluster 1
    if len(cluster_1_indices) > 0:
        sub_affinity_1 = affinity_mat[np.ix_(cluster_1_indices, cluster_1_indices)]
        sub_clusters_1 = n_cuts_recursive(sub_affinity_1, T1, T2)
        max_cluster_0 = np.max(cluster_idx[cluster_0_indices]) if len(cluster_0_indices) > 0 else -1
        cluster_idx[cluster_1_indices] = sub_clusters_1 + max_cluster_0 + 1

    return cluster_idx
