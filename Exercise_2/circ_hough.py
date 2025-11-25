import numpy as np
from typing import Tuple

def circ_hough(in_img_array: np.ndarray, R_max: float, R_min: float, dim: np.ndarray, V_min: int) -> Tuple[np.ndarray, np.ndarray]:
    
    height, width = in_img_array.shape
    K, L, M = dim

    # Initialize the accumulator array
    accumulator = np.zeros((K, L, M), dtype=int)

    # Create radius values
    R_vals = np.linspace(R_min, R_max, M)

    # Calculate step sizes
    step_a = int(width / K)
    step_b = int(height / L)
    step_r = int((R_max - R_min) / M)

    # Find edge points
    edge_points = np.argwhere(in_img_array == 1)

    for y, x in edge_points:  # y=row, x=col
        for r in R_vals:
            for theta in range(0, 360, 2):  # theta in degrees

                # Calculate potential circle centers
                a = int(x - r * np.cos(theta * np.pi / 180))  # Convert theta to radians
                b = int(y - r * np.sin(theta * np.pi / 180))

                # Convert to accumulator indices
                a_idx = int(a / step_a)
                b_idx = int(b / step_b)
                r_idx = int((r - R_min) / step_r)

                if 0 <= a_idx < K and 0 <= b_idx < L and 0 <= r_idx < M:
                    accumulator[a_idx, b_idx, r_idx] += 1


    # Extract circles above threshold
    centers = []
    radii = []
    for a_idx in range(K):
        for b_idx in range(L):
            for r_idx in range(M):
                if accumulator[a_idx, b_idx, r_idx] >= V_min:
                    a_i = (a_idx * step_a)
                    b_i = (b_idx * step_b)
                    r_i = (R_min + (r_idx * step_r))

                    a_ii = ((a_idx + 1) * step_a)
                    b_ii = ((b_idx + 1) * step_b)
                    r_ii = (R_min + ((r_idx + 1) * step_r))

                    a = (a_i + a_ii) / 2
                    b = (b_i + b_ii) / 2
                    r = (r_i + r_ii) / 2
                    
                    centers.append([a, b])
                    radii.append(r)

    return np.array(centers), np.array(radii)
