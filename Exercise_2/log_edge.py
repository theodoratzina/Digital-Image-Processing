import numpy as np
from fir_conv import fir_conv

def log_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    
    #This function applies the Laplacian of Gaussian (LoG) edge detection algorithm to the input image array.
        
    # Define the LoG filter kernel
    log_kernel = np.array([[0, 0, -1, 0, 0],
                            [0, -1, -2, -1, 0],
                            [-1, -2, 16, -2, -1],
                            [0, -1, -2, -1, 0],
                            [0, 0, -1, 0, 0]])
    
    # Use the fir_conv function to perform convolution
    filtered_img = fir_conv(in_img_array, log_kernel)
    
    # Initialize the output image array with zeros
    out_img_array = np.zeros(filtered_img.shape, dtype=int)

    # Thresholding to create a binary edge map
    for i in range(1, filtered_img.shape[0] - 1):  # Avoid top and bottom edges
        for j in range(1, filtered_img.shape[1] - 1):  # Avoid left and right edges
            center = filtered_img[i, j]
            neighbors = [filtered_img[i-1, j], filtered_img[i+1, j],
                        filtered_img[i, j-1], filtered_img[i, j+1],
                        filtered_img[i-1, j-1], filtered_img[i-1, j+1],
                        filtered_img[i+1, j-1], filtered_img[i+1, j+1]]
            
            # Check for a sign change
            if any((center > 0 and n < 0 and (center - n) > thres) or (center < 0 and n > 0 and (n - center) > thres) for n in neighbors):
                out_img_array[i, j] = 1

    return out_img_array