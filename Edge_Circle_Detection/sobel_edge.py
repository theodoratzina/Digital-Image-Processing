import numpy as np 
from fir_conv import fir_conv

def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    
    #This function applies the Sobel edge detection algorithm to the input image array.
    
    # Define the Sobel filter kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    # Apply the Sobel filter in x and y directions using FIR convolution
    grad_x = fir_conv(in_img_array, sobel_x)
    grad_y = fir_conv(in_img_array, sobel_y)

    # Calculate the gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Initialize the output image array with zeros
    out_img_array = np.zeros(in_img_array.shape, dtype=int)

    # Thresholding to create a binary edge map
    for i in range(in_img_array.shape[0]):
        for j in range(in_img_array.shape[1]):
            if grad_magnitude[i, j] > thres:
                out_img_array[i, j] = 1
            else:
                out_img_array[i, j] = 0

    return out_img_array