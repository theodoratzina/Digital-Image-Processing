import numpy as np

def fir_conv(in_img_array: np.ndarray, h: np.ndarray) -> np.ndarray:

    #This function performs FIR convolution on the input image array using the given filter kernel.
    
    # Calculate the padding size
    pad_height = h.shape[0] // 2
    pad_width = h.shape[1] // 2
    
    # Pad the input image with zeros on all sides
    padded_img = np.pad(in_img_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Initialize the output image array with zeros
    out_img_array = np.zeros_like(in_img_array)
    
    # Perform convolution
    for i in range(in_img_array.shape[0]):
        for j in range(in_img_array.shape[1]):
            # Extract the region of interest from the padded image
            region = padded_img[i:i + h.shape[0], j:j + h.shape[1]]
            # Perform element-wise multiplication and sum the result
            out_img_array[i, j] = np.sum(region * h)
    
    return out_img_array