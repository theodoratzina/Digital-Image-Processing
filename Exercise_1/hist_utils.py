import numpy as np
from typing import Dict

# This function calculates and returns the equalization transform of the input image.
def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:

    L = 256                         # Number of levels in the image (0-255)
    level_appearances = [0] * L     # Initialize with size 256
    hist = {}                       # Initialize the dictionary

    # Count appearances of each pixel value
    for i in img_array:
        for j in i: 
            # The index (i.e. j) represents the value taken by the pixel in the input image
            level = int(j * 255)
            level_appearances[level] += 1

    total_pixels = img_array.size   # Total number of pixels in the image, accounting for all dimensions

    for i in range(L):
        if return_normalized:
            # Case 1: Use normalized probabilities as values
            hist[f"f_{i}"] = level_appearances[i] / total_pixels
        else:
            # Case 2: Use raw counts as values
            hist[f"f_{i}"] = level_appearances[i]

    return hist


# This function applies the histogram modification transformation to the input image array.
def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:

    # Initialize the modified image as a black image with the dimensions of the input image
    modified_img = np.zeros(img_array.shape, img_array.dtype)

    # Get the modified image transformation
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            # Convert pixel value to key format "f_{intensity}"
            key = f"f_{int(img_array[i, j] * 255)}"
            # Retrieve mapped value and normalize
            mapped_value = modification_transform.get(key, 0)
            modified_img[i, j] = mapped_value / 255.0
 
    return modified_img