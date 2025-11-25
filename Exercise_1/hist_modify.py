import numpy as np 
from typing import Dict
from hist_utils import calculate_hist_of_img
from hist_utils import apply_hist_modification_transform


# This function performs histogram modification on the input image array based on the reference histogram and mode.
def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    
    # Initialize the modified image as a black image with the same dimensions as the input image
    modified_img = np.zeros(img_array.shape, img_array.dtype)     
    
    count_freq = calculate_hist_of_img(img_array, False)    # Calculate the counts of frequencies of the input image
    total_pixels = img_array.size                           # Total number of pixels in the image, accounting for all dimensions
    new_hist = {}                                           # Initialize the histogram modification transform
    

    # Greedy histogram modification
    if mode == 'greedy':   
        
        sigma_count = 0.0
        output_level = 0

        for i in range(256):  # i: input intensity level
            # Calculate the target number of pixels for the current output level
            target = total_pixels * hist_ref[f"f_{output_level}"]
            count = count_freq[f"f_{i}"]
            if output_level >= 256:
                # Any remaining levels map to the last output level
                new_hist[i] = 255
            else:
                sigma_count += count
                new_hist[f"f_{i}"] = output_level
                # Once we've reached target pixels for this bin, move to next output level
                if sigma_count >= (output_level + 1) * target:
                    output_level += 1

        modified_img = apply_hist_modification_transform(img_array, new_hist)  # apply the equalization transform to the image
    

    # Non-greedy histogram modification
    if mode == 'non-greedy':

        output_level = 0
        sigma_count = 0.0  

        for j in range(256):  # j: input intensity level
            count = count_freq[f"f_{j}"]
            deficiency = total_pixels * hist_ref[f"f_{output_level}"] - sigma_count   # Calculate deficiency

            if deficiency >= count / 2:
                # Assign input level j to current output level
                new_hist[f"f_{j}"] = output_level
                sigma_count += count
            else:
                # Move to next output level and assign j to it
                output_level += 1
                if output_level >= 256:
                    output_level = 255  # Any remaining levels map to the last output level
                sigma_count = 0.0
                new_hist[f"f_{j}"] = output_level
                sigma_count += count

        modified_img = apply_hist_modification_transform(img_array, new_hist)


    # Post-disturbance histogram modification
    if mode == 'post-disturbance':
        
        d = 1.0 / 255.0    # Calculate level distance
        
        noise = np.random.uniform(-d/2, d/2, img_array.shape)   # Generate random noise in the range [-d/2, d/2]
        disturbed_img = img_array + noise    # Add noise to the image array

        modified_img = perform_hist_modification(disturbed_img, hist_ref, 'greedy')

    return modified_img



# This function performs histogram equalization on the input image array based on the specified mode.
def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    
    hist_eq = {}   # Initialize the histogram equalization transform

    # Initialize the equalized image as a black image with the same dimensions as the input image
    equalized_img = np.zeros(img_array.shape, img_array.dtype) 

    # Define the ideal uniform histogram for equalization
    hist_eq = {f"f_{i}": 1/256 for i in range(256)}

    equalized_img = perform_hist_modification(img_array, hist_eq, mode)     # Apply the final equalization transform to the image

    return equalized_img



# This function performs histogram matching on the input image array based on the reference image and specified mode.
def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    
    img_array_ref_hist = calculate_hist_of_img(img_array_ref , True)   # Calculate the histogram of the reference image
    
    # Initialize the processed image as a black image with the same dimensions as the input image
    processed_img = np.zeros(img_array.shape, img_array.dtype) 
    processed_img = perform_hist_modification(img_array, img_array_ref_hist, mode)  # Apply the histogram matching transform to the image

    return processed_img