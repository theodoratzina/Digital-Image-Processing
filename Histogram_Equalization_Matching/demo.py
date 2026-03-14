import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from hist_utils import calculate_hist_of_img
from hist_modify import perform_hist_eq
from hist_modify import perform_hist_matching

# In demo, all functions are being combined in order to create the requested images and their corresponding histograms.


# Set the filepaths to the image files
filename = "input_img.jpg"
filename_ref = "ref_img.jpg"

# Read the images into a PIL entity
img = Image.open(fp=filename)
img_ref = Image.open(fp=filename_ref)

# Keep only the Luminance component of the images
bw_img = img.convert("L")
bw_img_ref = img_ref.convert("L")

# Obtain the underlying np arrays
img_array = np.array(bw_img).astype(float) / 255.0
img_array_ref = np.array(bw_img_ref).astype(float) / 255.0


# Calculate the requested images
equalized_img_greedy = perform_hist_eq(img_array, 'greedy')
equalized_img_non_greedy = perform_hist_eq(img_array, 'non-greedy')
equalized_img_post_dist = perform_hist_eq(img_array, 'post-disturbance')
matched_img_greedy = perform_hist_matching(img_array, img_array_ref, 'greedy')
matched_img_non_greedy = perform_hist_matching(img_array, img_array_ref, 'non-greedy')
matched_img_post_dist = perform_hist_matching(img_array, img_array_ref, 'post-disturbance')

# Calculate the requested histograms of the images
hist = calculate_hist_of_img(img_array, True)
hist_ref = calculate_hist_of_img(img_array_ref, True)
equalized_hist_greedy = calculate_hist_of_img(equalized_img_greedy, True)
equalized_hist_non_greedy = calculate_hist_of_img(equalized_img_non_greedy, True)
equalized_hist_post_dist = calculate_hist_of_img(equalized_img_post_dist, True)
matched_hist_greedy = calculate_hist_of_img(matched_img_greedy, True)
matched_hist_non_greedy = calculate_hist_of_img(matched_img_non_greedy, True)
matched_hist_post_dist = calculate_hist_of_img(matched_img_post_dist, True)

bin_centers = np.arange(256)


# Create a new figure for the image and histogram equalization
fig1 = plt.figure(1, figsize=(8, 8))

# Original image and histogram
plt.subplot(4, 2, 1)  
plt.imshow(img_array, cmap='gray')
plt.colorbar(location = 'left')
plt.title('Original Image') 
plt.axis('off')

plt.subplot(4, 2, 2)
plt.bar(bin_centers, hist.values(), width=1, color='indigo')

# Greedy equalization and histogram
plt.subplot(4, 2, 3)  
plt.imshow(equalized_img_greedy, cmap='gray')  
plt.colorbar(location = 'left')
plt.title('Greedy Equalized Image') 
plt.axis('off')

plt.subplot(4, 2, 4)
plt.bar(bin_centers, equalized_hist_greedy.values(), width=1, color='purple')

# Non-greedy equalization and histogram
plt.subplot(4, 2, 5)  
plt.imshow(equalized_img_non_greedy, cmap='gray')
plt.colorbar(location = 'left')  
plt.title('Non-Greedy Equalized Image') 
plt.axis('off')

plt.subplot(4, 2, 6)
plt.bar(bin_centers, equalized_hist_non_greedy.values(), width=1, color='darkviolet')

# Post-disturbance equalization and histogram
plt.subplot(4, 2, 7)  
plt.imshow(equalized_img_post_dist, cmap='gray')  
plt.colorbar(location = 'left')
plt.title('Post-Disturbance Equalized Image') 
plt.axis('off')

plt.subplot(4, 2, 8)
plt.bar(bin_centers, equalized_hist_post_dist.values(), width=1, color='orchid')

plt.suptitle("Equalization Histograms - Images ", fontsize=16,fontweight='bold')


# Create a new figure for the image and histogram matching
fig2 = plt.figure(2, figsize=(8, 8))

#Original image and histogram
plt.subplot(4, 2, 1)  
plt.imshow(img_array, cmap='gray')
plt.colorbar(location = 'left')
plt.title('Original Image') 
plt.axis('off')

plt.subplot(4, 2, 2)
plt.bar(bin_centers, hist.values(), width=1, color='maroon')

# Greedy matching and histogram
plt.subplot(4, 2, 3)  
plt.imshow(matched_img_greedy, cmap='gray')  
plt.colorbar(location = 'left')
plt.title('Greedy Matched Image') 
plt.axis('off')

plt.subplot(4, 2, 4)
plt.bar(bin_centers, matched_hist_greedy.values(), width=1, color='brown')

# Non-greedy matching and histogram
plt.subplot(4, 2, 5)  
plt.imshow(matched_img_non_greedy, cmap='gray')  
plt.colorbar(location = 'left')
plt.title('Non-Greedy Matched Image') 
plt.axis('off')

plt.subplot(4, 2, 6)
plt.bar(bin_centers, matched_hist_non_greedy.values(), width=1, color='red')

# Post-disturbance matching and histogram
plt.subplot(4, 2, 7)  
plt.imshow(matched_img_post_dist, cmap='gray')  
plt.colorbar(location = 'left')
plt.title('Post-Disturbance Mtched Image') 
plt.axis('off')

plt.subplot(4, 2, 8)
plt.bar(bin_centers, matched_hist_post_dist.values(), width=1, color='salmon')

plt.suptitle("Matching Histograms - Images ", fontsize=16,fontweight='bold')


# Create 2 new figures for the original and reference images
fig3 = plt.figure(3, figsize=(8, 8))  
plt.imshow(img_array, cmap='gray')
plt.title('Original Image') 
plt.axis('off')

fig4 = plt.figure(4, figsize=(8, 8))
plt.imshow(img_array_ref, cmap='gray')
plt.title('Reference Image')
plt.axis('off')


# Create a new figure for the reference image and histogram
fig5 = plt.figure(5, figsize=(8, 4))
plt.subplot(1, 2, 1)  
plt.imshow(img_array_ref, cmap='gray')
plt.colorbar(location = 'left')
plt.title('Reference Image') 
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(bin_centers, hist_ref.values(), width=1, color='black')


plt.show()  # Show the figures 