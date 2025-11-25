import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough

# In demo, all functions are being combined in order to create the requested images and diagrams.


# Set the filepaths to the image file
filename = "basketball_large.png"

# Read the image into a PIL entity
img = Image.open(fp=filename)

# Resize image to a smaller resolution, e.g., 75% of original size
scale_factor = 0.75
new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
img_resized = img.resize(new_size, Image.Resampling.LANCZOS)  # Use LANCZOS for better quality

# Keep only the Luminance component of the image
bw_img = img_resized.convert("L")


# Obtain the underlying np array
img_array = np.array(bw_img).astype(float) / 255.0

# Create a new figure for the original image
fig1 = plt.figure(1, figsize=(8, 8))  
plt.imshow(img_array, cmap='gray')
plt.title('Original Image') 
plt.axis('off')


# Calculate the sobel edge detection images for various thresholds
sobel_img_1 = sobel_edge(img_array, 0.1)
sobel_img_2 = sobel_edge(img_array, 0.2) 
sobel_img_3 = sobel_edge(img_array, 0.3)
sobel_img_4 = sobel_edge(img_array, 0.4)
sobel_img_5 = sobel_edge(img_array, 0.5)
sobel_img_6 = sobel_edge(img_array, 0.6)   

# Create a new figure for the sobel edge detection images
fig2 = plt.figure(2, figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.imshow(sobel_img_1, cmap='gray')
plt.title('Sobel Edge Detection (Threshold = 0.1)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(sobel_img_2, cmap='gray')
plt.title('Sobel Edge Detection (Threshold = 0.2)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sobel_img_3, cmap='gray')
plt.title('Sobel Edge Detection (Threshold = 0.3)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sobel_img_4, cmap='gray')
plt.title('Sobel Edge Detection (Threshold = 0.4)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(sobel_img_5, cmap='gray')
plt.title('Sobel Edge Detection (Threshold = 0.5)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(sobel_img_6, cmap='gray')
plt.title('Sobel Edge Detection (Threshold = 0.6)')
plt.axis('off')

# Create a new figure for the sobel edge detection diagram
xpoints = np.array([sobel_img_1.sum(), sobel_img_2.sum(), sobel_img_3.sum(), sobel_img_4.sum(), sobel_img_5.sum(), sobel_img_6.sum()])
ypoints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

fig3 = plt.figure(3, figsize=(8, 8))
plt.plot(xpoints, ypoints, 'o', ls = '-')
plt.xlabel("Number of Edges Detected")
plt.ylabel("Threshold Value")
plt.title('Sobel Edge Detection Diagram')


# This was the calculation of the initial LoG edge detection image before adding the threshold parameter
# log_img = log_edge(img_array)

# # Create a new figure for the LoG image
# fig4 = plt.figure(4, figsize=(8, 8))
# plt.imshow(log_img, cmap='gray')
# plt.title('LoG Edge Detection')
# plt.axis('off')


# Calculate the modified LoG edge detection images for various thresholds
log_img_1 = log_edge(img_array, 0)
log_img_2 = log_edge(img_array, 0.2)
log_img_3 = log_edge(img_array, 0.4)
log_img_4 = log_edge(img_array, 0.6)
log_img_5 = log_edge(img_array, 0.8)
log_img_6 = log_edge(img_array, 1)

# Create a new figure for the modified LoG images
fig5 = plt.figure(5, figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.imshow(log_img_1, cmap='gray')
plt.title('Modified LoG Edge Detection (Threshold = 0)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(log_img_2, cmap='gray')
plt.title('Modified LoG Edge Detection (Threshold = 0.2)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(log_img_3, cmap='gray')
plt.title('Modified LoG Edge Detection (Threshold = 0.4)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(log_img_4, cmap='gray')
plt.title('Modified LoG Detection (Threshold = 0.6)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(log_img_5, cmap='gray')
plt.title('Modified LoG Edge Detection (Threshold = 0.8)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(log_img_6, cmap='gray')
plt.title('Modified LoG Edge Detection (Threshold = 1)')
plt.axis('off')


# # Calculate the Hough transform images with Sobel edge detection for various V_min values
V_min1 = [600, 800, 1200, 1400, 1600]
R_max = 320
R_min = 280
dim = np.array([200, 150, 5])

for i in V_min1:
    # Create a new figure for the detected circles
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the original image
    ax.imshow(img_array, cmap='gray') 
    ax.set_xlim(0, img_array.shape[1])
    ax.set_ylim(img_array.shape[0], 0)  #Flip y-axis to match image coordinates
    plt.title('Hough Detected Circles with Sobel operator for V_min = {}'.format(i))
    plt.axis('off')

    # Draw circles using matplotlib.patches.Circle
    centers, radii = circ_hough(sobel_img_5, R_max, R_min, dim, i)
    for (x, y), r in zip(centers, radii):
        circle = plt.Circle((x, y), r, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)


# Calculate the Hough transform images with LoG edge detection for various V_min values
V_min2 = [1000, 1200, 1400, 1600, 1800]

for i in V_min2:
    # Create a new figure for the detected circles
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the original image
    ax.imshow(img_array, cmap='gray') 
    ax.set_xlim(0, img_array.shape[1])
    ax.set_ylim(img_array.shape[0], 0)  #Flip y-axis to match image coordinates
    plt.title('Hough Detected Circles with LoG operator for V_min = {}'.format(i))
    plt.axis('off')

    # Draw circles using matplotlib.patches.Circle
    centers, radii = circ_hough(log_img_4, R_max, R_min, dim, i)
    for (x, y), r in zip(centers, radii):
        circle = plt.Circle((x, y), r, fill=False, color='red', linewidth=2)
        ax.add_patch(circle)


plt.show()