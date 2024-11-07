# 8. Sobel Edge Detection
# The Sobel filter is used to compute the gradient magnitude of an image in both horizontal and vertical directions. This can help detect edges.
# 

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel filter kernels for horizontal and vertical edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Convert back to uint8
sobel_magnitude = np.uint8(np.absolute(sobel_magnitude))

# Display the images
plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Sobel X')
plt.imshow(sobel_x, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Sobel Y')
plt.imshow(sobel_y, cmap='gray')

plt.show()


# %% [markdown]
# 9. Prewitt Edge Detection
# The Prewitt filter is similar to the Sobel filter and is used to detect horizontal and vertical gradients in an image.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Prewitt filter kernels for horizontal and vertical edges
prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

# Apply convolution using filter2D
prewitt_x_img = cv2.filter2D(image, -1, prewitt_x)
prewitt_y_img = cv2.filter2D(image, -1, prewitt_y)

# Convert to float32 before calculating magnitude
prewitt_x_img = np.float32(prewitt_x_img)
prewitt_y_img = np.float32(prewitt_y_img)

# Gradient magnitude
prewitt_magnitude = cv2.magnitude(prewitt_x_img, prewitt_y_img)

# Convert back to uint8
prewitt_magnitude = np.uint8(np.absolute(prewitt_magnitude))

# Display the images
plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Prewitt X')
plt.imshow(prewitt_x_img, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Prewitt Y')
plt.imshow(prewitt_y_img, cmap='gray')

plt.show()


# %% [markdown]
# 10. Canny Edge Detection
# The Canny edge detection algorithm uses two threshold values to detect edges. You can fine-tune the thresholds for better edge detection.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
lower_threshold = 100
upper_threshold = 200
edges_canny = cv2.Canny(image, lower_threshold, upper_threshold)

# Display the original and edge-detected images
plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Canny Edges')
plt.imshow(edges_canny, cmap='gray')

plt.show()


# %% [markdown]
# 11. Edge Detection on Noisy Image

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noise = np.random.normal(0, 25, image.shape)  # Mean = 0, Stddev = 25
noisy_image = np.uint8(np.clip(image + noise, 0, 255))

# Apply Averaging filter
average_filtered = cv2.blur(noisy_image, (5, 5))

# Apply Gaussian filter
gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 1)

# Apply Median filter
median_filtered = cv2.medianBlur(noisy_image, 5)

# Display results
plt.figure(figsize=(10, 8))

# Noisy Image
plt.subplot(3, 2, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')

# Averaging Filter
plt.subplot(3, 2, 2)
plt.title('Averaging Filter')
plt.imshow(average_filtered, cmap='gray')

# Gaussian Filter
plt.subplot(3, 2, 3)
plt.title('Gaussian Filter')
plt.imshow(gaussian_filtered, cmap='gray')

# Median Filter
plt.subplot(3, 2, 4)
plt.title('Median Filter')
plt.imshow(median_filtered, cmap='gray')

plt.tight_layout()
plt.show()
