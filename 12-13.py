# 12: Apply Averaging, Gaussian, and Median Filters to a Noisy Image

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
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Averaging Filter')
plt.imshow(average_filtered, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Gaussian Filter')
plt.imshow(gaussian_filtered, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Median Filter')
plt.imshow(median_filtered, cmap='gray')

plt.tight_layout()
plt.show()


# %% [markdown]
# 13: Apply Smoothing Filters Before Edge Detection

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noise = np.random.normal(0, 25, image.shape)  # Mean = 0, Stddev = 25
noisy_image = np.uint8(np.clip(image + noise, 0, 255))

# Apply Gaussian filter for smoothing
gaussian_smoothed = cv2.GaussianBlur(noisy_image, (5, 5), 1)

# Apply Median filter for smoothing
median_smoothed = cv2.medianBlur(noisy_image, 5)

# Edge detection on original noisy image (without smoothing)
sobel_x_noisy = cv2.Sobel(noisy_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_noisy = cv2.Sobel(noisy_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges_noisy = np.uint8(np.absolute(cv2.magnitude(sobel_x_noisy, sobel_y_noisy)))

# Edge detection on Gaussian smoothed image
sobel_x_gaussian = cv2.Sobel(gaussian_smoothed, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_gaussian = cv2.Sobel(gaussian_smoothed, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges_gaussian = np.uint8(np.absolute(cv2.magnitude(sobel_x_gaussian, sobel_y_gaussian)))

# Edge detection on Median smoothed image
sobel_x_median = cv2.Sobel(median_smoothed, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_median = cv2.Sobel(median_smoothed, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges_median = np.uint8(np.absolute(cv2.magnitude(sobel_x_median, sobel_y_median)))

# Display results
plt.figure(figsize=(10, 10))

# Noisy Image
plt.subplot(3, 2, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')

# Gaussian Smoothed
plt.subplot(3, 2, 2)
plt.title('Gaussian Smoothed')
plt.imshow(gaussian_smoothed, cmap='gray')

# Sobel on Noisy Image
plt.subplot(3, 2, 3)
plt.title('Sobel on Noisy Image')
plt.imshow(sobel_edges_noisy, cmap='gray')

# Sobel on Gaussian Smoothed
plt.subplot(3, 2, 4)
plt.title('Sobel on Gaussian Smoothed')
plt.imshow(sobel_edges_gaussian, cmap='gray')

# Sobel on Median Smoothed
plt.subplot(3, 2, 5)
plt.title('Sobel on Median Smoothed')
plt.imshow(sobel_edges_median, cmap='gray')

plt.tight_layout()
plt.show()
