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



# ------------------------wit perforace ---------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to add Gaussian noise to the image
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy_image = cv2.add(image.astype('float32'), noise)
    return np.clip(noisy_image, 0, 255).astype('uint8')

# Function to calculate SNR
def calculate_snr(original, processed):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - processed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Load a grayscale image
image = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noisy_image = add_gaussian_noise(image)

# Apply filters
average_filter = cv2.blur(noisy_image, (5, 5))
gaussian_filter = cv2.GaussianBlur(noisy_image, (5, 5), 0)
median_filter = cv2.medianBlur(noisy_image, 5)

# Calculate SNR for each filter
snr_noisy = calculate_snr(image, noisy_image)
snr_average = calculate_snr(image, average_filter)
snr_gaussian = calculate_snr(image, gaussian_filter)
snr_median = calculate_snr(image, median_filter)

# Display results
filters = ['Original', 'Noisy', 'Averaging', 'Gaussian', 'Median']
images = [image, noisy_image, average_filter, gaussian_filter, median_filter]

plt.figure(figsize=(15, 10))
for i, (img, title) in enumerate(zip(images, filters)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Print SNR values
print(f"SNR of noisy image: {snr_noisy:.2f} dB")
print(f"SNR after Averaging Filter: {snr_average:.2f} dB")
print(f"SNR after Gaussian Filter: {snr_gaussian:.2f} dB")
print(f"SNR after Median Filter: {snr_median:.2f} dB")

# --------------------------------------------------------------------------------------------------------------------------

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





# ------------------------------------------------------wit perforace-------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (original clean image)
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

# Function to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(original_image, noisy_image, filtered_image):
    signal = np.mean(original_image)
    noise = np.std(np.abs(noisy_image - filtered_image))
    snr = signal / noise
    return snr

# Calculate SNR for Gaussian and Median smoothed images
snr_gaussian = calculate_snr(image, noisy_image, gaussian_smoothed)
snr_median = calculate_snr(image, noisy_image, median_smoothed)

# Function to calculate Edge Preservation Index (EPI)
def calculate_epi(noisy_edges, filtered_edges):
    # EPI is calculated as the ratio of common edges to the total edges in the noisy image
    intersection = np.sum((noisy_edges > 0) & (filtered_edges > 0))
    total_edges_noisy = np.sum(noisy_edges > 0)
    epi = intersection / total_edges_noisy if total_edges_noisy > 0 else 0
    return epi

# Calculate EPI for Gaussian and Median smoothed images
epi_gaussian = calculate_epi(sobel_edges_noisy, sobel_edges_gaussian)
epi_median = calculate_epi(sobel_edges_noisy, sobel_edges_median)

# Display results
print(f"SNR for Gaussian Filter: {snr_gaussian:.4f}")
print(f"SNR for Median Filter: {snr_median:.4f}")
print(f"EPI for Gaussian Filter: {epi_gaussian:.4f}")
print(f"EPI for Median Filter: {epi_median:.4f}")

# Display the images and Sobel edge results
plt.figure(figsize=(12, 10))

# Noisy Image
plt.subplot(3, 2, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')

# Gaussian Smoothed
plt.subplot(3, 2, 2)
plt.title(f'Gaussian Smoothed (SNR={snr_gaussian:.4f}, EPI={epi_gaussian:.4f})')
plt.imshow(gaussian_smoothed, cmap='gray')

# Sobel on Noisy Image
plt.subplot(3, 2, 3)
plt.title('Sobel on Noisy Image')
plt.imshow(sobel_edges_noisy, cmap='gray')

# Sobel on Gaussian Smoothed
plt.subplot(3, 2, 4)
plt.title(f'Sobel on Gaussian Smoothed (EPI={epi_gaussian:.4f})')
plt.imshow(sobel_edges_gaussian, cmap='gray')

# Sobel on Median Smoothed
plt.subplot(3, 2, 5)
plt.title(f'Sobel on Median Smoothed (EPI={epi_median:.4f})')
plt.imshow(sobel_edges_median, cmap='gray')

plt.tight_layout()
plt.show()
# ---------------------------------------------------------
