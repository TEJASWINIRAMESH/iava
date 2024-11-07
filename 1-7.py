# 1. Simulation and Display of an Image, Negative of an Image (Binary & Gray Scale)

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Binary Negative
binary_negative = cv2.bitwise_not(cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1])
plt.subplot(1, 3, 2)
plt.title('Binary Negative')
plt.imshow(binary_negative, cmap='gray')

# Grayscale Negative
grayscale_negative = 255 - image
plt.subplot(1, 3, 3)
plt.title('Grayscale Negative')
plt.imshow(grayscale_negative, cmap='gray')

plt.show()


# %% [markdown]
# 2. Contrast Stretching, Histogram, and Histogram Equalization

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)
# Contrast Stretching
min_val, max_val = np.min(image), np.max(image)
contrast_stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Histogram and Histogram Equalization
hist_original, bins = np.histogram(image.flatten(), 256, [0, 256])
hist_eq = cv2.equalizeHist(image)
hist_equalized, bins_eq = np.histogram(hist_eq.flatten(), 256, [0, 256])

plt.figure(figsize=(12, 8))

# Original Image and Histogram
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(2, 3, 2)
plt.title('Histogram (Original)')
plt.plot(hist_original)

# Contrast Stretched Image and its Histogram
plt.subplot(2, 3, 3)
plt.title('Contrast Stretched Image')
plt.imshow(contrast_stretched, cmap='gray')

# Histogram Equalized Image and its Histogram
plt.subplot(2, 3, 4)
plt.title('Histogram Equalized Image')
plt.imshow(hist_eq, cmap='gray')
plt.subplot(2, 3, 5)
plt.title('Histogram (Equalized)')
plt.plot(hist_equalized)

plt.tight_layout()
plt.show()


# %% [markdown]
# 3. Implementation of Transformations (Scaling, Rotation, Translation)
# python
# Copy code
# 

# %%
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Scaling
scaled = cv2.resize(image, None, fx=1.5, fy=1.5)

# Rotation
center = (image.shape[1] // 2, image.shape[0] // 2)
rot_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

# Translation
tx, ty = 50, 50  # Translate by 50 pixels
trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(image, trans_matrix, (image.shape[1], image.shape[0]))

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Scaled Image')
plt.imshow(scaled, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Rotated Image')
plt.imshow(rotated, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Translated Image')
plt.imshow(translated, cmap='gray')
plt.show()


# %% [markdown]
# 4. Implementation of Relationships between Pixels (Gradient)

# %%
image = cv2.imread('/content/drive/MyDrive/image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel operator for gradient calculation
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(grad_x, grad_y)

plt.title('Gradient Magnitude')
plt.imshow(gradient_magnitude, cmap='gray')
plt.show()


# %% [markdown]
# 5. Display of Bit Planes of an Image

# %%
bit_planes = [(image >> i) & 1 for i in range(8)]

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.title(f'Bit Plane {i}')
    plt.imshow(bit_planes[i] * 255, cmap='gray')
plt.show()

# %% [markdown]
# 6. Display of FFT (1-D & 2-D) of an Image

# %%
# 2-D FFT
fft_2d = np.fft.fftshift(np.fft.fft2(image))
magnitude_2d = 20 * np.log(np.abs(fft_2d))

# 1-D FFT along a specific row
fft_1d = np.fft.fftshift(np.fft.fft(image[image.shape[0] // 2, :]))
magnitude_1d = 20 * np.log(np.abs(fft_1d))

plt.figure(figsize=(12, 6))

# 2D FFT
plt.subplot(1, 2, 1)
plt.title('2-D FFT')
plt.imshow(magnitude_2d, cmap='gray')

# 1D FFT
plt.subplot(1, 2, 2)
plt.title('1-D FFT')
plt.plot(magnitude_1d)
plt.show()

# %% [markdown]
# 7. Computation of Mean, Standard Deviation, Correlation Coefficient

# %%
# Mean and Standard Deviation
mean = np.mean(image)
std_dev = np.std(image)

# Correlation Coefficient with another image (assuming we have a second image)
second_image = cv2.imread('/content/drive/MyDrive/image2.jpg', cv2.IMREAD_GRAYSCALE)
if image.shape != second_image.shape:
    second_image = cv2.resize(second_image, (image.shape[1], image.shape[0]))

correlation_coeff = np.corrcoef(image.flatten(), second_image.flatten())[0, 1]

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Correlation Coefficient:", correlation_coeff)
