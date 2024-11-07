# template matching using cv2.matchtemplate method 

import cv2
import matplotlib.pyplot as plt
import numpy as np # Make sure numpy is imported

# Load the scene image and the template image in grayscale
scene_image = cv2.imread('/content/drive/MyDrive/1d.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('/content/drive/MyDrive/liamm.png', cv2.IMREAD_GRAYSCALE)

# Ensure template is smaller than scene image
# If template is larger, resize it
if template.shape[0] > scene_image.shape[0] or template.shape[1] > scene_image.shape[1]:
    template = cv2.resize(template, (scene_image.shape[1] // 2, scene_image.shape[0] // 2))  # Resize by half

# Get dimensions of the template image (after potential resize)
template_height, template_width = template.shape

# Perform template matching
result = cv2.matchTemplate(scene_image, template, cv2.TM_CCOEFF_NORMED)

# Set a threshold to identify match locations
threshold = 0.8
locations = np.where(result >= threshold)

# Draw rectangles around matched locations
for pt in zip(*locations[::-1]):  # Switch columns and rows
    cv2.rectangle(scene_image, pt, (pt[0] + template_width, pt[1] + template_height), (10, 0, 200), 2)

# Display the results
plt.figure(figsize=(10, 5))
plt.title("Detected Template Matches")
plt.imshow(scene_image, cmap='gray')
plt.axis('off')
plt.show()

# template matching usiong SIFT
import cv2
import matplotlib.pyplot as plt

# Load the template and scene images in grayscale
template = cv2.imread('/content/drive/MyDrive/liamm.png', cv2.IMREAD_GRAYSCALE)
scene_image = cv2.imread('/content/drive/MyDrive/1d.png', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(template, None)
keypoints2, descriptors2 = sift.detectAndCompute(scene_image, None)

# Initialize the BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors between the two images
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 50 matches
matched_image = cv2.drawMatches(template, keypoints1, scene_image, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the results
plt.figure(figsize=(15, 10))
plt.title("Keypoint Matches using SIFT")
plt.imshow(matched_image)
plt.axis('off')
plt.show()

