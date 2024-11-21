import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt

# Function to load and preprocess dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):  # Ensure it's a folder
            if folder_name not in label_map:
                label_map[folder_name] = label_counter
                label_counter += 1

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:  # Skip invalid images
                    image = cv2.resize(image, (100, 100))  # Uniform size
                    images.append(image.flatten())
                    labels.append(label_map[folder_name])

    return np.array(images), np.array(labels), label_map

# Path to your labeled dataset
dataset_path = r"C:\Users\Gowsika\Downloads\pca\labels\lfw-deepfunneled\lfw-deepfunneled"

# Load the dataset
images, labels, label_map = load_dataset(dataset_path)

# Apply PCA
num_components = 100  # Number of principal components
pca = PCA(n_components=num_components, whiten=True)
images_pca = pca.fit_transform(images)

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(images_pca, labels)

# Initialize video capture
video_capture = cv2.VideoCapture(r"C:\Users\Gowsika\Downloads\pca\inner\8762656-uhd_3840_2160_25fps (1).mp4")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)
        face_pca = pca.transform(face_resized)

        prediction = knn.predict(face_pca)
        predicted_label = list(label_map.keys())[list(label_map.values()).index(prediction[0])]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert BGR (OpenCV format) to RGB (Matplotlib format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display using Matplotlib
    plt.imshow(rgb_frame)
    plt.axis('off')  # Hide axes for clarity
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
