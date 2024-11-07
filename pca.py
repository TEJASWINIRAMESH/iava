import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow


# import cv2

# # Path to the image file
# frame = r"C:\Users\PC\Downloads\images.jpeg"

# # Read the image
# image = cv2.imread(frame)

# # Display the image
# cv2.imshow("Image", image)
# Path to the dataset
dataset_path = '/content/drive/MyDrive/archive'

# Initialize variables
images = []
labels = []
label_dict = {}
current_label = 0

# Load images and labels
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    # Assign a label for each person
    label_dict[current_label] = person_name
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))  # Resize to a consistent size
        images.append(img)
        labels.append(current_label)

    current_label += 1

# Check if images were loaded
if not images or not labels:
    print("No images or labels found in the dataset.")
    exit()

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Create the EigenFace Recognizer model
face_recognizer = cv2.face.EigenFaceRecognizer_create()

# Train the recognizer on the dataset
face_recognizer.train(images, labels)

# Start capturing video
cap = cv2.VideoCapture("/content/drive/MyDrive/210574_tiny.mp4")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (200, 200))  # Resize to match training data

        # Predict the label
        label, confidence = face_recognizer.predict(face_img_resized)
        name = label_dict.get(label, "Unknown")
        cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2_imshow(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
