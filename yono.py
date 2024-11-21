

# %%
from ultralytics import YOLO

# Step 1: Load YOLOv8 pretrained model (replace with your desired model, e.g., 'yolov8n')
model = YOLO('yolov8n.pt')  # Pretrained model

# Step 2: Specify the path to your dataset configuration file (in YOLO format)
config_path = r"C:\Users\dell\Desktop\yolo\config.yaml"  # Update this with the path to your dataset.yaml

# Step 3: Train the model with custom dataset and configurations
model.train(
    data=config_path,  # Path to your dataset configuration file
    epochs=300,                   # Number of epochs to train
    patience=300,                 # Patience for early stopping
    batch=16,                     # Batch size
    lr0=0.001,                    # Initial learning rate
    augment=True                   # Enable data augmentations
)

# Step 4: Validate the model to get performance metrics after training
print("\n--- Running Validation ---")
results = model.val()  # This will run the validation on your dataset

# Step 5: Print validation results (mean Average Precision, precision, recall, etc.)
print("\n--- Validation Results ---")
print(results)  # Results contain metrics like mAP, precision, recall

# Step 6: Test the model on a specific test image after training
test_image_path = r"C:\Users\dell\Desktop\yolo\images\train\download.jpg"

# Run inference/prediction on a test image using the trained model
results_pretrained = model.predict(test_image_path, save=True)

# ----------------cofig yaml-----------------------
# path: C:\Users\dell\Desktop\yolo
# train: images/train
# val: images/val

# #Classes
# names:
#   0: person
#   1: car


# ----------------------small code -------------------------------------

from ultralytics import YOLO

# Step 1: Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # Replace with the correct path or model name if you have a custom model
test_image_path = "/content/10815824_2997e03d76.jpg"

# Run inference/prediction on a test image using the trained model
results_pretrained = model.predict(test_image_path, save=True)

# Display predictions (assuming results_pretrained is an object, not a list)
results_pretrained[0].show()  # Display predictions for the first image in the result
