import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Paths
raw_data_path = 'raw_data'
yolo_data_path = 'yolo_data'
yolo_train_path = os.path.join(yolo_data_path, 'train')
yolo_val_path = os.path.join(yolo_data_path, 'val')
yolo_test_path = os.path.join(yolo_data_path, 'test')

# Create necessary directories
os.makedirs(os.path.join(yolo_train_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(yolo_train_path, 'labels'), exist_ok=True)
os.makedirs(os.path.join(yolo_val_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(yolo_val_path, 'labels'), exist_ok=True)
os.makedirs(os.path.join(yolo_test_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(yolo_test_path, 'labels'), exist_ok=True)

# Get all image and label file paths
images = glob(os.path.join(raw_data_path, 'images', '*.jpg'))
labels = glob(os.path.join(raw_data_path, 'labels', '*.txt'))

# Ensure each image has a corresponding label
images.sort()
labels.sort()

# Split data into train, val, test
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)

# Function to move files
def move_files(images, labels, dest_images, dest_labels):
    for img, lbl in zip(images, labels):
        shutil.copy(img, dest_images)
        shutil.copy(lbl, dest_labels)

# Move files to respective directories
move_files(train_images, train_labels, os.path.join(yolo_train_path, 'images'), os.path.join(yolo_train_path, 'labels'))
move_files(val_images, val_labels, os.path.join(yolo_val_path, 'images'), os.path.join(yolo_val_path, 'labels'))
move_files(test_images, test_labels, os.path.join(yolo_test_path, 'images'), os.path.join(yolo_test_path, 'labels'))

print("Dataset split and moved successfully!")
