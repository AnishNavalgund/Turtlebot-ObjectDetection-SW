import os
import cv2
import matplotlib.pyplot as plt
from glob import glob
import random

# Function to load labels
def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = []
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append((int(class_id), x_center, y_center, width, height))
    return labels

# Function to draw bounding boxes on the image
def draw_bboxes(image, labels):
    h, w = image.shape[:2]
    # Use different colors for different classes
    colors = {}
    for class_id, x_center, y_center, width, height in labels:
        if class_id not in colors:
            colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)
        color = colors[class_id]
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        image = cv2.putText(image, str(class_id), (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

# Function to visualize images
def visualize_images(images_path, labels_path, output_vis_path):
    os.makedirs(output_vis_path, exist_ok=True)
    image_files = glob(os.path.join(images_path, '*.jpg'))
    for image_file in image_files:
        image = cv2.imread(image_file)
        label_file = os.path.join(labels_path, os.path.basename(image_file).replace('.jpg', '.txt'))
        labels = load_labels(label_file)
        image_with_bboxes = draw_bboxes(image, labels)
        
        # Save the image with bounding boxes
        output_image_path = os.path.join(output_vis_path, os.path.basename(image_file))
        cv2.imwrite(output_image_path, image_with_bboxes)
        print(f"Saved visualized image: {output_image_path}")

# Paths for visualization
visualize_images_path = 'output/visualized/'  # Path to save images with bounding boxes drawn
output_images_path = 'yolo_data/val/images'  # Path to the test images
output_labels_path = 'yolo_data/val/labels'  # Path to the test labels

# Visualize all the augmented images with bounding boxes
visualize_images(output_images_path, output_labels_path, visualize_images_path)
