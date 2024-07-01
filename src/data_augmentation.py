import os
import cv2
import albumentations as A
import numpy as np
from glob import glob

def augment_data(images_path, labels_path, output_images_path, output_labels_path, num_augmentations=2):
    # Define the augmentation pipeline with appropriate bounding box parameters
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Blur(blur_limit=3, p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_area=0.0, min_visibility=0.0))

    # Function to load image and corresponding labels
    def load_image_and_labels(image_path, labels_path):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        label_file = os.path.join(labels_path, os.path.basename(image_path).replace('.jpg', '.txt'))
        with open(label_file, 'r') as f:
            boxes = []
            class_labels = []
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_min = (x_center - width / 2) * w
                y_min = (y_center - height / 2) * h
                x_max = (x_center + width / 2) * w
                y_max = (y_center + height / 2) * h
                boxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(int(class_id))
        
        return image, boxes, class_labels, h, w

    # Function to convert bounding boxes back to YOLO format
    def convert_bboxes_to_yolo(bboxes, img_h, img_w):
        yolo_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h
            yolo_bboxes.append([x_center, y_center, width, height])
        return yolo_bboxes

    # Function to clip bounding box values to be within image dimensions
    def clip_bboxes(bboxes, img_h, img_w):
        clipped_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0, min(x_min, img_w))
            y_min = max(0, min(y_min, img_h))
            x_max = max(0, min(x_max, img_w))
            y_max = max(0, min(y_max, img_h))
            if x_max > x_min and y_max > y_min:
                clipped_bboxes.append([x_min, y_min, x_max, y_max])
        return clipped_bboxes

    # Function to save augmented image and labels
    def save_augmented_data(image, bboxes, class_labels, output_image_path, output_label_path):
        cv2.imwrite(output_image_path, image)
        with open(output_label_path, 'w') as f:
            for bbox, class_label in zip(bboxes, class_labels):
                f.write(f"{class_label} {' '.join(map(str, bbox))}\n")

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Augment and save
    image_files = glob(os.path.join(images_path, '*.jpg'))
    for image_file in image_files:
        # Load the image and its labels
        image, bboxes, class_labels, img_h, img_w = load_image_and_labels(image_file, labels_path)
        
        for i in range(num_augmentations):
            # Ensure bounding boxes are valid before augmentation
            bboxes = clip_bboxes(bboxes, img_h, img_w)
            
            # Apply the augmentations
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"Augmentation failed for {image_file} with error: {e}")
                continue

            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']
            
            # Ensure bounding boxes are valid after augmentation
            augmented_bboxes = clip_bboxes(augmented_bboxes, img_h, img_w)
            
            # Convert bounding boxes back to YOLO format
            yolo_bboxes = convert_bboxes_to_yolo(augmented_bboxes, img_h, img_w)
            
            # Define the output paths
            output_image_path = os.path.join(output_images_path, f"{os.path.splitext(os.path.basename(image_file))[0]}_aug_{i}.jpg")
            output_label_path = os.path.join(output_labels_path, f"{os.path.splitext(os.path.basename(image_file))[0]}_aug_{i}.txt")
            
            # Save the augmented data
            save_augmented_data(augmented_image, yolo_bboxes, augmented_class_labels, output_image_path, output_label_path)

    print("Data augmentation completed.")


images_path = 'yolo_data_v2/train/images'  # Path to the training images
labels_path = 'yolo_data_v2/train/labels'  # Path to the training labels
output_images_path = 'yolo_data_v2/train/images'  # Output path for augmented images
output_labels_path = 'yolo_data_v2/train/labels'  # Output path for augmented labels

augment_data(images_path, labels_path, output_images_path, output_labels_path)
