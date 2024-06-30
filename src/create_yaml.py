# Code not working as expected. Do not use this code. todo: Fix the code. #
import os
import yaml

# Paths
yolo_data_path = 'yolo_data'

# Data for YAML
data_yaml = {
    'train': '../train/images',
    'val': '../valid/images',
    'test': '../test/images',
    'nc': 2,
    'names': ['obstacle', 'turtlebot']
}

# Write the YAML file to yolo_data_path
with open(os.path.join(yolo_data_path, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print("data.yaml file created successfully!")
